"""
(venv_py38) python -m trainers.train_conDATripletNews --batch_size 256 --max_epochs 1 --target_domain bbc,guardian --base_model blip-2 --loss_type simclr
"""
import sys
sys.path.append('.')


import os
import random
import logging
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from multiprocessing import Process

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from itertools import cycle
from functools import reduce

from models.VDT import ProjectionMLP, MLLMClassificationHead, VDTModule  
from models.vae_model import *

from datasets.newsCLIPpingsDataset import get_dataloader
from configs.configNews import ConfigNews
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from utils.helper import accuracy_at_eer, compute_auc


DISTRIBUTED_FLAG = False

# Logger
logger = logging.getLogger()
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="[%(asctime)s]:[%(processName)-11s]" + "[%(levelname)-s]:[%(name)s] %(message)s",
)

    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_distributed(port=29500):
    if not DISTRIBUTED_FLAG:
        return 0, 1   # indicating that distributed training is not configured

    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1   # If it's not feasible, it returns default values (0, 1).

    # if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
    #     from mpi4py import MPI   # cannot be installed
    #     mpi_rank = MPI.COMM_WORLD.Get_rank()
    #     mpi_size = MPI.COMM_WORLD.Get_size()
    #
    #     os.environ["MASTER_ADDR"] = '127.0.0.1'
    #     os.environ["MASTER_PORT"] = str(port)
    #
    #     dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
    #     return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def distributed():
    # return dist.is_available() and dist.is_initialized()
    return False ## only because I want to use one GPU


def summary(model: nn.Module, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:   # logits: [bs, 2], labels: [bs, ]
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:   # logits: [bs,]
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def return_classification(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:   # logits: [bs, 2], labels: [bs, ]
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:   # logits: [bs,]
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return classification.cpu(), labels.cpu()



def train(cfg, model: nn.Module, optimizer, device: str, src_loader: DataLoader,
          tgt_loader: DataLoader, summary_writer: SummaryWriter, desc='Train', lambda_1=2,lambda_2=0.5, lambda_3=1):
    model.train()

    src_train_accuracy = 0
    tgt_train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    if len(src_loader) == len(tgt_loader):
        double_loader = enumerate(zip(src_loader, tgt_loader))
    elif len(src_loader) < len(tgt_loader):
        print("Src smaller than Tgt")
        double_loader = enumerate(zip(cycle(src_loader), tgt_loader))   # zip() only iterates over the smallest iterable
    else:
        double_loader = enumerate(zip(src_loader, cycle(tgt_loader)))
    is_main_process = not distributed() or dist.get_rank() == 0
    with tqdm(double_loader, desc=desc, disable=not is_main_process) as loop:
        torch.cuda.empty_cache()
        for i, (src_data, tgt_data) in loop:
            # (1) Prepare the data inputs and labels          
            src_emb, src_labels = src_data["original_multimodal_emb"], src_data["original_label"]
            src_emb, src_labels = src_emb.to(device), src_labels.to(device)
            batch_size = src_emb.shape[0]

            tgt_emb, tgt_labels = tgt_data["original_multimodal_emb"], tgt_data["original_label"]
            tgt_emb, tgt_labels = tgt_emb.to(device), tgt_labels.to(device)

            src_domain_label, tgt_domain_label = src_data["domain_label"], tgt_data['domain_label']
            
            # (2) optimizer set to zero_grad()
            optimizer.zero_grad()

            # (3) model is the overall model, including MLLMClsHead and the projection mlp,
            # model.forward() will address the loss computation on each module
            output_dic = model(src_emb, tgt_emb, src_labels, tgt_labels)

            loss = output_dic.total_loss

            # (4) Back-propagation: compute the gradients
            loss.backward()

            # (5) Update the model parameters
            optimizer.step()

            # (6) Evaluate
            src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
            src_train_accuracy += src_batch_accuracy
            tgt_batch_accuracy = accuracy_sum(output_dic.tgt_logits, tgt_labels)
            tgt_train_accuracy += tgt_batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item()

            loop.set_postfix(loss=loss.item(), src_acc=src_train_accuracy / train_epoch_size,
                             tgt_acc=tgt_train_accuracy / train_epoch_size,
                             src_LCE_loss=output_dic.src_ce_loss.item(),
                            )            
    return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss,
    }


def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    bbc_validation_accuracy = 0
    guardian_validation_accuracy = 0
    usa_today_validation_accuracy = 0
    washington_post_validation_accuracy = 0
    validation_epoch_size = 0
    bbc_epoch_size = 0
    guardian_epoch_size = 0
    usa_today_epoch_size = 0
    washington_post_epoch_size = 0
    validation_loss = 0


    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc) as loop, torch.no_grad():
        targets = []
        outputs = []
        domain_labels_list = []
        output_logits = []
        results_list = []
        for example in loop:
            losses = []
            logit_votes = []
            pre_output_votes =[]
            # print(example)
            for data in example:
                # print(data)
                emb, labels, domain_labels = data["original_multimodal_emb"], data["original_label"], data["domain_label"]
                emb, labels = emb.to(device), labels.to(device)
                batch_size = emb.shape[0]

                F_t, _, _, _ = model(src_emb=None, tgt_emb=emb, src_labels=None, tgt_labels=labels, mode='test') 
                logits = model.model(F_t)
                loss, softmax_logits = model.model.compute_loss(logits, labels=labels), model.model.compute_softmax_logits(logits)
                losses.append(loss)
                logit_votes.append(logits)

            bbc_ind = np.array([i for i, d_label in enumerate(domain_labels) if d_label=="bbc"])
            guardian_ind = np.array([i for i, d_label in enumerate(domain_labels) if d_label=="guardian"])
            usa_today_ind = np.array([i for i, d_label in enumerate(domain_labels) if d_label=="usa_today"])
            washington_post_ind = np.array([i for i, d_label in enumerate(domain_labels) if d_label=="washington_post"])

            loss = torch.stack(losses).mean(dim=0)
            out_logits = torch.stack(logit_votes).mean(dim=0)
            
            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            bbc_batch_accuracy = accuracy_sum(logits[bbc_ind], labels[bbc_ind])
            bbc_validation_accuracy += bbc_batch_accuracy
            guardian_batch_accuracy = accuracy_sum(logits[guardian_ind], labels[guardian_ind])
            guardian_validation_accuracy += guardian_batch_accuracy
            usa_today_batch_accuracy = accuracy_sum(logits[usa_today_ind], labels[usa_today_ind])
            usa_today_validation_accuracy += usa_today_batch_accuracy
            washington_post_batch_accuracy = accuracy_sum(logits[washington_post_ind], labels[washington_post_ind])
            washington_post_validation_accuracy += washington_post_batch_accuracy
            
            bbc_epoch_size += bbc_ind.shape[0]
            guardian_epoch_size += guardian_ind.shape[0]
            usa_today_epoch_size += usa_today_ind.shape[0]
            washington_post_epoch_size += washington_post_ind.shape[0]

            classifications, labels = return_classification(out_logits, labels)
            targets.append(labels)
            outputs.append(classifications)
            output_logits.append(logits.cpu())
            domain_labels_list += list(domain_labels)

            loop.set_postfix(loss=loss.item(), acc="{:.4f}".format(validation_accuracy / validation_epoch_size))

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        output_logits = np.concatenate(output_logits)
        pred_logits = output_logits[np.arange(output_logits.shape[0]), (1-targets).flatten()]
        accuracy, eer, eer_threshold = accuracy_at_eer(targets, pred_logits)
        print(f"Accuracy at EER: {accuracy}")
        print(f"EER: {eer}")
        print(f"Threshold at EER: {eer_threshold}")
        auc_score = compute_auc(targets, outputs)
        print(f"AUC score: {auc_score}")
        f1 = f1_score(targets, outputs, average='weighted', zero_division=0)
        f1_real, f1_fake = f1_score(targets, outputs, average=None)
        Acc = accuracy_score(targets, outputs, normalize=True, sample_weight=None)
        cls_report = classification_report(targets, outputs, digits=4, zero_division=0)
        cm = confusion_matrix(targets, outputs)    # confuse matrix
        print(f"f1: {f1}")
        print(f"Acc: {Acc}")
        print(f"f1_real: {f1_real}")
        print(f"f1_fake: {f1_fake}")
        print(cls_report)
        print(cm)
    
    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }, {'f1': f1,'Acc': Acc, 'f1_real': f1_real, 'f1_fake': f1_fake}

 
        
def test_time_adaptation_trainvae(model, test_loader):
    model.eval()   # Set other modules to eval mode
    feature_model = model.feature_model
    classifier = model.model

    pseudo_labels, UncertaintyWeight, kept_indices = get_pseudo_label(model, test_loader)

    
    for p in feature_model.module_decoder.parameters():
        p.requires_grad = False
    
    learning_rate = 2e-5
    weight_decay = 0
    optimizer_encoder = Adam(feature_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_classifier = Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    for batch_idx, data in enumerate(test_loader):
        emb, true_labels = data["original_multimodal_emb"], data["original_label"]
        labels = pseudo_labels[batch_idx]
        kept_index = kept_indices[batch_idx]
        uncertainty_weight = UncertaintyWeight[batch_idx].detach()[kept_index]
        
        emb, labels, true_labels = emb.to(device), labels.to(device), true_labels.to(device)
        kept_index = kept_index.to(device)
        uncertainty_weight = uncertainty_weight.to(device)
        
        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()
        
        F_t, y_t, mu_t, log_var_t =  model(src_emb=None, tgt_emb=emb, src_labels=None, tgt_labels=labels, mode='test') 
        logits = classifier(F_t)        

        
        ce = F.cross_entropy(logits[kept_index] , labels[kept_index] , reduction='none')
        loss_cls = ce.mean()
        loss_tgt = feature_model.vae_loss_function(y_t[kept_index], emb[kept_index], mu_t[kept_index], log_var_t[kept_index])
        
        loss = loss_cls + loss_tgt
        # (4) Back-propagation: compute the gradients
        loss.backward()   # for mask

        # (5) Update the model parameters
        optimizer_encoder.step()
        optimizer_classifier.step()


def get_pseudo_label(model, test_loader):
    model.eval()
    pseudo_labels = []
    lambda_u = 5.0  # control the intensity of the impact of uncertainty
    threshold = 0.9    
    kept_indices = []
    uncertainty_weight = []

    
    for batch_idx, data in enumerate(test_loader):
        emb, labels = data["original_multimodal_emb"], data["original_label"]
        emb, labels = emb.to(device), labels.to(device)

        z, _, _, sigma = model(src_emb=None, tgt_emb=emb, src_labels=None, tgt_labels=labels, mode='test') 
        logits = model.model(z)
        logits_softmax = F.softmax(logits, dim=1)
        
        max_values = logits_softmax.max(dim=1).values
        uncertainty = sigma.abs().mean(dim=1)
        weights = torch.exp(-lambda_u * uncertainty)
        conf_value = 2 * max_values - weights
        
        classifications, labels = return_classification(logits_softmax, labels)
        
        indices = (conf_value > threshold).nonzero(as_tuple=True)[0]
        kept_indices.append(indices)
        pseudo_labels.append(classifications)
        uncertainty_weight.append(weights)
    
    return pseudo_labels, uncertainty_weight, kept_indices

def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        # torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d


def run(cfg, seed, device):

    model_save_path = cfg.args.model_save_path
    model_save_name = cfg.args.model_save_name
    batch_size = cfg.args.batch_size
    loss_type = cfg.args.loss_type
    max_epochs = cfg.args.max_epochs
    epoch_size = None
    seed = seed   # seed
    token_dropout = None
    large = False
    learning_rate = cfg.args.learning_rate
    weight_decay = 0
    lambda_1 = cfg.args.lambda_1
    lambda_2 = cfg.args.lambda_2
    lambda_3 = cfg.args.lambda_3
    load_from_checkpoint = False
    checkpoint_name = ''

    args = locals()   # returns a dictionary containing the current local symbol table
    rank, world_size = setup_distributed()   # if not set to distributed, rank=0, world_size=1

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    # if device=='cpu':
    #    print("Could not find GPU")
    #    exit()

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    # Set the logs directory
    logdir = os.environ.get("OPENAI_LOGDIR", "logs")
    os.makedirs(logdir, exist_ok=True)

    # Creates a SummaryWriter object with the specified log directory (logdir).
    # The SummaryWriter is typically used for writing TensorBoard logs,
    # which can be visualized to monitor training metrics.
    writer = SummaryWriter(logdir) if rank == 0 else None

    import torch.distributed as dist
    if distributed() and rank > 0:
        # Synchronize processes before moving to the next stage
        dist.barrier()
        
    set_seed(seed)

    # (1) classification MLP
    mllm_cls_head = MLLMClassificationHead(cfg).to(device)
    
    # # (2) projection MLP
    mlp = ProjectionMLP(cfg).to(device)
    
    # (+) domain-invariant feature extractor
    domain_feature = VAE(cfg).to(device)    # BetaVAE

    # (3) the entire VDT framework
    model = VDTModule(model=mllm_cls_head, mlp=mlp, domain_feature=domain_feature, loss_type=loss_type, logger=writer, device=device, 
                                      lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3)
    # one process
    if rank == 0:
        # summary(model)
        if distributed():   # always false
            dist.barrier()

    # more than one processes
    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)

    src_excluded_topic = cfg.args.target_domain.split(',')   # e.g. bbc
    tgt_domain = cfg.args.target_domain.split(',')
    src_excluded_topic.append('bbc')
    tgt_excluded_topic = ['bbc', 'guardian', 'usa_today', 'washington_post']
    for topic in tgt_domain:
        tgt_excluded_topic.remove(topic)   # e.g. ['guardian', 'usa_today', 'washington_post']
    print(f"src_excluded_topic: {src_excluded_topic}")
    print(f"tgt_excluded_topic: {tgt_excluded_topic}")
    # loading data
    src_train_loader, src_train_dataset_size = get_dataloader(cfg, seed=seed, target_domain=src_excluded_topic, shuffle=True, seed_worker=seed_worker, phase="train")
    src_validation_loader, src_validation_dataset_size = get_dataloader(cfg, seed=seed, target_domain=src_excluded_topic, shuffle=False, phase="test")

    tgt_train_loader, tgt_train_dataset_size = get_dataloader(cfg, seed=seed, target_domain=tgt_excluded_topic, shuffle=True, seed_worker=seed_worker, phase="train")
    tgt_validation_loader, tgt_validation_dataset_size = get_dataloader(cfg, seed=seed, target_domain=tgt_excluded_topic, shuffle=False, phase="test")

    print(f"source train dataset size: {src_train_dataset_size}, target train dataset size: {tgt_train_dataset_size}")
    print(f"source validation dataset size: {src_validation_dataset_size}, target validation dataset size: {tgt_validation_dataset_size}")

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)   # count(1) is an infinite iterator

    best_validation_accuracy = 0
    without_progress = 0
    earlystop_epochs = 5
    results_list = []

    # print(">> Building memory from source domain")
    # memory_bank = build_and_save_memory(model, src_train_loader, tgt_train_loader, device)
    # model.set_memory(memory_bank.get_memory())
    
    for epoch in epoch_loop:

        if world_size > 1:
            src_train_loader.sampler.set_epoch(epoch)
            src_validation_loader.sampler.set_epoch(epoch)
            tgt_train_loader.sampler.set_epoch(epoch)
            tgt_validation_loader.sampler.set_epoch(epoch)


        train_metrics = train(cfg, epoch, model, optimizer, device, src_train_loader, tgt_train_loader, writer,
                              f'Epoch {epoch}', lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3)    # , lambda_w=lambda_w

        test_time_adaptation_trainvae(model, tgt_validation_loader)
        ###########################
        validation_metrics,results_dic = validate(model, device,
                                      tgt_validation_loader)  ## we are only using supervision on the source, compatible with ContrastiveLearningAndTripletLossZModule

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)
        

        combined_metrics["train/src_accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        if rank == 0:
            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            # if combined_metrics["validation/accuracy"] > best_validation_accuracy:
            if results_dic['f1'] > best_validation_accuracy:
                without_progress = 0
                best_validation_accuracy = results_dic['f1']   #  combined_metrics["validation/accuracy"]

                model_to_save = mllm_cls_head
                model_to_save = model
                torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model_to_save.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    # args=args
                ),
                    os.path.join(model_save_path, model_save_name)
                )

        without_progress += 1
       
        if without_progress >= earlystop_epochs:
            break


def main(cfg, seed, device):
    # number of process = number of gpus
    nproc = int(subprocess.check_output([sys.executable, '-c', "import torch;"
                                                               "print(torch.cuda.device_count() if torch.cuda.is_available() else 1)"]))
    nproc = 1
    # for machine compatibility

    if nproc > 1:
        print(f'Launching {nproc} processes ...', file=sys.stderr)

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(29500)
        os.environ['WORLD_SIZE'] = str(nproc)
        os.environ['OMP_NUM_THREAD'] = str(1)
        subprocesses = []

        for i in range(nproc):
            os.environ['RANK'] = str(i)
            os.environ['LOCAL_RANK'] = str(i)
            process = Process(target=run, kwargs=vars(cfg.args))
            process.start()
            subprocesses.append(process)

        for process in subprocesses:
            process.join()
    else:
        run(cfg, seed, device)   # get a dictionary of the object's attributes, args is obtained from parse.parse_args()


if __name__ == '__main__':
    cfg = ConfigNews()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger.info(device)

    for seed in [2025]:  # random seed
        print(f"\n======================================================== Training with seed {seed} ========================================================")
        main(cfg, seed, device)