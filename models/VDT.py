'''
Stores the core modules of conDA model
'''
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from models.mmd_code import MMD
from torch.nn import TripletMarginLoss



# (1) MLP
class ProjectionMLP(nn.Module):
    """
    Model to project [CLS] representation onto
    another space, where the contrastive loss will
    be calculated.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.args.in_dim, cfg.args.in_dim),   # 768 -> 768
            nn.ReLU(),
            nn.Linear(cfg.args.in_dim, cfg.args.proj_dim),   # 768 -> 500
        )

    def forward(self, input_features):
        # input_features: [bs, 768], previously from [:, 0, :]
        return self.layers(input_features)


# (2) Classifier in the figure
class MLLMClassificationHead(nn.Module):
    """
    A classifier following the MLLM embedding
    Reference: https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/roberta/modeling_roberta.py#L1426
    (RobertaClassificationHead, RobertaForSequenceClassification)
    """
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.args.hidden_size, cfg.args.hidden_size)
        classifier_dropout = (
            cfg.args.classifier_dropout if cfg.args.classifier_dropout is not None else cfg.args.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(cfg.args.hidden_size, cfg.args.num_labels)
        self.soft_max = nn.Softmax(dim=1)
        self.num_labels = cfg.args.num_labels
        self.relu = nn.ReLU()

        ###### Add more dense layers to the original classifier ######
        self.ln1 = nn.Linear(768, 1024)
        self.ln2 = nn.Linear(1024, 4096)
        self.ln3 = nn.Linear(4096, 4096)
        self.ln4 = nn.Linear(4096, 1024)
        self.ln5 = nn.Linear(1024, 768)
        self.dense2 = nn.Linear(cfg.args.hidden_size, cfg.args.hidden_size)
        #############################
        
        ###### Add batch normalization ######
        self.bn = nn.BatchNorm1d(cfg.args.hidden_size)
        self.bn1 = nn.BatchNorm1d(cfg.args.hidden_size)
        #############################

    def forward(self, features):
        """
        Return the logits
        """
        x = self.dropout(features)
        x = self.dense(x)   # 500 -> 500
        x = self.bn1(x)   # adding this bn boost performance in z->cls
        x = self.relu(x)
        ###### Add more dense layers to the original classifier ######
        # x = self.ln1(x)   # 768 -> 1024
        # x = self.bn1(x)
        # x = torch.tanh(x)
        # x = self.ln2(x)   # 1024 -> 4096
        # x = self.bn2(x)
        # x = torch.tanh(x)
        # x = self.ln3(x)   # 4096 -> 4096
        # x = self.bn3(x)
        # x = torch.tanh(x)
        # x = self.ln4(x)   # 4096 -> 1024
        # x = self.bn4(x)
        # x = torch.tanh(x)
        # x = self.ln5(x)   # 1024 -> 768
        # x = self.bn5(x)
        # x = torch.tanh(x)
        # x = self.dense2(x)   # 500 -> 500
        # x = self.bn(x)
        # # x = torch.tanh(x)
        # x = self.relu(x)
        #############################
        x = self.dropout(x)
        x = self.out_proj(x)   # 500 -> 2
        logits = x
        return logits

    # L_CE loss in the figure
    def compute_loss(self, logits, labels):
        # logits is the forward() output
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

    def compute_softmax_logits(self, logits):
        softmax_logits = self.soft_max(logits)
        return softmax_logits


# (3a) L_CTR in the figure
class SimCLRContrastiveLoss(nn.Module):
    """
    SimCLR style contrastive loss
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pair
        (z_i, z_j) as per SimCLR paper
        """
        # Normalize each embedding (no need for BLIP-2?)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)   # [2*bs, 2*bs]

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)   # sim_ij = sim_ji?
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


# (3b)''' The overall model + our proposed triplet loss + input z to the classifier instead of h - lce_neg - ltriplet
class VDTModule(nn.Module):
    def __init__(self, model, mlp, domain_feature, loss_type, logger, device, lambda_1, lambda_2, lambda_3):
        # model, mlp is initialized outside
        super().__init__()
        self.model = model   # RobertaForContrastiveLearning in the original paper, here is MLLMClassificationHead
        self.mlp = mlp   # Projection mlp
        self.loss_type = loss_type   # "simclr"
        self.logger = logger
        self.device = device
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        
        self.feature_model = domain_feature
        
        self.ln0 = nn.Linear(128, 500)        
        self.ln1 = nn.Linear(128, 500)
        self.ln1.to(self.device)
        self.ln0.to(self.device)

    def forward(self, src_emb, tgt_emb, src_labels, tgt_labels, mode='train'):
        if mode == 'train':
            src_batch_size = src_emb.shape[0]
            tgt_batch_size = tgt_emb.shape[0]

            # (2) Compute diva losses

            if self.loss_type == "simclr":
                if src_batch_size == tgt_batch_size:
                    ctr_loss = SimCLRContrastiveLoss(batch_size=src_batch_size)
                    ctr_loss.to(self.device)
                else:
                    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
            
            DD = self.feature_model(src_emb, tgt_emb)  # domain feature dictionary
            
            mu_s, var_s = DD['F_s']['mu_s'], DD['F_s']['var_s']
            mu_t, var_t = DD['F_t']['mu_t'], DD['F_t']['var_t']
            
            gate_s = torch.sigmoid(var_s)
            gate_t = torch.sigmoid(var_t)
            
            F_s = mu_s * (1 - gate_s)
            F_t = mu_t * (1 - gate_t)

            # (1) Compute L_CE loss
            # source
            src_logits = self.model(self.ln0(F_s))
            src_ce_loss, src_logits = self.model.compute_loss(src_logits, src_labels), self.model.compute_softmax_logits(src_logits)

            # target
            tgt_logits = self.model(self.ln0(F_t))
            tgt_ce_loss, tgt_logits = self.model.compute_loss(tgt_logits, tgt_labels), self.model.compute_softmax_logits(tgt_logits)
            
            # (2) diva loss
            diva_loss = ctr_loss(mu_s, mu_t)

            # (3) vae recon loss
            src_reconloss = DD['F_s']['loss']
            tgt_reconloss = DD['F_t']['loss']
            recon_loss = (src_reconloss + tgt_reconloss)

            # Full loss
            loss = self.lambda_1 * src_ce_loss  + self.lambda_2 * diva_loss + self.lambda_3 * recon_loss


            data = {"total_loss": loss, "src_reconloss": src_reconloss, "tgt_reconloss": tgt_reconloss,"src_logits": src_logits,
                    "tgt_logits":tgt_logits, 'src_ce_loss': src_ce_loss, 'tgt_ce_loss': tgt_ce_loss}

            if isinstance(data, dict):
                data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
                data = data_named_tuple(**data)
            elif isinstance(data, list):
                data = tuple(data)

            return data
        
        elif mode == 'test':
            tgt_batch_size = tgt_emb.shape[0]
                      
            DD_t = self.feature_model(tgt_emb, mode='test')
            
            mu, var, y_tgt = DD_t['F_t']['mu_t'], DD_t['F_t']['var_t'], DD_t['F_t']['y_t']
            
            gate = torch.sigmoid(var)
            F_t = self.ln0(mu * (1 - gate))
            
            
            return F_t, y_tgt, mu, var