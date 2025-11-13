import argparse


class ConfigNews(object):
    def __init__(self):
        parser, args = parse()
        self.parser = parser
        self.args = args

        self.set_configuration()

    def set_configuration(self):
        self.args.in_dim = 768
        self.args.proj_dim = 500
        self.args.hidden_size = 500
        self.args.num_labels = 2
        self.args.learning_rate = 1e-4
        self.args.model_save_path = "./saved_model"
        self.args.model_save_name = "VDTNews.pt"
        self.args.classifier_dropout = 0.2


def parse():
    p = argparse.ArgumentParser()

    p.add_argument("--batch_size", type=int, required=True,  help="batch size")   # 256
    p.add_argument("--max_epochs", type=int, required=True,  help="number of training epochs")
    p.add_argument("--target_domain", type=str, required=True, help="target topic")    # domain_list = ['bbc', 'guardian', 'usa_today', 'washington_post']
    p.add_argument("--base_model", type=str, required=True, help="{clip, blip-2, albef}")
    p.add_argument("--lambda_1", type=float, required=False, default=2, help="weight of the cls loss")
    p.add_argument("--lambda_2", type=float, required=False, default=0.5, help="weight of the diva loss")
    p.add_argument("--lambda_3", type=float, required=False, default=1, help="weight of the recon loss")
    p.add_argument("--loss_type", type=str, required=True, help="simclr")
    args = p.parse_args()
    return p, args


# Unit test
if __name__ == '__main__':
    cfg = ConfigNews()
    print(cfg.args)