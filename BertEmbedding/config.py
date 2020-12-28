import torch

class Config(object):
    def __init__(self):
        self.embedding_size = 768
        self.hidden_size = 1024
        self.num_layers = 2
        self.tag_size = 10
        self.bert_path = "../chinese_wwm_ext_pytorch"
        self.use_cuda = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.01
        self.weight_decay = 1e-3
        self.epoch = 20