import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

class BertBILSTMCrf(nn.Module):
    def __init__(self, config):
        super(BertBILSTMCrf, self).__init__()

        self.embedding = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.tag_size = config.tag_size

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(self.embedding, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.tag_size)
        self.dropout = nn.Dropout()

        self.crf = CRF(num_tags=self.tag_size, batch_first=True)

    def forward(self, input_ids, token_type_ids, att_mask=None):
        out = self.bert(input_ids, token_type_ids, att_mask)
        out, _ = self.lstm(out[0])
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def loss(self, out, tags, mask=None):
        loss = -1 * self.crf(out, tags, mask=mask)
        return loss