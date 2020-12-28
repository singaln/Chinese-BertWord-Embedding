import torch
from transformers import BertModel

bert = BertModel.from_pretrained("../chinese_wwm_ext_pytorch")

input_ids = torch.tensor([[101, 32, 45, 345, 21, 102]])
token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
attention_mask = None

out = bert(input_ids, token_type_ids, attention_mask)
print(out)
