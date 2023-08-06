import torch
from config import Config
from debugs import train_loader
from model import BertBILSTMCrf
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

config = Config()

if config.use_cuda:
    model = BertBILSTMCrf(config).to(config.device)
else:
    model = BertBILSTMCrf(config)
# print(input_ids, token_type_ids, labels)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

def train():
    total_loss = 0
    for epoch in range(config.epoch):
        step = 0
        print("epoch" + " " + str(epoch))
        for i, batch in enumerate(train_loader):
            step += 1
            batch = (b.to(config.device) for b in batch)
            bat = [b for b in batch]
            input_ids, token_type_ids, labels = bat[0], bat[2], bat[1]
            model.train()
            model.zero_grad()
            output = model(input_ids=input_ids, token_type_ids=token_type_ids)
            print(output)

            loss = model.loss(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print("step: {} |  loss: {}".format(step, loss.item()))


train()
