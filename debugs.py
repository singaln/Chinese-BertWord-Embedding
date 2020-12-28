import torch
from torch.utils.data import TensorDataset, DataLoader
class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def load_vocab(vocab_file):
    """
    :param vocab_file:vocab file path
    :return:
    """
    vocab = {}
    labels = {}
    index_ = 0
    index = 0
    words = []
    tags = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        sentences = f.readlines()
        for sentence in sentences:
            token = sentence.strip().split("\t")[0]
            token = token.split()
            label = sentence.strip().split("\t")[1]
            label = label.split()
            for word in token:
                if word not in words:
                    words.append(word)
                else:
                    continue
            for tag in label:
                if tag not in tags:
                    tags.append(tag)
                else:
                    continue
    for w in words:
        vocab[w] = index_
        index_ += 1

    for l in tags:
        labels[l] = index
        index += 1
    vocab.update({"UNK": len(vocab), "CLS": len(vocab) + 1, "SEP": len(vocab) + 2})
    labels.update({"<start>": len(labels), "<end>": len(labels) + 1, "<pad>": len(labels) + 2})
    return vocab, labels

def load_train_data(data_path, max_len, label_dic, vocab):
    """
    :param data_path:train_data path
    :param max_len: the max length of sentences
    :param label_dic: the label dictionary
    :param vocab: vocab dictionary
    :return:
    """
    with open(data_path, "r", encoding="utf-8") as file:
        content = file.readlines()
        result = []
        for lines in content:
            text, label = lines.strip().split("\t")
            tokens = text.split()
            labels = label.split()
            if len(tokens) > max_len - 2:
                tokens = tokens[: (max_len - 2)]
                labels = labels[: (max_len - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            labels = ["<start>"] + labels + ["<end>"]
            input_ids = [int(vocab[i]) if i in vocab else int(vocab["UNK"]) for i in tokens]
            label_ids = [label_dic[i] for i in labels]
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_len:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(label_dic["<pad>"])
            assert len(input_ids) == max_len
            assert len(input_mask) == max_len
            assert len(label_ids) == max_len
            features = InputFeatures(input_id=input_ids, label_id=label_ids, input_mask=input_mask)
            result.append(features)
        return result

vocab, label_dic = load_vocab("train.txt")
data = load_train_data("train.txt", max_len=50, vocab=vocab, label_dic=label_dic)

train_ids = torch.LongTensor([temp.input_id for temp in data])
train_tags = torch.LongTensor([temp.label_id for temp in data])
train_mask = torch.LongTensor([temp.input_mask for temp in data])

train_dataset = TensorDataset(train_ids, train_tags, train_mask)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
