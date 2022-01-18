#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: SinGaln
# @time: 2021/12/24 13:57

import jieba
import logging
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer

jieba.setLogLevel(logging.INFO)

bert_path = "D:\Project\Fine_Tune_Bert_Distilled\pytorch_bert_model"
bert = BertModel.from_pretrained(bert_path)
token = BertTokenizer.from_pretrained(bert_path)


# Bert 字向量生成
def get_data(path, char):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
        if char:
            for sent in sentences:
                words.append(sent.strip())
        else:
            for sentence in sentences:
                cut_word = jieba.lcut(sentence.strip().replace(" ", ""))
                words.extend([w for w in cut_word if w not in words])
    return words


def get_bert_embed(path, char=False):
    words = get_data(path, char)
    file_word = open("word_embed.txt", "a+", encoding="utf-8")
    file_word.write(str(len(words)) + " " + "768" + "\n")
    linear = nn.Linear(768, 100)
    # 字向量
    if char:
        file_char = open("char_embed.txt", "a+", encoding="utf-8")
        file_char.write(str(len(words)) + " " + "100" + "\n")
        for word in words:
            inputs = token.encode_plus(word, padding="max_length", truncation=True, max_length=10,
                                       add_special_tokens=True,
                                       return_tensors="pt")
            out = bert(**inputs)
            out = linear(out[0])
            out = out.detach().numpy().tolist()
            out_str = " ".join("%s" % embed for embed in out[0][1])
            embed_out = word + " " + out_str + "\n"
            file_char.write(embed_out)
        file_char.close()

    # 词向量 (采用字向量累加求均值)
    for word in words:
        words_embed = np.zeros(768)  # bert tensor shape is 768
        inputs = token.encode_plus(word, padding="max_length", truncation=True, max_length=50, add_special_tokens=True,
                                   return_tensors="pt")
        out = bert(**inputs)
        out = linear(out[0])
        word_len = len(word)
        out_ = out.detach().numpy()
        for i in range(1, word_len + 1):
            out_str = out_[0][i]
            words_embed += out_str
        words_embed = words_embed / word_len
        words_embedding = words_embed.tolist()
        result = word + " " + " ".join("%s" % embed for embed in words_embedding) + "\n"
        file_word.write(result)

    file_word.close()


# char 为False时执行的是词向量生成， 为True则执行字向量生成
get_bert_embed("D:\Project\Fine_Tune_Bert_Distilled\pytorch_bert_model/vocab.txt", char=True)
print("Generate Finished!!!")
