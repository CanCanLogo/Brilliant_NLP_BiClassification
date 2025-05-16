#! ~/opt/anaconda3/bin/python3.8
# -*- coding: utf-8 -*-
import numpy
import random
import os
"""
程序说明：数据预处理
输入：原始语料，输出：train.txt，test.txt，
在输出中，每条评论占一行，positive样本以1开头，negative样本以0开头
这个程序把所有的评论合并成了两个txt，朴素贝叶斯不需要词向量处理
"""
# 原始语料路径:
neg_path = '../LabeledData/negative/'
pos_path = '../LabeledData/positive/'
# 输出路径:
train_path = "./train.txt"
test_path = "./test.txt"

train_split = 0.8  # 训练集占比

files = os.listdir(pos_path)
random.shuffle(files)
files_neg = os.listdir(neg_path)
random.shuffle(files_neg)

# 生成训练集：
with open(train_path, 'w') as tar:
    for file in files[0:int(train_split*len(files))]:
        path = pos_path + file
        with open(path, 'r', encoding='utf-8') as f:
            content = ""
            for line in f:
                if line[:-1] == "":
                    continue
                content += line[:-1]
                content += '。'
            content = "1"+content+'\n'
            tar.write(content)
    for file in files_neg[0:int(train_split*len(files))]:
        path = neg_path + file
        with open(path, 'r', encoding='utf-8') as f:
            content = ""
            for line in f:
                if line[:-1] == "":
                    continue
                content += line[:-1]
                content += '。'
            content = "0" + content + '\n'
            tar.write(content)
# 生成测试集：
with open(test_path, 'w') as tar:
    for file in files[int(train_split*len(files)):]:
        path = pos_path + file
        with open(path, 'r', encoding='utf-8') as f:
            content = ""
            for line in f:
                if line[:-1] == "":
                    continue
                content += line[:-1]
                content += '。'
            content = "1"+content+'\n'
            tar.write(content)
    for file in files_neg[int(train_split*len(files)):]:
        path = neg_path + file
        with open(path, 'r', encoding='utf-8') as f:
            content = ""
            for line in f:
                if line[:-1] == "":
                    continue
                content += line[:-1]
                content += '。'
            content = "0" + content + '\n'
            tar.write(content)
