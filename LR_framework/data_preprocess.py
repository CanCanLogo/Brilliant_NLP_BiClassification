# -*- coding='utf-8' -*-
import numpy as np
import random
import jieba
import os
from torch.utils.data import DataLoader, Dataset
from gensim.models.keyedvectors import KeyedVectors
import pickle
"""
程序功能：读取原始语料数据，制作Dataset并保存，以用于在训练时生成mini-batch
注意：本实现中，模型的输入是拼接句子中的所有词向量后得到的一个向量，设置句子的长度为固定值80（超出的部分截断，不足的部分padding），
可以根据任务要求决定模型输入为词向量求和还是拼接（请修改string2vec和get_data中的实现）
"""
TRAIN_SPLIT = 0.6  # 训练集所占比例
WORD_DIM = 300  # 词向量维度
MAX_SENT_LEN = 80 # 设置句子定长
stopwords = []    # 停词表 在stopwords.txt读取

def mystrip(ls):
    """
    函数功能：消除句尾换行
    strip删除列表中的每一个字符串的首尾空格和结尾的换行符
    在get_data中用于处理stopword.txt
    """
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    """
    函数功能：去掉为空的分词

    """
    _i = 0
    for _ in range(len(_words)):
        if _words[_i] in stopwords or _words[_i].strip() == "":
            # print(_words[_i])
            _words.pop(_i)
        else:
            _i += 1
    return _words

def load_data():
    """
    函数功能：读取原始语料。
    :return: 积极句子，积极label，消极句子，消极label
    """
    jieba.setLogLevel(jieba.logging.INFO)

    pos_sentence, pos_label, neg_sentence, neg_label = [], [], [], []

    pos_fname = '../LabeledData/positive'  # 请改为积极语料库实际路径
    neg_fname = '../LabeledData/negative'  # 请改为消极语料库实际路径

    '''
    pos_fname+'/'+f_name可以巧妙读取各个txt文件
    '''
    for f_name in os.listdir(pos_fname):
        with open(pos_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)

            pos_sentence.append(remove_stopwords(words))
            '''
            这里remove_stopwords是我们自己定义的函数,去掉为空的分词
            '''

            pos_label.append(1)  # label为1表示积极，label为0表示消极
            
    for f_name in os.listdir(neg_fname):
        with open(neg_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)
            neg_sentence.append(remove_stopwords(words))
            neg_label.append(0)

    return pos_sentence, pos_label, neg_sentence, neg_label

def string2vec(word_vectors, sentence):
    """
    函数功能：将sentence中的string词语转换为词向量
    :param word_vectors: 词向量表,在158行左右被加载
    :param sentence: 原始句子（单词为string格式）
    :return: 将string改为词向量
    注意：如果采用词向量求和的方式，请取消本函数中截断和padding的操作
    """
    '''
    登陆词就是在词向量之中的词
    真实的文本切分中，未登录词大约九成都是专有名词，其余的为通用新词或专业术语。
    因此，未登录词识别就是包括中国人名、译名、日本人名、地理位置名称、组织机构名称等专有名词的识别。再加上数字、日期等被统称为命名实体
    '''
    '''
    padding词
    在NLP中，文本一般是不定长的，所以在进行 batch训练之前，要先进行长度的统一，
    过长的句子可以通过truncating 截断到固定的长度，
    过短的句子可以通过 padding 增加到固定的长度，
    但是 padding 对应的字符只是为了统一长度，并没有实际的价值，
    因此希望在之后的计算中屏蔽它们，这时候就需要 Mask。
    '''
    for i in range(len(sentence)):  # 遍历所有句子
        sentence[i] = sentence[i][:MAX_SENT_LEN]  # 截断句子
        line = sentence[i]
        for j in range(len(line)):
            if line[j] in word_vectors:  # 如果是登录词 得到其词向量表示
                line[j] = word_vectors.get_vector(line[j])
            else:  # 如果不是登录词 设置为随机词向量
                line[j] = np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32")
        if len(line) < MAX_SENT_LEN:  # padding词设置为随机词向量
            for k in range(MAX_SENT_LEN-len(line)):
                sentence[i].append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))
    return sentence


class Corpus(Dataset):
    """
    定义数据集对象，用于构建DataLoader迭代器
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def get_data():
    global stopwords
    '''
    在汉语中，有一类没有多少意义的词语，
    比如组词“的”，连词“以及”、副词“甚至”，语气词“吧”，被称为停用词。
    一个句子去掉这些停用词，并不影响理解。
    所以，进行自然语言处理时，我们一般将停用词过滤掉。
    '''
    # 读取停用词列表：
    with open("../stopwords.txt", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)
    # 读取原始数据：
    pos_sentence, pos_label, neg_sentence, neg_label = load_data()

    sentence = pos_sentence + neg_sentence
    label = pos_label + neg_label

    sentence = sentence[:]
    label = label[:]

    shuffle = list(zip(sentence, label))
    random.shuffle(shuffle)  # 打乱数据集
    sentence[:], label[:] = zip(*shuffle)

    # 划分训练集、测试集
    assert len(sentence) == len(label)
    length = int(TRAIN_SPLIT*len(sentence))
    train_sentence = sentence[:length]
    train_label = label[:length]
    test_sentence = sentence[length:]
    test_label = label[length:]

    # 加载词向量
    '''
    gensim用于加载训练好的词向量
    '''
    print("loading word2vec...")
    # word_vectors = KeyedVectors.load_word2vec_format("sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5")
    # word_vectors = KeyedVectors.load_word2vec_format("../sgns.weibo.bigram-char/sgns.weibo.bigram-char")
    word_vectors = KeyedVectors.load_word2vec_format('../sgns.weibo.word')
    print("loading end")

    # 将string单词转为词向量
    train_sentence = string2vec(word_vectors, train_sentence)
    test_sentence = string2vec(word_vectors, test_sentence)

    # 拼接一句话中的所有词向量（可根据要求改为对所有词向量求和）
    train_sentence = [np.concatenate(wordvecs) for wordvecs in train_sentence]
    test_sentence = [np.concatenate(wordvecs) for wordvecs in test_sentence]

    # 生成数据集
    train_set = Corpus(train_sentence, train_label)
    test_set = Corpus(test_sentence, test_label)

    return train_set, test_set

if __name__ == "__main__":
    # 生成并保存数据集，注意根据实际情况设置输出路径
    train_set, test_set = get_data()
    outpath = './train_set1.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(train_set, f)
    outpath = './test_set1.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(test_set, f)
