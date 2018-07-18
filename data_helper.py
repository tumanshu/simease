# coding=utf-8
import numpy as np
import os
import sys
import codecs
import csv
import copy
import sys
import random
import jieba
from langconv import *
#reload(sys)
#sys.setdefaultencoding('utf-8')


def array_one_hot(input_, length):

    target = np.zeros((len(input_),length))
    j = 0
    for i in input_:
        target[j][int(i)] = 1
        j = j+1
    return target

def writeDic(path, dic):
    write_path = codecs.open(path,"w")
    for key,item in dic.items():
        write_path.write(key+"\t\t"+str(item)+"\r\n")
    write_path.close()
def readDic(path):
    read_path = codecs.open(path, "r", "utf-8")
    s = read_path.readlines()
    read_path.close()
    dic = {}

    for line in s:
        lines = line.strip("\r\n").split("\t")
        dic[lines[0]] = int(lines[1])
        #if lines[0]=="余额宝":
            #print(lines[1])


    return dic

def get_flod(train_path, dev_path, dict_all, max_sentence_length, dropprob):
    y = []
    y_ = []
    x1 = []
    x2 = []
    sen_len_1 = []
    sen_len_2 = []
    f_ = codecs.open(train_path, "r", "utf-8")
    s_1 = f_.readlines()
    f_.close()

    f_ = codecs.open(dev_path, "r", "utf-8")
    s_2 = f_.readlines()
    f_.close()
    for s_ in[s_1,s_2]:
        for line in s_:
            lines = line.strip().split("\t")
            y.append(int(lines[0]))

            ori_len1 = len(lines[1].split())
            ori_len2 = len(lines[2].split())

            if ori_len1 > max_sentence_length:
                ori_len1 = max_sentence_length
                sen_len_1.append(max_sentence_length)
            else:
                sen_len_1.append(ori_len1)

            if ori_len2 > max_sentence_length:
                ori_len2 = max_sentence_length
                sen_len_2.append(max_sentence_length)
            else:
                sen_len_2.append(ori_len2)

            line1 = get_idx(lines[1], dict_all, max_sentence_length)
            x1.append(line1)

            line2 = get_idx(lines[2], dict_all,max_sentence_length)
            x2.append(line2)


    # dropout
    if dropprob != 1:

        x1_, x2_, y_, l1, l2 = selective_dropout(x1, x2, y, sen_len_1, sen_len_2, dropprob, dict_all,
                                                 "data/drop_list.txt")
        x1_all = x1 + x1_
        x2_all = x2 + x2_
        y_all = y + y_
        l1_all = sen_len_1 + l1
        l2_all = sen_len_2 + l2
    else:
        x1_all = x1
        x2_all = x2
        y_all = y
        l1_all = sen_len_1
        l2_all = sen_len_2
    y_ = array_one_hot(y_all, 2)

    return np.array(x1_all), np.array(x2_all), np.array(y_), np.array(y), np.array(l1_all) \
        , np.array(l2_all), len(y)

def getmatrix(gold_label, pre):
    matrix = np.zeros((2,2))
    for i in range(len(gold_label)):
        matrix[gold_label[i]][pre[i]]=matrix[gold_label[i]][pre[i]]+1
    #print(matrix)


    P = float(matrix[1][1]) / float(matrix[1][1] + matrix[1][0]+1)
    R = float(matrix[1][1]) / float(matrix[1][1] + matrix[0][1]+1)
    if P==0 or R==0:
        F1=0
    else:
        F1 = 2 * P * R / float(P + R)

    return matrix, P, R, F1
def get_idx(lines, dictall, max_sentencelen):
    i = 0
    line = np.zeros((max_sentencelen))
    for word in lines.split():
        if word in dictall:
            line[i] = int(dictall.get(word))
            if i>= max_sentencelen-1:
                break
            else:
                i = i+1
    return line

def get_idx_char(lines, dictall, max_sentencelen):
    i = 0
    line = np.zeros((max_sentencelen))
    for word in "".join(lines.split()):
        if word in dictall:
            line[i] = int(dictall.get(word))
            if i>= max_sentencelen-1:
                break
            else:
                i = i+1
    return line

def make_idx_data_x1_x2_test(all_sentence, dictall, max_sentencelen):
    y=[]
    x1 = []
    x2 = []
    sen_len_1 = []
    sen_len_2 = []

    count_ = 0
    for line in all_sentence:
        lines = line.strip().split("\t")
        y.append(lines[0])
        line1 = get_idx(lines[1], dictall, max_sentencelen)
        line2 = get_idx(lines[2], dictall, max_sentencelen)

        if len(lines[1].split())>max_sentencelen:
            sen_len_1.append(max_sentencelen)
        else:
            sen_len_1.append(len(lines[1].split()))
        if len(lines[2].split())>max_sentencelen:
            sen_len_2.append(max_sentencelen)
        else:
            sen_len_2.append(len(lines[2].split()))
        x1.append(line1)
        x2.append(line2)

    return np.array(x1), np.array(x2), y, np.array(sen_len_1), np.array(sen_len_2) ,len(y)

def dropput(d, s_index, p):
    drop_out = copy.copy(s_index)
    try:
        index = np.random.choice(d, int(d*p))
        for dex in index:
            drop_out[dex] = 0
    except:
        print(s_index)


    return drop_out

def selective_dropout(x1,x2,y,sen1_len, sent2_len, prob,dictall,dicpath):
    assert len(x1)==len(x2),'number of sentences in X1 and X2 unequal!'
    num=int(np.floor(len(x1)*prob))
    idx=np.random.randint(0,len(x1),size=num).tolist()
    drop_ids=[]
    with codecs.open(dicpath,'r','utf8') as f:
        lines=f.readlines()
        drop_words=[line.strip() for line in lines]
        for w in drop_words:

            try:
                drop_ids.append(dictall[w])
            except:
                print(w,dictall[w])
    x11=[]
    x22=[]
    y2=[]
    l1=[]
    l2=[]
    for i in idx:
        x11.append(x1[i])
        x22.append(x2[i])
        y2.append(y[i])
        l1.append(sen1_len[i])
        l2.append(sent2_len[i])
    for e1,e2 in zip(x11,x22):
        for drop in drop_ids:
            if drop in e1 and drop in e2:
                for j in range(len(e1)):
                    if e1[j]==drop:
                        e1[j]=0
                for j in range(len(e2)):
                    if e2[j]==drop:
                        e2[j]=0

    return x11,x22,y2,l1,l2

def make_idx_data_x1_x2(path, dictall, max_sentencelen, prob, flag):
    y=[]
    y_ = []
    x1 = []
    x2 = []
    sen_len_1 = []
    sen_len_2 = []
    f_ = codecs.open(path,"r", "utf-8")
    s_ = f_.readlines()
    f_.close()
    count_ = 0
    for line in s_:
        lines = line.strip().split("\t")
        y.append(int(lines[0]))

        if flag:
            ori_len1 = len("".join(lines[1].split()))
            ori_len2 = len("".join(lines[2].split()))

            if ori_len1 > max_sentencelen:
                ori_len1 = max_sentencelen
                sen_len_1.append(max_sentencelen)
            else:
                sen_len_1.append(ori_len1)

            if ori_len2 > max_sentencelen:
                ori_len2 = max_sentencelen
                sen_len_2.append(max_sentencelen)
            else:
                sen_len_2.append(ori_len2)
        else:
            ori_len1 = len(lines[1].split())
            ori_len2 = len(lines[2].split())

            if ori_len1 > max_sentencelen:
                ori_len1 = max_sentencelen
                sen_len_1.append(max_sentencelen)
            else:
                sen_len_1.append(ori_len1)

            if ori_len2 > max_sentencelen:
                ori_len2 = max_sentencelen
                sen_len_2.append(max_sentencelen)
            else:
                sen_len_2.append(ori_len2)

        if flag:
            line1 = get_idx_char(lines[1], dictall, max_sentencelen)
        else:
            line1 = get_idx(lines[1], dictall, max_sentencelen)
        x1.append(line1)

        if flag:
            line2 = get_idx_char(lines[2], dictall, max_sentencelen)
        else:
            line2 = get_idx(lines[2], dictall, max_sentencelen)

        x2.append(line2)

    if prob != 1:

        x1_,x2_, y_,l1,l2=selective_dropout(x1,x2,y,sen_len_1,sen_len_2,prob,dictall,"data/drop_list.txt")
        x1_all = x1+x1_
        x2_all = x2+x2_
        y_all = y+y_
        l1_all = sen_len_1+l1
        l2_all = sen_len_2+l2
    else:
        x1_all = x1
        x2_all = x2
        y_all = y
        l1_all = sen_len_1
        l2_all = sen_len_2
    y_ = array_one_hot(y_all, 2)

    return np.array(x1_all), np.array(x2_all), np.array(y_), y, np.array(l1_all)\
        , np.array(l2_all), len(y)


def stopwordslist(filepath):
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', "utf-8").readlines()]
    return stopwords
def tradition2simple(line):

    line = Converter('zh-hans').convert(line)
    return line
def readmodel(path):
    path_list = os.listdir(path)
    pathlist = []
    length_list = []
    for abspath in path_list:
        if "meta" not in abspath:
            continue
        else:
            absname = abspath.split(".meta")[0]
            length = int(absname.split("_")[-1])
            length_list.append(length)
            pathlist.append(path+"/"+absname)
    return pathlist,length_list

def tradition2simple(line):
    line = Converter('zh-hans').convert(line)
    return line

def replaceword(sentence, word_dict, stop_words):
    sentence = sentence.replace("***", " ")
    if "收钱吗" in sentence and sentence.find("收钱吗")!=len(sentence)-3:
        print(sentence.find("收钱吗"), len(sentence))
        sentence = sentence.replace("收钱吗","收钱码")
    if "收款吗" in sentence and sentence.find("收款吗")!=len(sentence)-3:
        print(sentence.find("收款吗"), len(sentence))
        sentence = sentence.replace("收款吗","收款码")

    for key, value in word_dict.items():
        for word in value:
            if word in sentence:
                sentence = sentence.replace(word, key)

    sentence = tradition2simple(sentence)
    sentence = "".join(sentence.split(" "))
    sentence = " ".join(jieba.cut(sentence))
    #print (sentence)
    outstr = ""
    for wordss in sentence.split():
        if wordss not in stop_words:
                outstr += wordss
                outstr += " "
    return outstr

def getallreplacesentence(train_lines, list_all_sentence, stop_words, same_dict):
    list_none = []
    for lines in train_lines:
        l_s = lines.strip().split("\t")
        sentence1 = replaceword(l_s[1], same_dict, stop_words)
        #print(sentence1)
        sentence2 = replaceword(l_s[2], same_dict, stop_words)
        sentence = l_s[0]+"\t"+sentence1 + "\t" + sentence2
        #print (sentence)
        if len(sentence1) <=1 or len(sentence2)<=1:
            list_none.append(l_s[0])
            continue

        if ('花呗' in sentence1 and '借呗' in sentence2 and not (
                '花呗' in sentence1 and '借呗' in sentence1 and
                '花呗' in sentence2 and '借呗' in sentence2)) or ('借呗'
            in sentence1 and '花呗' in sentence2 and not ('花呗' in sentence1 and '借呗' in sentence1
            and '花呗' in sentence2 and '借呗' in sentence2)):
            #print(l_s[0])
            list_none.append(l_s[0])
            continue

        list_all_sentence.append(sentence)
    return list_all_sentence,list_none
def load_vector_dic(vector_path, encoding):

    word_vecs = {}
    with codecs.open(vector_path, "r", "utf-8") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        i = 1
        for i in range(vocab_size):
            line_list = f.readline().split(' ')
            word = line_list[0]
            vec = ','.join(line_list[1:layer1_size + 1])
            word_vecs[word] = np.fromstring(vec, dtype='float32', sep=',')

    return word_vecs





def shuffled(lenth, x, y, z):
    np.random.seed(19911111)

    shuffle_indices = np.random.permutation(np.arange(lenth))
    x_shuffled = x[shuffle_indices]

    y_shuffled = y[shuffle_indices]

    z_shuffled = z[shuffle_indices]

    return np.array(x_shuffled), np.array(y_shuffled), np.array(z_shuffled)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
       Generates a batch iterator for a dataset.
       """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def writeCSV(dataList,path):

    with open(path, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()

def readFileName(filePath):
    path_list = os.listdir(filePath)
    path_list_ret = []
    for file in path_list:
        path_list_ret.append(os.path.join(filePath, file))

    return path_list_ret





if __name__=="__main__":
    #depart train/dev
    dir_path = sys.argv[1] #"human_reboot"#"ada_content" #
    isDir = sys.argv[2] #True #


