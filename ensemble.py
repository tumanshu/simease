import codecs
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

path1 = codecs.open("result_45.87.txt", "r", "utf-8")
r_1 = path1.readlines()
path1.close()
path2 = codecs.open("result_46.49.txt", "r", "utf-8")
r_2 = path2.readlines()
path2.close()
path3 = codecs.open("result_46.43.txt", "r", "utf-8")
r_3 = path3.readlines()
path3.close()

dict_aveg={}
right_result = []
aveg_cos = []


for i in range(len(r_1)):
    ave_cos = (float(r_1[i].split("\t")[1])+float(r_2[i].split("\t")[1])+float(r_3[i].split("\t")[1]))/3
    aveg_cos.append(ave_cos)
    right_result.append(int(r_1[i].split("\t")[0]))

'''
dict = {}
for i in range(-100,100,1):
    thred = float(i / float(100))
    pred = []
    for cos in aveg_cos:
        if cos>thred:
            pred.append(1)
        else:
            pred.append(0)
    coutMatrix = confusion_matrix(right_result, pred)
    #print(coutMatrix)

    precision_rate = precision_score(right_result, pred)
    recall_rate = recall_score(right_result, pred)
    F = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    if precision_rate==0 and recall_rate==0:
        F =0.0
    dict[thred] = F
       #w_.write(str(cos)+"\r\n")
dictss = sorted(dict.items(), key=lambda d: d[1], reverse=True)
for key in dictss:
    print(key)
'''