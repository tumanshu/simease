# coding=utf-8
#! /usr/bin/env python

import tensorflow as tf
#import absl
import os
import time
import codecs
import data_helper
import numpy as np
from atcnn import atCNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import datetime

# Parameters
# ==================================================
# Data loading params
flags = tf.flags

flags.DEFINE_string("train_path", "train_resame.txt", "Data source for the source path")
flags.DEFINE_string("dev_path", "dev_resame.txt", "Data target for the target path")
flags.DEFINE_string("all_data_path", "word_dict", "Data target for the target path")
flags.DEFINE_string("dir_path", "data", "Data source for the target path")
flags.DEFINE_boolean("word_vec", True, "if have word2vec")
flags.DEFINE_boolean("ten_floder", True, "if have word2vec")
flags.DEFINE_string("word2vec_path", "word2vec_i10_w3_n10_m5.txt", "the word2vec path")
flags.DEFINE_integer("max_sentence_length", 10, "the word max sentence char length")


#the cnn params
flags.DEFINE_integer("embedding_size", 100, "The converlutional filter size")
flags.DEFINE_integer("attention_size", 100, "The converlutional filter size")
flags.DEFINE_string('filter_size', "1,2,3", "filter_size")
flags.DEFINE_integer('hidden_size', 256, "filter_size")
flags.DEFINE_integer('filter_num', 16, "filter_num")
flags.DEFINE_integer("pooling_size", 9, "The pooling size")
flags.DEFINE_integer("k_max_pool", 8, "The k_max pooling value (default :3) ")
flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_float("margin", 0.25, "the threadhold")
flags.DEFINE_integer("sentiment_class", 2, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_integer('word_attention_size', 100, "word attention size")

flags.DEFINE_integer('rnn_size', 150, 'RNN unit size')


# Training parameters
flags.DEFINE_integer("batch_size", 5, "Batch Size (default: 64)")

flags.DEFINE_integer("num_epochs", 120, "Number of training epochs (default: 120)")
flags.DEFINE_float("dropprob", 0.01, "Number of training epochs (default: 120)")
flags.DEFINE_float("shuffelprob", 0.01, "Number of training epochs (default: 120)")
flags.DEFINE_float("keep_prob", 0.5, "Number of training epochs (default: 120)")
flags.DEFINE_float("initial_learning_rate", 0.01, "Number of training epochs (default: 200)")
flags.DEFINE_integer("train_num", 1, "Number of training epochs (default: 200)")
flags.DEFINE_integer("dev_num", 1, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 5)")


# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

conf = tf.flags.FLAGS
dir_path = conf.dir_path
para_dict = {}
print("\nParameters:")
for attr, value in conf.__flags.items():
    para_dict[attr.upper()] = value
    print("{}={}".format(attr.upper(), value))
print("")

dict_all = data_helper.readDic(os.path.join(dir_path, conf.all_data_path))
print(len(dict_all))
#print( list(dict_all.keys())[list(dict_all.values()).index(282)])
#print(dict_all["余额宝"])
if conf.word_vec:
    word2 = data_helper.load_vector_dic(os.path.join(dir_path,conf.word2vec_path),"utf-8")
    word2vec = np.zeros((len(dict_all)+2, conf.embedding_size))

    for i in range(2,len(word2)):
        t = list(dict_all.keys())[list(dict_all.values()).index(i)]
        #print (t)
        if t not in word2:
            continue
        word2vec[i] = word2.get(t)
        #print(word2vec[i])
    word2vec = word2vec.astype('float32')

else:
    word2vec = None




# Data Preparation
# ==================================================
print("Loading dic...")
#word2vec = word2vec.[:][1:]


print("Loading data...")
#add all data used for domian classifier
train_x1, train_x2, train_y, train_y_o, train_sen_1, train_sen_2,\
            num_train = data_helper.make_idx_data_x1_x2(os.path.join(dir_path,
                                conf.train_path), dict_all, conf.max_sentence_length, conf.dropprob,False)

dev_x1, dev_x2, dev_y, dev_y_o, dev_sen_1, dev_sen_2,\
                    num_dev = data_helper.make_idx_data_x1_x2(os.path.join(dir_path,
                                conf.dev_path), dict_all,conf.max_sentence_length, 1, False)





#se_x, se_y = data_helper.copy_data((num_d), se_x, se_y)

print("number of train data: " + str(num_train))
print("number of dev data:" + str(num_dev))
print("loading over")

# Training
# ==================================================
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth= False)
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=conf.allow_soft_placement,
        log_device_placement=conf.log_device_placement)
    sess = tf.Session(config=session_conf)
    with tf.device('/gpu:0') and sess.as_default():

        domain_model = atCNN(conf, len(dict_all)+2, word2vec)

        loss_adv = domain_model.loss(conf, domain_model.cos_sim, domain_model.l2_loss, conf.l2_reg_lambda)
        accuracy = domain_model.accuracy(domain_model.cos_sim)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        #optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
        #rate_ = tf.div(0.0075,tf.sqrt(tf.sqrt(tf.pow(tf.add(1.0,tf.div(global_step,120.0)), 3.0))))
        '''
        learning_rate = tf.train.exponential_decay(conf.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10, decay_rate=0.95)
        '''
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        optimizer = tf.train.AdamOptimizer(conf.initial_learning_rate)  # tf.train.AdamOptimizer(0.0001) #
        train_op = optimizer.minimize(loss_adv, global_step=global_step)
        #grads_and_vars = optimizer.compute_gradients(loss_adv)
        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs" + "_" + conf.dir_path, timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf.num_checkpoints)

        w_path = os.path.join(checkpoint_dir, "log.txt")
        w_para = codecs.open(w_path, "w", "utf-8")
        w_para.write("max_sentence_length"+"\t"+str(conf.max_sentence_length)+"\r\n")
        w_para.write("dropprob"+"\t"+str(conf.dropprob)+"\r\n")


        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(train_x1, train_x2, train_y, train_y_, train_sen_1, train_sen_2, flags):
            """
            A single training step
            """
            feed_dict = {
                domain_model.input_x1: train_x1,
                domain_model.input_x2: train_x2,
                domain_model.y: train_y_,
                domain_model.input_x1_len:train_sen_1,
                domain_model.input_x2_len:train_sen_2,
                #domain_model.batch_len:len(train_x1),
                domain_model.training: 1,
                domain_model.keep_prob:conf.keep_prob
            }

            _, step, cos, p_loss, input, loss, acc = sess.run(
                [train_op, global_step, domain_model.cos_sim,
                 domain_model.positive_loss,
                 domain_model.input_x2, loss_adv,accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print(input)
            #matrix, p, r, f = data_helper.getmatrix(train_y_, predict)

            if flags:
                print("step:{}, acc:{}, loss: {}".format(step, acc, loss))
                '''
                print(" P value is"+ str(p) +" ")
                print(" r value is"+ str(r) +" ")
                print(" f value is"+ str(f) +" ")
                '''
            return cos
        def dev_step(dev_x1, dev_x2, dev_y, dev_sen1, dev_sen2):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                domain_model.input_x1: dev_x1,
                domain_model.input_x2: dev_x2,
                domain_model.y: dev_y,
                domain_model.input_x1_len: dev_sen1,
                domain_model.input_x2_len: dev_sen2,
                #domain_model.batch_len:len(dev_x1),
                domain_model.training: 0,
                domain_model.keep_prob: 1,

            }
            #test_data_domian_x, test_data_domian_y = sess.run([sentiment_feature.input_x_se, sentiment_feature.se_y],
            #                                                    feed_dict)
            #print(test_data_domian_x, test_data_domian_y)
            step, cosin, acc, pre = sess.run(
                [global_step, domain_model.cos_sim, accuracy, domain_model.predictions_int],
                feed_dict)

            return step, cosin, acc, pre


        # Generate batches

        #write_path_dev = os.path.join(checkpoint_prefix, "result_dev.txt")
        #write_path_train = os.path.join(checkpoint_prefix, "result_train.txt")
        #if not os.path.exists(write_path_dev):
        #    os.mkdir(checkpoint_prefix)


        batches_train = data_helper.batch_iter(
            list(zip(train_x1, train_x2, train_y, train_y_o, train_sen_1, train_sen_2)),
            conf.batch_size, conf.num_epochs, shuffle=True)

        # Training loop. For each batch...
        i = 0
        #w_t = codecs.open(write_path_train, "w")
        all_cos = []
        for batch in batches_train:
            x1_train, x2_train, y_train, y_o_train, sen_train_1, sen_train_2 = zip(*batch)

            if i == conf.train_num:
                cos = train_step(x1_train, x2_train, y_train, y_o_train, sen_train_1, sen_train_2, flags=True)
                all_cos.append(cos)
                i = 0
            else:
                cos = train_step(x1_train, x2_train, y_train, y_o_train, sen_train_1, sen_train_2, flags=False)
                all_cos.append(cos)
                i = i + 1

            current_step = tf.train.global_step(sess, global_step)
            loss_all = []
            acc_all = []
            p_all = []
            r_all = []
            f_all = []
            cos_all = []
            final_loss = 0
            final_acc = 0
            final_p = 0
            final_r = 0
            final_f = 0

            if current_step % conf.dev_num == 0:
                batches_dev = data_helper.batch_iter(
                    list(zip(dev_x1, dev_x2, dev_y, dev_y_o, dev_sen_1, dev_sen_2)), conf.batch_size, 1, False)
                print("\nEvaluation:")
                #w_d = codecs.open(write_path_dev, "w")

                for bat in batches_dev:
                    x1_dev, x2_dev, y_dev, y_o_dev, sen1_dev, sen2_dev = zip(*bat)
                    step, cosin, acc, pre = dev_step( x1_dev, x2_dev, y_o_dev, sen1_dev, sen2_dev)
                    cos_all.append(cosin)
                    acc_all.append(acc)
                    #print(gold_p)
                    #print(dev_y__)
                    matrix, p, r, f = data_helper.getmatrix(y_o_dev, pre)
                    #print(matrix)
                    p_all.append(p)
                    r_all.append(r)
                    f_all.append(f)
                time_str = datetime.datetime.now().isoformat()
                dicts = {}
                for j in range(-100, 100, 1):
                    thred = float(j / float(100))
                    pred = []
                    for cosin in cos_all:
                        for cos in cosin:
                            if cos > thred:
                                pred.append(1)
                            else:
                                pred.append(0)
                    coutMatrix = confusion_matrix(dev_y_o, pred)
                    #print(coutMatrix)
                    precision_rate = precision_score(dev_y_o, pred)
                    recall_rate = recall_score(dev_y_o, pred)
                    if precision_rate==0 and recall_rate==0:
                        F=0.0
                    else:
                        F = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
                    #print(F)
                    dicts[thred] = F

                dictss = sorted(dicts.items(), key=lambda d: d[1], reverse=True)
                print(dictss[0])

            if current_step % conf.dev_num == 0:
                checkpoint_prefix = os.path.join(checkpoint_dir, str(dictss[0]))
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))







