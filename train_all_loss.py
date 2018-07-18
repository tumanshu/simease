# coding=utf-8
# ! /usr/bin/env python

import tensorflow as tf
# import absl
import os
import time
import codecs
import data_helper
import numpy as np
from base_model import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from input_layer import*
from input_layer_char import*
from singlelstm import*
from lstm_att import*
from two_lstm import*
from cnn import*
import loss_model

import datetime

# Parameters
# ==================================================
# Data loading params
flags = tf.flags

flags.DEFINE_string("train_path", "train_resame.txt", "Data source for the source path")
flags.DEFINE_string("dev_path", "dev_resame.txt", "Data target for the target path")
flags.DEFINE_string("all_data_path", "word_dict", "Data target for the target path")
flags.DEFINE_string("all_data_path_char", "char_dict", "Data target for the target path")
flags.DEFINE_string("dir_path", "data", "Data source for the target path")
flags.DEFINE_boolean("word_vec", True, "if have word2vec")
flags.DEFINE_string("word2vec_path", "word2vec_i10_w3_n10_m5.txt", "the word2vec path")
flags.DEFINE_string("word2vec_char_path", "char2vec_i10_w3_n10_m5.txt", "the word2vec path")
flags.DEFINE_integer("max_sentence_length", 10, "the word max sentence char length")
flags.DEFINE_integer("max_char_length", 10, "the word max sentence char length")

# the cnn params
flags.DEFINE_integer("embedding_size", 100, "The converlutional filter size")
flags.DEFINE_integer("embedding_char_size", 100, "The converlutional filter size")
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
flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")

flags.DEFINE_integer("num_epochs", 120, "Number of training epochs (default: 120)")
flags.DEFINE_float("dropprob", 1, "Number of training epochs (default: 120)")
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

#prob
flags.DEFINE_float("alpha", 0.5, "Number of training epochs (default: 120)")
flags.DEFINE_float("beta", 0.2, "Number of training epochs (default: 120)")
flags.DEFINE_float("gamma", 0.2, "Number of training epochs (default: 120)")
flags.DEFINE_float("cigama", 0.2, "Number of training epochs (default: 120)")



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

dict_all_char = data_helper.readDic(os.path.join(dir_path, conf.all_data_path_char))
print(len(dict_all_char))
# print( list(dict_all.keys())[list(dict_all.values()).index(282)])
# print(dict_all["余额宝"])
if conf.word_vec:
    word2 = data_helper.load_vector_dic(os.path.join(dir_path, conf.word2vec_path), "utf-8")
    word2vec = np.zeros((len(dict_all) + 2, conf.embedding_size))

    for i in range(2, len(word2)):
        t = list(dict_all.keys())[list(dict_all.values()).index(i)]
        # print (t)
        if t not in word2:
            continue
        word2vec[i] = word2.get(t)
        # print(word2vec[i])
    word2vec = word2vec.astype('float32')

    word2_char = data_helper.load_vector_dic(os.path.join(dir_path, conf.word2vec_char_path), "utf-8")
    word2vec_char = np.zeros((len(dict_all) + 2, conf.embedding_char_size))

    for i in range(2, len(word2_char)):
        t = list(dict_all_char.keys())[list(dict_all_char.values()).index(i)]
        # print (t)
        if t not in word2_char:
            continue
        word2vec_char[i] = word2_char.get(t)
        # print(word2vec[i])
    word2vec_char = word2vec_char.astype('float32')

else:
    word2vec = None
    word2vec_char = None

# Data Preparation
# ==================================================
print("Loading dic...")
# word2vec = word2vec.[:][1:]


print("Loading data...")
# add all data used for domian classifier
train_x1, train_x2, train_y, train_y_o, train_sen_1, train_sen_2, \
num_train = data_helper.make_idx_data_x1_x2(os.path.join(dir_path,
                                                         conf.train_path), dict_all, conf.max_sentence_length,
                                            conf.dropprob, False)

train_x1_char, train_x2_char, _, _, train_sen_1_char, train_sen_2_char, _\
    = data_helper.make_idx_data_x1_x2(os.path.join(dir_path,
                                        conf.train_path), dict_all_char, conf.max_char_length, conf.dropprob, True)

dev_x1, dev_x2, dev_y, dev_y_o, dev_sen_1, dev_sen_2, \
num_dev = data_helper.make_idx_data_x1_x2(os.path.join(dir_path,
                                                       conf.dev_path), dict_all, conf.max_sentence_length, 1, False)

dev_x1_char, dev_x2_char, _, _, dev_sen_1_char, dev_sen_2_char, _ \
    = data_helper.make_idx_data_x1_x2(os.path.join(dir_path, conf.dev_path), dict_all_char, conf.max_char_length, 1, True)

# se_x, se_y = data_helper.copy_data((num_d), se_x, se_y)

print("number of train data: " + str(num_train))
print("number of dev data:" + str(num_dev))
print("loading over")

# Training
# ==================================================
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=False)
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=conf.allow_soft_placement,
        log_device_placement=conf.log_device_placement)
    sess = tf.Session(config=session_conf)
    with tf.device('/gpu:0') and sess.as_default():
        input_layer = input(conf, len(dict_all) + 2, word2vec)
        input_layer_char =input_char(conf, len(dict_all_char)+2, word2vec_char)
        si_lstm_word = si_lstm(conf, input_layer,"word")
        si_lstm_char = si_lstm(conf, input_layer_char,"char")
        two_lstm_word = two_lstm(conf, input_layer, "word")
        two_lstm_char = two_lstm(conf, input_layer_char, "char")
        lstm_att_word = lstm_att(conf, input_layer,"word")
        lstm_att_char = lstm_att(conf, input_layer, "char")
        #cnn = cnn(conf, input_layer)
        '''
        feature_con = tf.concat([tf.expand_dims(si_lstm.cosin,-1),
                              tf.expand_dims(two_lstm.cosin,-1),
                              tf.expand_dims(lstm_att.cosin, -1),
                              #tf.expand_dims(cnn.cosin, -1),
                              #tf.expand_dims(si_lstm.eru,-1),
                              #tf.expand_dims(two_lstm.eru,-1),
                              #tf.expand_dims(lstm_att.eru,-1),
                              #tf.expand_dims(cnn.eru,-1),
                              ],1)
        
        feature_con=[]
        feature_con.append(tf.concat([si_lstm.features[0],two_lstm.features[0],lstm_att.features[0]],1))
        feature_con.append(tf.concat([si_lstm.features[1],two_lstm.features[1],lstm_att.features[1]],1))
        features = Cos_si(feature_con)
        '''

        all_features = tf.concat([lstm_att_word.mul, lstm_att_word.man, two_lstm_word.man,
                                  two_lstm_word.mul, si_lstm_word.mul, si_lstm_word.man,
                                  lstm_att_char.mul, lstm_att_char.man, two_lstm_char.man,
                                  two_lstm_char.mul, si_lstm_char.mul,si_lstm_char.man],1)
                                  #si_lstm_char.man,
                                  #two_lstm_char.man,si_lstm_char.mul,
                                  #two_lstm_char.mul],1)
        features_1, l2_loss = linear(all_features, 32, input_layer.TRAIN, "fc_layer")
        features_2, l2_loss1 = linear(all_features, 48, input_layer.TRAIN, "fc_layer1")
        all_features = tf.concat([tf.nn.relu(features_1), tf.nn.sigmoid(features_2)],1)
        features, l2_loss2 = linear(all_features,1, input_layer.TRAIN, "fc_layer2")

        features = tf.nn.sigmoid(features)
        features = tf.concat([features, tf.expand_dims(lstm_att_word.cosin,-1), tf.expand_dims(two_lstm_word.cosin,-1),
                              tf.expand_dims(si_lstm_word.cosin, -1),tf.expand_dims(lstm_att_char.cosin, -1),
                              tf.expand_dims(two_lstm_char.cosin, -1), tf.expand_dims(si_lstm_char.cosin,-1)
                              ],1)
        features, l2_loss3 = linear(features, 1, input_layer.TRAIN, "fc_layer3")

        #features = 2*tf.subtract(features, 0.5)
        #loss1 = loss_model.loss(conf, si_lstm.cosin, 0,  input_layer)
        #loss2 = loss_model.loss(conf, two_lstm.cosin, 0, input_layer)
        #loss3 = loss_model.loss(conf, lstm_att.cosin, 0, input_layer)
        #loss4 = loss_model.loss(conf, cnn.cosin, cnn.loss_l2,input_layer)
        #cosin = conf.alpha*si_lstm.cosin+conf.beta*two_lstm.cosin+conf.gamma*lstm_att.cosin+conf.cigama*cnn.cosin
        #features = (si_lstm.cosin+lstm_att.cosin+two_lstm.cosin)/3
        #loss,pos_loss, neg_loss = loss_model.loss(conf, features, cnn.loss_l2,input_layer)
        accuracy, loss, pred = loss_model.class_loss(input_layer, features, l2_loss+l2_loss1+l2_loss2)
        #accuracy, prediction_init = loss_model.accuracy(features, input_layer)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
        # rate_ = tf.div(0.0075,tf.sqrt(tf.sqrt(tf.pow(tf.add(1.0,tf.div(global_step,120.0)), 3.0))))
        '''
        learning_rate = tf.train.exponential_decay(conf.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10, decay_rate=0.95)
        '''
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        optimizer = tf.train.AdamOptimizer(conf.initial_learning_rate)  # tf.train.AdamOptimizer(0.0001) #
        train_op = optimizer.minimize(loss, global_step=global_step)
        # grads_and_vars = optimizer.compute_gradients(loss_adv)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs" + "_" + conf.dir_path, timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=conf.num_checkpoints)

        w_path = os.path.join(checkpoint_dir, "log.txt")
        w_para = codecs.open(w_path, "w", "utf-8")
        w_para.write("max_sentence_length" + "\t" + str(conf.max_sentence_length) + "\r\n")
        w_para.write("dropprob" + "\t" + str(conf.dropprob) + "\r\n")

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(train_x1, train_x2, train_x1_char, train_x2_char, train_y_, train_sen_1, train_sen_2,
                       train_sen_1_char, train_sen_2_char, flags):
            """
            A single training step
            """
            feed_dict = {
                input_layer.input_x1: train_x1,
                input_layer_char.input_x1_char: train_x1_char,
                input_layer.input_x2: train_x2,
                input_layer_char.input_x2_char: train_x2_char,
                input_layer.y: train_y_,
                input_layer.input_x1_len: train_sen_1,
                input_layer_char.input_x1_len: train_sen_1_char,
                input_layer.input_x2_len: train_sen_2,
                input_layer_char.input_x2_len: train_sen_2_char,
                # domain_model.batch_len:len(train_x1),
                input_layer.training: 1,
                input_layer_char.training: 1,
                input_layer.keep_prob: conf.keep_prob,
                input_layer_char.keep_prob: conf.keep_prob
            }
            _, step, loss_all= sess.run(
                [train_op, global_step,  loss], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            # print(input)
            # matrix, p, r, f = data_helper.getmatrix(train_y_, predict)

            if flags:
                print("step:{}, loss: {}".format(step, loss_all))
                '''
                print(" P value is"+ str(p) +" ")
                print(" r value is"+ str(r) +" ")
                print(" f value is"+ str(f) +" ")
                '''



        def dev_step(dev_x1, dev_x2, dev_x1_char, dev_x2_char, dev_y, dev_sen1, dev_sen2, dev_sen1_char, dev_sen2_char):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                input_layer.input_x1: dev_x1,
                input_layer_char.input_x1_char: dev_x1_char,
                input_layer.input_x2: dev_x2,
                input_layer_char.input_x2_char: dev_x2_char,
                input_layer.y: dev_y,
                input_layer.input_x1_len: dev_sen1,
                input_layer_char.input_x1_len: dev_sen1_char,
                input_layer.input_x2_len: dev_sen2,
                input_layer_char.input_x2_len: dev_sen2_char,
                # domain_model.batch_len:len(dev_x1),
                input_layer.training: 0,
                input_layer_char.training: 0,
                input_layer.keep_prob: 1,
                input_layer_char.keep_prob: 1,

            }
            # test_data_domian_x, test_data_domian_y = sess.run([sentiment_feature.input_x_se, sentiment_feature.se_y],
            #                                                    feed_dict)
            # print(test_data_domian_x, test_data_domian_y)
            step,  cosin_dev = sess.run(
                [global_step, features],# feature_confeatures,features,value,, con
                feed_dict)

            return step, cosin_dev


        # Generate batches

        # write_path_dev = os.path.join(checkpoint_prefix, "result_dev.txt")
        # write_path_train = os.path.join(checkpoint_prefix, "result_train.txt")
        # if not os.path.exists(write_path_dev):
        #    os.mkdir(checkpoint_prefix)

        batches_train = data_helper.batch_iter(
            list(zip(train_x1, train_x2, train_x1_char, train_x2_char,
                     train_y, train_y_o, train_sen_1, train_sen_2,
                     train_sen_1_char, train_sen_2_char)),
            conf.batch_size, conf.num_epochs, shuffle=True)

        # Training loop. For each batch...
        i = 0
        # w_t = codecs.open(write_path_train, "w")
        all_cos = []
        for batch in batches_train:
            x1_train, x2_train, x1_train_char, x2_train_char, \
            y_train, y_o_train, sen_train_1, sen_train_2, \
            sen_train_1_char, sen_train_2_char = zip(*batch)

            if i == conf.train_num:
                train_step(x1_train, x2_train, x1_train_char, x2_train_char,
                          y_o_train, sen_train_1, sen_train_2,sen_train_1_char, sen_train_2_char,
                           flags=True)
                i = 0
            else:
                train_step(x1_train, x2_train, x1_train_char, x2_train_char,
                          y_o_train, sen_train_1, sen_train_2,sen_train_1_char, sen_train_2_char,
                           flags=False)
                i = i + 1

            current_step = tf.train.global_step(sess, global_step)
            loss_all = []
            acc_all = []
            p_all = []
            r_all = []
            f_all = []
            cos_all = []
            final_loss = 0.0
            final_acc = 0.0
            final_p = 0.0
            final_r = 0.0
            final_f = 0.0

            if current_step % conf.dev_num == 0:
                batches_dev = data_helper.batch_iter(
                    list(zip(dev_x1, dev_x2,dev_x1_char, dev_x2_char,
                             dev_y, dev_y_o, dev_sen_1, dev_sen_2, dev_sen_1_char, dev_sen_2_char)), conf.batch_size, 1, False)
                print("\nEvaluation:")
                # w_d = codecs.open(write_path_dev, "w")

                for bat in batches_dev:
                    x1_dev, x2_dev, x1_dev_char, x2_dev_char, y_dev, y_o_dev, \
                    sen1_dev, sen2_dev, sen1_dev_char, sen2_dev_char = zip(*bat)
                    step, cosin_dev = dev_step(x1_dev, x2_dev, x1_dev_char, x2_dev_char,
                                               y_o_dev, sen1_dev, sen2_dev,sen1_dev_char, sen2_dev_char)
                    cos_all.append(cosin_dev)
                    #acc_all.append(acc)
                    # print(gold_p)
                    # print(dev_y__)
                    #matrix, p, r, f = data_helper.getmatrix(y_o_dev, pre)
                    # print(matrix)
                    #p_all.append(p)
                    #r_all.append(r)
                    #f_all.append(f)
                time_str = datetime.datetime.now().isoformat()
                dicts = {}

                for j in range(0, 1000, 1):
                    thred = float(j / float(1000))
                    pred = []
                    for cosin_ in cos_all:
                        for cos in cosin_:
                            if cos > thred:
                                pred.append(1)
                            else:
                                pred.append(0)
                    coutMatrix = confusion_matrix(dev_y_o, pred)
                    #print(coutMatrix)
                    precision_rate = precision_score(dev_y_o, pred)
                    recall_rate = recall_score(dev_y_o, pred)
                    if precision_rate == 0 and recall_rate == 0:
                        F = 0.0
                    else:
                        F = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
                    # print(F)
                    dicts[thred] = F

                dictss = sorted(dicts.items(), key=lambda d: d[1], reverse=True)
                print(dictss[0])

                '''
                for f_value in f_all:
                    final_f += f_value
                final_f = final_f/len(f_all)
                print(final_f)
                '''
            if current_step % conf.dev_num == 0:
                checkpoint_prefix = os.path.join(checkpoint_dir, str(dictss[0]))
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))

        # w_.close()






