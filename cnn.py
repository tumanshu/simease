from base_model import *
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn



class cnn(object):
    def __init__(self, conf, input_layer):
        norm = tf.random_normal_initializer(stddev=0.1)
        self.l2_loss = tf.Variable(0.0)
        with tf.variable_scope('layer_0', reuse=tf.AUTO_REUSE):
            self.feature = []
            for i, filter_size in enumerate(input_layer.filter_size):
                filter_shape0 = [filter_size, conf.embedding_size, 1, 32]
                strides0 = [1, 1, conf.embedding_size, 1]
                self.filter_0 = tf.get_variable('%s%d'%('filter_',i), filter_shape0, initializer=norm)
                for embedding in input_layer.embedding_list_expand:
                    h0_1 = Conv(embedding, self.filter_0, strides0, input_layer.TRAIN, 'layer_0')
                    h0_2,_,_ = Convolutional_Block(h0_1,64,None,None,input_layer.TRAIN,'layer_1_2')
                    pooled = max_pool(h0_2, 64, "pool_1")
                    feature_1, self.loss_l2 = fc_layer(
                        tf.reshape(pooled, [-1, pooled.get_shape()[1] * pooled.get_shape()[-1]])
                        , conf.hidden_size, self.l2_loss, "fc-1-2-3_1", input_layer.TRAIN)
                    self.feature.append(feature_1)
            features_1=[]
            features_2=[]
            self.features=[]
            for j in range(len(self.feature)):
                if (j+2)%2==0:
                    features_1.append(self.feature[j])
                else:
                    features_2.append(self.feature[j])
            self.features.append(tf.concat(features_1,1))
            self.features.append(tf.concat(features_2,1))
            self.cosin = Cos_si(self.features)
            self.eru=Eru(self.features)

