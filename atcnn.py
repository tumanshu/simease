from base_model import *
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn

def get_predictions(scores):
    predictions = tf.sign(scores, name="predictions")
    return(predictions)

def attention(inputs, size, scope, reuse):
    with tf.variable_scope(scope, reuse):
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[size],
                                                   regularizer=layers.l2_regularizer(scale=L2_REG),
                                                   dtype=tf.float32)
        outputs = []
        for emb in inputs:
            input_projection = layers.fully_connected(emb, size,
                                                  activation_fn=tf.tanh,
                                                  weights_regularizer=layers.l2_regularizer(scale=L2_REG))
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = tf.multiply(emb, attention_weights)
            output = tf.reduce_sum(weighted_projection, axis=1)
            outputs.append(output)
    return outputs

class atCNN(object):
    def __init__(self, conf, num_quantized_chars, word2vec):


        length = []
        #self.positin = tf.get_variable("position", [None, conf.max_sentence_length], initializer=)
        self.input_x1 = tf.placeholder(tf.int32, [None, conf.max_sentence_length], name="input_x1")
        self.input_x1_len = tf.placeholder(tf.int32, [None], name="input_x1_len")
        length.append(self.input_x1_len)

        self.input_x2 = tf.placeholder(tf.int32, [None, conf.max_sentence_length], name="input_x2")
        self.input_x2_len = tf.placeholder(tf.int32, [None], name="input_x2_len")
        length.append(self.input_x2_len)
        #self.batch_len =  tf.placeholder(tf.int32,1, name="batch_length")
        table = tf.cast(tf.range(0,conf.batch_size,1),tf.int64)

        self.y = tf.placeholder(tf.float32, [None], name="y")
        self.training = tf.placeholder(tf.int32, name="trainable")
        self.filter_size = list(map(int, conf.filter_size.split(",")))
        self.keep_prob = tf.placeholder(tf.float32, name="keepprob")

        if self.training == 0:
            TRAIN = False
        else:
            TRAIN = True


        self.l2_loss = tf.Variable(0.0)
        if not conf.word_vec:
            self.W0 = tf.get_variable("W", [num_quantized_chars, conf.embedding_size], trainable=TRAIN)
        else:
            self.W0 = tf.get_variable("W", initializer=word2vec, trainable=True, dtype='float32')


        norm = tf.random_normal_initializer(stddev=0.1)


        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_list = []
            self.embedded_characters_x1 = tf.nn.embedding_lookup(self.W0, self.input_x1)
            self.embedded_characters_expanded_x1 = tf.expand_dims(self.embedded_characters_x1, -1, name="embedding_input")

            self.embedded_characters_x2 = tf.nn.embedding_lookup(self.W0, self.input_x2)
            self.embedded_characters_expanded_x2 = tf.expand_dims(self.embedded_characters_x2, -1,
                                                               name="embedding_input")
            self.position = tf.cast(tf.reshape(tf.tile(table, [conf.max_sentence_length]),
                                               [-1,conf.max_sentence_length]), tf.int64)
            self.embedding_list.append(self.embedded_characters_x1)
            self.embedding_list.append(self.embedded_characters_x2)

        with tf.variable_scope("em_att") as scope:
             self.att = attention(self.embedding_list, conf.attention_size, scope=scope, reuse=tf.AUTO_REUSE)


        '''
        with tf.variable_scope('layer_0', reuse=tf.AUTO_REUSE):
            self.feature = []
            for i, filter_size in enumerate(self.filter_size):
                filter_shape0 = [filter_size, conf.embedding_size, 1, 32]
                strides0 = [1, 1, conf.embedding_size, 1]
                self.filter_0 = tf.get_variable('%s%d'%('filter_',i), filter_shape0, initializer=norm)
                for embedding in self.embedding_list:
                    h0_1 = Conv(embedding, self.filter_0, strides0, TRAIN, 'layer_0')
                    h0_2,_,_ = Convolutional_Block(h0_1,64,None,None,TRAIN,'layer_1_2')
                    pooled = max_pool(h0_2, 64, "pool_1")
                    feature_1, self.loss_l2 = fc_layer(
                        tf.reshape(pooled, [-1, pooled.get_shape()[1] * pooled.get_shape()[-1]])
                        , conf.hidden_size, self.l2_loss, "fc-1-2-3_1", TRAIN)
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
        
        with tf.variable_scope('match') as scope:
            self.input_match= tf.matmul(self.embedded_characters_x1, self.embedded_characters_x2, transpose_a=False,
                        transpose_b=True)
            self.get_max = tf.argmax(self.input_match, 2)
            self.get_position = tf.concat([tf.transpose(self.position,[1,0]), tf.transpose(self.get_max,[1,0])],0)
            self.row_index = tf.reshape(tf.transpose(self.get_position,[1,0]),[-1,conf.max_sentence_length,2])

            #self.gather = tf.gather_nd(self.input_x2,self.row_index)
            #self.sort_x2 = tf.nn.embedding_lookup(self.W0, self.gather)
            self.embedding_list.append(self.embedded_characters_x1)
            self.embedding_list.append(self.embedded_characters_x2)
        '''
        with tf.variable_scope('2layer_LSTM'):
            h_state = []
            cell_unit1 = tf.contrib.rnn.LSTMCell  # tf.nn.rnn_cell.BasicLSTMCell


            # Forward direction cell
            lstm_cell1 = cell_unit1(conf.rnn_size, forget_bias=1.0)
            lstm_cell2 = cell_unit1(conf.rnn_size, forget_bias=1.0)
            with tf.variable_scope("first_layer_lstm"):
                outputs0, h_state0 = tf.nn.dynamic_rnn(cell=lstm_cell1, inputs=self.embedding_list[0],
                                                        dtype=tf.float32, sequence_length=self.input_x1_len)

                outputs1, h_state1 = tf.nn.dynamic_rnn(cell=lstm_cell1, inputs=self.embedding_list[1],
                                                           dtype=tf.float32, sequence_length=self.input_x2_len)
            with tf.variable_scope("second_layer_lstm"):
                outputs0, h_state0 = tf.nn.dynamic_rnn(cell=lstm_cell2, inputs=outputs0,
                                                       dtype=tf.float32, sequence_length=self.input_x1_len)

                outputs1, h_state1 = tf.nn.dynamic_rnn(cell=lstm_cell2, inputs=outputs1,
                                                       dtype=tf.float32, sequence_length=self.input_x2_len)
            h_state.append(h_state0[0])
            h_state.append(h_state1[0])




        with tf.variable_scope('BiLSTM', reuse=tf.AUTO_REUSE) as scope:
            self.features = []

            cell_unit = tf.contrib.rnn.BasicLSTMCell  # tf.nn.rnn_cell.BasicLSTMCell

            # Forward direction cell
            lstm_forward_cell = cell_unit(conf.rnn_size, forget_bias=1.0)
            lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell,
                                                              output_keep_prob=self.keep_prob)

            # Backward direction cell
            lstm_backward_cell = cell_unit(conf.rnn_size, forget_bias=1.0)
            lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell,
                                                               output_keep_prob=self.keep_prob)

            A = tf.get_variable(name="A", shape=[2 * conf.rnn_size, conf.hidden_size],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b", shape=[conf.hidden_size], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.1))
            # Create bidirectional layer
            i =0
            for embedding in self.embedding_list:
                enc_outputs, enc_out = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward_cell,
                                                       cell_bw=lstm_backward_cell,
                                                       inputs=embedding, dtype=tf.float32,
                                                       sequence_length=length[i])

                #final_state = tf.concat(enc_outputs,2)
                #final_state = attention(final_state, conf.attention_size, scope=scope)
                final_state = tf.concat([enc_out[0][0], enc_out[1][0]], 1)
                #embedding_sum = tf.squeeze(tf.reduce_mean(embedding, 1))
                #final_state = tf.concat([final_state,embedding_sum], 1)
                
                # Fully connected layer
                #final_output = tf.concat([self.att[i],final_state,h_state[i]],1)
                final_output = tf.matmul(final_state, A) + b

                final_output = tf.nn.dropout(final_output, self.keep_prob)
                self.features.append(final_output)
                i = i + 1

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            self.query_norm = tf.sqrt(tf.reduce_sum(tf.square(self.features[0]), 1, True))
            self.doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.features[1]), 1, True))

            self.prod = tf.reduce_sum(tf.multiply(self.features[0], self.features[1]), 1, True)
            norm_prod = tf.multiply(self.query_norm, self.doc_norm)

            cos_sim_raw = tf.truediv(self.prod, norm_prod)
            self.cos_sim = tf.squeeze(cos_sim_raw, [1])
            #self.cos_sim = tf.nn.sigmoid(self.cos_sim)

        with tf.variable_scope('EuclideanDis'):
            self.dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.features[0],self.features[1])),1))
            self.dis = 1.0/(1.0+self.dis)

    def loss(self, conf, feature, l2_loss,  l2_reg_lambda):
        #prob = tf.nn.softmax(feature)

        pos_loss = tf.square(tf.subtract(1., feature))

        self.positive_loss = self.y*pos_loss

        neg_mult = tf.subtract(1., tf.cast(self.y, tf.float32))
        self.negative_loss = neg_mult*tf.square(feature)

        self.loss = tf.add(self.positive_loss, self.negative_loss)

        target_zero = tf.equal(tf.cast(self.y, tf.float32), 0.)
        # Check if cosine outputs is smaller than margin
        less_than_margin = tf.less(feature, conf.margin)
        # Check if both are true
        both_logical = tf.logical_and(target_zero, less_than_margin)
        both_logical = tf.cast(both_logical, tf.float32)
        # If both are true, then multiply by (1-1)=0.
        multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
        total_loss = tf.multiply(self.loss, multiplicative_factor)

        # Average loss over batch
        self.avg_loss = tf.reduce_mean(total_loss)+l2_reg_lambda*l2_loss
        return self.avg_loss

    def accuracy(self, scores):
        self.predictions = get_predictions(scores)
        # Cast into integers (outputs can only be -1 or +1)
        y_target_int = tf.cast(self.y, tf.int32)
        # Change targets from (0,1) --> (-1, 1)
        #    via (2 * x - 1)
        y_target_int = tf.subtract(tf.multiply(y_target_int, 2), 1)
        self.predictions_int = tf.cast(tf.sign(self.predictions), tf.int32)
        correct_predictions = tf.equal(self.predictions_int, y_target_int)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return (accuracy)


