# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
import random
try:
    import tensorflow as tf
    import tensorflow.keras as tfk
    import tensorflow.keras.backend as K
    import tensorflow.keras.models as tfkm
    import tensorflow.keras.optimizers as tfko
    import tensorflow.keras.layers as tfkl
    import tensorflow.keras.activations as tfka
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False


class DoubleDuelingRDQN_NN(object):
    """
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """
    def __init__(self,
                 action_size,
                 observation_size,
                 learning_rate = 1e-5):
        
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        
        self.action_size = action_size
        self.observation_size = observation_size
        self.h_size = 512

        self.lr = learning_rate
        
        self.model = None
        self.construct_q_network()

    def construct_q_network(self):
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, dtype=tf.int32, name="trace_length")#序列长度。
        self.dropout_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="dropout_rate")#用于控制输入的 dropout 比率（69行用到）。Dropout 是一种正则化技术，有助于减少过拟合。
        input_mem_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_mem_state')
        #这行代码创建了一个名为input_mem_state的输入层，用于表示LSTM的初始隐藏状态。
        input_carry_state = tfk.Input(dtype=tf.float32, shape=(self.h_size), name='input_carry_state')
        #这行代码创建了一个名为 input_carry_state 的输入层，用于表示 LSTM 的初始细胞状态。
        input_layer = tfk.Input(dtype=tf.float32, shape=(None, self.observation_size), name='input_obs')
        #这行代码创建了一个名为 input_layer 的输入层，用于表示观测序列,这个输入层可以接受任何形状的张量作为输入，
        # 只要张量的最后一个维度与输入层的形状匹配即可。因此，不管输入的张量是一维、二维、三维或更高维度，都可以作为输入传递给模型。
        # dtype=tf.float32 指定了输入数据的数据类型为 32 位浮点数。
        # self.observation_size 表示每个时间步观测的特征维度。name='input_obs' 指定了该输入层的名称为 'input_obs'。
        #self.observation_size 是指每个时间步观测的特征维度，这意味着在每个时间步，有一个包含455个特征的观测值。


        # Get shapes from input_layer，这段代码逐行获取了输入层 input_layer 的形状信息，分别表示批次大小、序列长度和特征维度。
        batch_size = tf.shape(input_layer)[0]
        trace_len = tf.shape(input_layer)[1]
        data_size = tf.shape(input_layer)[-1]

        # Reshape for dense processing
        input_format = tf.reshape(input_layer, (-1, input_layer.shape[-1]), name="dense_reshape")

        # Bayesian NN simulate
        lay1 = tfkl.Dropout(self.dropout_rate, name="bnn_dropout")(input_format)
        # Forward pass
        lay1 = tfkl.Dense(512, name="fc_1")(lay1)
        lay1 = tf.nn.leaky_relu(lay1, alpha=0.01, name="leak_fc_1")
        lay2 = tfkl.Dense(256, name="fc_2")(lay1)
        lay2 = tf.nn.leaky_relu(lay2, alpha=0.01, name="leak_fc_2")
        lay3 = tfkl.Dense(128, name="fc_3")(lay2)
        lay3 = tf.nn.leaky_relu(lay3, alpha=0.01, name="leak_fc_3")
        lay4 = tfkl.Dense(self.h_size, name="fc_4")(lay3)

        # Reshape to (batch_size, trace_len, data_size) for rnn
        rnn_format = tf.reshape(lay4, (batch_size, trace_len, self.h_size),name="rnn_reshape")
        # Recurring part
        lstm_layer = tfkl.LSTM(self.h_size, return_state=True, name="lstm")
        #return_state 参数决定是否返回最后一个时间步的隐藏状态和细胞状态
        lstm_state = [input_mem_state, input_carry_state]
        lstm_output, mem_s, carry_s = lstm_layer(rnn_format, initial_state=lstm_state)

        # Advantage and Value streams
        advantage = tfkl.Dense(64, name="fc_adv")(lstm_output)
        advantage = tf.nn.leaky_relu(advantage, alpha=0.01, name="leak_adv")
        advantage = tfkl.Dense(self.action_size, name="adv")(advantage)
        advantage_mean = tf.math.reduce_mean(advantage, axis=1,
                                             keepdims=True, name="adv_mean")
        advantage = tfkl.subtract([advantage, advantage_mean], name="adv_sub")

        value = tfkl.Dense(64, name="fc_val")(lstm_output)
        value = tf.nn.leaky_relu(value, alpha=0.01, name="leak_val")
        value = tfkl.Dense(1, name="val")(value)

        Q = tf.math.add(value, advantage, name="Qout")

        # Backwards pass
        model_inputs = [input_mem_state, input_carry_state, input_layer]
        model_outputs = [Q, mem_s, carry_s]
        self.model = tfk.Model(inputs=model_inputs,
                               outputs=model_outputs,
                               name=self.__class__.__name__)
        losses = [
            self._mse_loss,
            self._no_loss,
            self._no_loss
        ]
        self.optimizer = tfko.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.model.compile(loss=losses, optimizer=self.optimizer)

    def _no_loss(self, y_true, y_pred):
        return 0.0

    def _mse_loss(self, Qnext, Q):
        loss = tf.math.reduce_mean(tf.math.square(Qnext - Q), name="loss_mse")
        return loss

    def bayesian_move(self, data, mem, carry, rate = 0.0):
        self.dropout_rate.assign(float(rate))
        self.trace_length.assign(1)

        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry

    def random_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1) 
        move = np.random.randint(0, self.action_size)

        return move, mem, carry
        
    def predict_move(self, data, mem, carry):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        data_input = data.reshape(1, 1, -1)
        mem_input = mem.reshape(1, -1)
        carry_input = carry.reshape(1, -1)
        model_input = [mem_input, carry_input, data_input]

        Q, mem, carry = self.model.predict(model_input, batch_size = 1)
        move = np.argmax(Q)

        return move, Q, mem, carry

    def update_target_hard(self, target_model):
        this_weights = self.model.get_weights()
        target_model.set_weights(this_weights)

    def update_target_soft(self, target_model, tau=1e-2):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.trainable_variables
        main_params = self.model.trainable_variables

        # Update each param
        for i, var in enumerate(target_params):
            var_persist = var.value() * tau_inv
            var_update = main_params[i].value() * tau
            # Poliak averaging
            var.assign(var_update + var_persist)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save_weights(path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        print("Successfully loaded network from: {}".format(path))
