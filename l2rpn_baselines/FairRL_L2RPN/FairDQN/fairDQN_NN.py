import os
import numpy as np
from sqlalchemy import true
import tensorflow as tf

from Common.trainingParam import TrainingParam

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add, Reshape
    from tensorflow.keras.layers import Input, Lambda, Concatenate
    from tensorflow.keras.losses import mean_squared_error

# class RLQvalue(object):
#     """
#     This class aims at representing the Q value (or more in case of SAC) parametrization by
#     a neural network.
#
#     It is composed of 2 different networks:
#     - model: which is the main model
#     - target_model: which has the same architecture and same initial weights as "model" but is updated less frequently
#       to stabilize training
#
#     It has basic methods to make predictions, to train the model, and train the target model.
#     """
#     def __init__(self, action_size, observation_size,
#                  lr=1e-5,
#                  training_param=TrainingParam()):
#         self.action_size = action_size
#         self.observation_size = observation_size
#         self.lr_ = lr
#         self.qvalue_evolution = np.zeros((0,))
#         self.training_param = training_param
#
#         self.model = None
#         self.target_model = None
#
#     def construct_q_network(self):
#         raise NotImplementedError("Not implemented")
#
#     def predict_movement(self, data, epsilon):
#         """Predict movement of game controler where is epsilon
#         probability randomly move."""
#         rand_val = np.random.random(data.shape[0])
#         q_actions = self.model.predict(data)
#         opt_policy = np.argmax(np.abs(q_actions), axis=-1)
#         opt_policy[rand_val < epsilon] = np.random.randint(0, self.action_size, size=(np.sum(rand_val < epsilon)))
#
#         self.qvalue_evolution = np.concatenate((self.qvalue_evolution, q_actions[0, opt_policy]))
#         return opt_policy, q_actions[0, opt_policy]
#
#     def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
#         """Trains network to fit given parameters"""
#         targets = self.model.predict(s_batch)
#         fut_action = self.target_model.predict(s2_batch)
#         targets[:, a_batch] = r_batch
#         targets[d_batch, a_batch[d_batch]] += self.training_param.decay_rate * np.max(fut_action[d_batch], axis=-1)
#
#         loss = self.model.train_on_batch(s_batch, targets)
#         # Print the loss every 100 iterations.
#         if observation_num % 100 == 0:
#             print("We had a loss equal to ", loss)
#         return np.all(np.isfinite(loss))
#
#     @staticmethod
#     def _get_path_model(path, name=None):
#         if name is None:
#             path_model = path
#         else:
#             path_model = os.path.join(path, name)
#         path_target_model = "{}_target".format(path_model)
#         return path_model, path_target_model
#
#     def save_network(self, path, name=None, ext="h5"):
#         # Saves model at specified path as h5 file
#         # nothing has changed
#         path_model, path_target_model = self._get_path_model(path, name)
#         self.model.save('{}.{}'.format(path_model, ext))
#         self.target_model.save('{}.{}'.format(path_target_model, ext))
#         print("Successfully saved network at: {}".format(path))
#
#     def load_network(self, path, name=None, ext="h5"):
#         # nothing has changed
#         path_model, path_target_model = self._get_path_model(path, name)
#         self.model = load_model('{}.{}'.format(path_model, ext))
#         self.target_model = load_model('{}.{}'.format(path_target_model, ext))
#         print("Successfully loaded network from: {}".format(path))
#
#     def target_train(self):
#         # nothing has changed from the original implementation
#         model_weights = self.model.get_weights()
#         target_model_weights = self.target_model.get_weights()
#         for i in range(len(model_weights)):
#             target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * \
#                                       target_model_weights[i]
#         self.target_model.set_weights(target_model_weights)

class DeepQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self,
                 action_size,
                 observation_size,
                 lr=1e-5,
                 training_param=TrainingParam(),
                 ):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.training_param = training_param
        self.n_user = self.training_param.n_user
        # self.nn_archi_sizes = self.training_param.kwargs_archi["sizes"]  # 新增参数
        # self.nn_archi_activs = self.training_param.kwargs_archi['activs']  # 新增参数
        self.construct_q_network()

    def construct_q_network(self):
        self.model = Sequential()
        # encoder input
        input_layer = Input(shape=(self.observation_size,),name="observation")

        # encoder level
        lay_encoder = Dense(128, name="layer_encoder_hidden1")(input_layer)
        lay_encoder = Activation("relu")(lay_encoder)
        lay_encoder = Dropout(0.05)(lay_encoder)

        # bottleneck layer
        layer_feature = Dense(64, name="layer_feature")(lay_encoder)
        layer_feature = Activation("relu")(layer_feature)

        # decoder level
        lay_decoder = Dense(128, name="layer_decoder_hidden1")(layer_feature)
        lay_decoder = Activation("relu")(lay_decoder)
        lay_decoder = Dropout(0.05)(lay_decoder)

        # decoder output
        reverse_observation = Dense(self.observation_size, name="layer_AE_head_output")(lay_decoder)

        lay_Qnet = layer_feature
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["sizes"], self.training_param.kwargs_archi['activs'])):
            lay_Qnet = Dense(size, name="layer_Qnet_hidden{}".format(lay_num))(lay_Qnet)  # put at self.action_size全连接层
            lay_Qnet = Activation(act)(lay_Qnet)           #激活层

        q_output = Dense(self.action_size*self.training_param.reward_size)(lay_Qnet)
        q_out = Reshape([self.action_size, self.training_param.reward_size], name="layer_Qnet_output")(q_output)  # 将结果reshape为|A|*|R|的矩阵
        self.model_AE_head = Model(inputs=[input_layer], outputs=[reverse_observation],name="AE-{}".format(self.training_param.model_name))
        self.model_Qnet_head = Model(inputs=[input_layer], outputs=[q_out],name="Qnet-{}".format(self.training_param.model_name))
        self.model = Model(inputs=[input_layer], outputs=[q_out, reverse_observation],name=self.training_param.model_name)

        # self.model.compile(loss='mse', optimizer=Adam(lr=self.lr_,clipnorm = 0.5))
        print(self.model.summary())
        self.target_model = Model(inputs=[input_layer], outputs=[q_out])
        self.target_model.set_weights(self.model_Qnet_head.get_weights())

    # def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, batch_size):  # 普通buffer
    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, w_batch, idx_batch, batch_size):  # priority buffer的需多加输入idx_batch, batch_size
        # 根据状态得到对应q表
        q_value = self.model_Qnet_head.predict(s_batch)  # model输出的Q值
        q_target_value = self.target_model.predict(s2_batch)  # target_model输出的Q值

        #求a*
        rew_batch = tf.constant(r_batch, dtype=tf.float32)
        donee = tf.constant(d_batch, dtype=tf.float32)
        done = tf.expand_dims((1.0 - donee), axis=1)                                                               # done为0，非done为1，用于筛选非done
        target_q = tf.expand_dims(done, axis=1) * q_target_value                                                     # 非done的target_q_value
        m = tf.expand_dims(rew_batch[:,:self.n_user], axis=1) + self.training_param.gama * target_q[:,:,:self.n_user]  # m=r[1:D]+γtarget_q[1:D]
        weight_coef = tf.constant(self.training_param.weight_coef, dtype=tf.float32)       #调整数据类型
        target = tf.tensordot(q_target_value, weight_coef, axes=1) + self.training_param.alpha * self.GGF(m, self.n_user, 2)   # target=w*Q^(s',a') + λGGF(r[1:D]+γtarget_q[1:D])
        fut_actions = tf.argmax(target, axis=1)                                                                       # 求出a*， = argmax[w*Q^(s',a') + λGGF(r[1:D]+γQ^(s'，a')[1:D])]
        #求Q(s,a)的实际值
        target_selected_actions = tf.one_hot(fut_actions, self.action_size )                                          # 对a*加onehot编码便于筛选
        target_valid_action = tf.reshape(target_selected_actions, shape=[-1, self.action_size, 1])
        max_future_q = tf.reduce_sum((q_target_value * target_valid_action),axis=1)                                   # Q^(s',a*),即a*对应的targetQ值（Qhead）
        max_future_q_masked = done * max_future_q                                                                     # 挑出非done的Q^(s',a*)，done的全为0
        q_sa_selected_target = rew_batch + self.training_param.gama * max_future_q_masked                               # 求出更新后的Q(s,a)，=r+γQ^(s'，a*),包括done的

        act_batch = tf.one_hot(a_batch, self.action_size )                        # 对a加onehot编码便于筛选
        act_batch = tf.reshape(act_batch, shape=[-1, self.action_size, 1])
        q_sa_selected = tf.reduce_sum((q_value * act_batch),axis=1)              #Q(s,a)的预测值
        y_batch = q_value               # 用于记录更新后的Q值，即模型训练用的y
        for i in range(batch_size):
            # q_sa_selected[i] = y_batch[i, a_batch[i]]              #Q(s,a)的预测值
            y_batch[i, a_batch[i]] = q_sa_selected_target[i]

        # loss = self.model_Qnet_head.train_on_batch(s_batch, y_batch)
        # 编译
        self.model.compile(optimizer=Adam(learning_rate=self.lr_, clipnorm=0.5),
                           loss=[mean_squared_error, mean_squared_error],
                           loss_weights=self.training_param.loss_weight,
                           run_eagerly=True)
        h = self.model.fit(x=s_batch, y={'layer_Qnet_output': y_batch, 'layer_AE_head_output': s_batch},
                           batch_size=batch_size, epochs=1, shuffle=true)
        # batch_size = batch_size, epochs = 1,
        loss_Q = h.history["layer_Qnet_output_loss"]
        loss_AE = h.history["layer_AE_head_output_loss"]

        # 针对priority buffer的
        sq_error = tf.math.square(q_sa_selected - q_sa_selected_target)
        batch_sq_error = tf.reduce_mean(sq_error, axis=1)
        td_error = q_sa_selected - q_sa_selected_target
        batch_td_error = tf.reduce_mean(td_error, axis=1)
        priorities = batch_td_error.numpy()
        # priorities = tf.square(y_batch - tf.stop_gradient(q_value))
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)

        #更新taget_model,在agent.train

        # return loss                #普通buffer
        return tf.reduce_mean(loss_Q),tf.reduce_mean(loss_AE), priorities  # priority buffer的需多返回一个priorities

    def GGF(self, qq_values, reward_n, weight):
        sorted_q_values = tf.sort(qq_values, axis=-1, direction='ASCENDING')
        if weight == 1:
            ome = [1 / n for n in range(1, reward_n + 1)]
        else:
            ome = [1 / weight ** n for n in range(reward_n)]
        omega = tf.constant(ome, dtype=tf.float32)  # tf.constant创建一个常量tensor，此为将ome转成张量
        w = tf.tensordot(sorted_q_values, omega, axes=1)
        return w

    def convert_q_to_TQ(self,q_values):           #q_values为一个batch的state对应的q，shape为（1，actionspace_size，reward_size）
        GGF_value = self.GGF(q_values[:,:,:self.n_user], self.n_user, 2)
        weight_coef = tf.constant(self.training_param.weight_coef, dtype=tf.float32)
        # TQ_value = tf.tensordot(q_values, weight_coef, axes=1) + GGF_value
        TQ_value = tf.tensordot(q_values, weight_coef, axes=1) + self.training_param.alpha*GGF_value
        return TQ_value

    #返回动作编号，动作对应的q，所有的q
    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        data = np.expand_dims(data, axis=0)
        q_values = self.model_Qnet_head.predict(data)
        Q_values = self.convert_q_to_TQ(q_values)
        opt_policy = np.argmax(Q_values, axis=1)
        opt_policy = int(opt_policy)
        rand_val = np.random.random(1)
        if(rand_val < epsilon):
            opt_policy = np.random.randint(0, self.action_size)

        return int(opt_policy), Q_values[0: opt_policy], Q_values

    def random_move(self):
        opt_policy = np.random.randint(0, self.action_size)
        return opt_policy

    def target_train(self):
        # nothing has changed from the original implementation
        model_weights = self.model_Qnet_head.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.training_param.TAU * model_weights[i] + (1 - self.training_param.TAU) * \
                                      target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    # @staticmethod
    # def _get_path_model(path, name=None):
    #     if name is None:
    #         path_model = path
    #     else:
    #         path_model = os.path.join(path, name)
    #     path_target_model = "{}_target".format(path_model)
    #     return path_model, path_target_model
    #
    # def save_network(self, path, name=None, ext="h5"):
    #     # Saves model at specified path as h5 file
    #     # nothing has changed
    #     path_model, path_target_model = self._get_path_model(path, name)
    #     self.model.save('{}.{}'.format(path_model, ext))
    #     self.target_model.save('{}.{}'.format(path_target_model, ext))
    #     print("Successfully saved network at: {}".format(path))
    #
    # def load_network(self, path, name=None, ext="h5"):
    #     # nothing has changed
    #     path_model, path_target_model = self._get_path_model(path, name)
    #     self.model = load_model('{}.{}'.format(path_model, ext))
    #     self.target_model = load_model('{}.{}'.format(path_target_model, ext))
    #     print("Successfully loaded network from: {}".format(path))

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_Qnet_model = "{}_Q_head".format(path_model)
        return path_model, path_Qnet_model

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        # nothing has changed
        path_model, path_Qnet_model = self._get_path_model(path, name)
        self.model.save('{}.{}'.format(path_model, ext))
        self.model_Qnet_head.save('{}.{}'.format(path_Qnet_model, ext))
        print("Successfully saved network at: {}".format(path))

    def load_network(self, path):
        # Load from a model.h5 file
        self.model.load_weights(path)
        print("Successfully loaded network from: {}".format(path))
