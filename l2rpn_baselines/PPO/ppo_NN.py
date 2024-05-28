import os
import numpy as np
from sqlalchemy import true
import tensorflow as tf
import time
# tf.compat.v1.disable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)
import sys
sys.path.append(r'/home/fly/zyh')
from Common.trainingParam import TrainingParam

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dense, Dropout
    from tensorflow.keras.layers import Input
    from tensorflow.keras.losses import mean_squared_error
    from tensorflow.keras.callbacks import TensorBoard

class PPO(object):
    """Constructs the desired actor critic network"""

    def __init__(self, action_size, observation_size, lr=1e-5,
                training_param=TrainingParam()):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.training_param = training_param

        self.n_user = self.training_param.n_user
        # self.shared_archi_sizes = self.training_param.kwargs_archi["layer_shared_sizes"]  # 新增参数
        # self.shared_archi_activs = self.training_param.kwargs_archi['layer_shared_activs']  # 新增参数
        # self.policy_archi_size = self.training_param.kwargs_archi["layer_actor_sizes"]  # 新增参数
        # self.policy_archi_activs = self.training_param.kwargs_archi["layer_actor_activs"]  # 新增参数
        # self.critic_archi_size = self.training_param.kwargs_archi["layer_critic_sizes"]  # 新增参数
        # self.critic_archi_activs = self.training_param.kwargs_archi["layer_critic_activs"]  # 新增参数
        self.construct_network()

        # self.advs = np.zeros(shape=(self.training_param.MINIBATCH_SIZE,self.training_param.reward_size))
        # self.oldpas = np.zeros(shape=(self.training_param.MINIBATCH_SIZE,action_size))

    def GGF(self, qq_values, reward_n, weight):  # 输入的qq_values为[要计算GGF的数量，比如动作空间大小，用户数]
        sorted_q_values = tf.sort(qq_values, axis=-1, direction='ASCENDING')
        if weight == 1:
            ome = [1 / n for n in range(1, reward_n + 1)]
        else:
            ome = [1 / weight ** n for n in range(reward_n)]
        omega = tf.constant(ome, dtype=tf.float32)  # tf.constant创建一个常量tensor，此为将ome转成张量
        w = tf.tensordot(sorted_q_values, omega, axes=1)
        return w

    def custom_loss(self):
        def loss(data, y_pred):
            # advantagtes的shape(n_step,reward_size)
            # advantages = tf.Variable(self.advs)
            # oldpa = tf.Variable(self.oldpas)       #on-policy转成off-policy
            # oldpolicy_probs = K.sum(y_true * oldpa, axis=1)
            # y_true = data[:,:self.action_size]
            # adv = data[:,self.action_size:]
            y_true, adv, oldpa = tf.split(data,[self.action_size,1,self.action_size],axis=1)

            newpolicy_probs = K.sum(y_true * y_pred, axis=1)
            oldpolicy_probs = K.sum(y_true * oldpa, axis=1)
            # #老师
            # newpolicy_logits = -K.log(newpolicy_probs + 1e-10)
            # newpolicy_logits = tf.tile(tf.expand_dims(newpolicy_logits, axis=1),
            #                         multiples=[1, self.training_param.reward_size])
            # p1 = adv * newpolicy_logits
            # weight_coef1 = tf.constant(self.training_param.weight_coef, dtype=tf.float32)
            # loss_actor = tf.tensordot(p1, weight_coef1, axes=1)
            # loss_actor = tf.reduce_mean(loss_actor)


            # #我
            # p2 = -(adv * tf.expand_dims(K.log(newpolicy_probs+ 1e-10), axis=1))
            # # clipping_val先设为0.2，带PPO梯度裁剪
            # ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            # # p2 = -(adv * tf.expand_dims(ratio, axis=1))
            # p3 = -(tf.expand_dims(K.clip(ratio, min_value=1 - 0.2, max_value=1 + 0.2),axis=1) * adv)
            # p = K.minimum(p2, p3)

            #KL
            # kl = tf.distributions.kl_divergence(oldpolicy_probs,newpolicy_probs)

            p2 = -(adv * K.log(newpolicy_probs + 1e-10))
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p3 = -(K.clip(ratio, min_value=1 - 0.2, max_value=1 + 0.2) * adv)
            loss_actor2 = K.minimum(p2, p3)
            loss_actor = tf.reduce_mean(loss_actor2)

            # loss_entropy = -K.sum(y_pred[:][1] * K.log(y_pred[:][1] + 1e-10), axis=1)
            # loss_critic = tf.reduce_mean(K.square(y_true[:][2] - y_pred[:][2]))

            # loss = loss_actor + loss_critic + loss_entropy

            loss_entropy = -K.sum(y_pred * K.log(y_pred + 1e-10))           #loss with entropy
            loss_entropy = tf.reduce_mean(loss_entropy)
            loss_actor  += loss_entropy

            return loss_actor
        return loss

    def construct_network(self):
        #建立Policy网络
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


        #共享层
        # input_layer = Input(shape=(self.observation_size,), name="observation")
        lay_share = layer_feature
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["sharesizes"], self.training_param.kwargs_archi["shareactivs"])):
            lay_share = Dense(size, name="layer_shared_hidden{}".format(lay_num))(lay_share)  # put at self.action_size全连接层
            lay_share = Activation(act)(lay_share)  # 激活层

        #actor网络
        lay_actor = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Actorsizes"], self.training_param.kwargs_archi["Actoractivs"])):
            lay_actor = Dense(size, name="layer_actor_head_hidden{}".format(lay_num))(lay_actor)  # put at self.action_size全连接层
            lay_actor = Activation(act)(lay_actor)  # 激活层
        soft_proba = Dense(self.action_size,name="layer_actor_head_output", activation="softmax", kernel_initializer='uniform')(lay_actor)

        #critic网络
        lay_critic = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Criticsizes"], self.training_param.kwargs_archi["Criticactivs"])):
            lay_critic = Dense(size, name="layer_critic_head_hidden{}".format(lay_num))(lay_critic)  # put at self.action_size全连接层
            lay_critic = Activation(act)(lay_critic)  # 激活层
        v_output = Dense(1,name="layer_critic_head_output")(lay_critic)

        self.model_AE_head     = Model(inputs=[input_layer], outputs=[reverse_observation],name="AE-{}".format(self.training_param.model_name))
        self.model_policy_head = Model(inputs=[input_layer], outputs=[soft_proba],name="Actor-{}".format(self.training_param.model_name))
        self.model_critic_head = Model(inputs=[input_layer], outputs=[v_output],name="Critic-{}".format(self.training_param.model_name))
        self.model             = Model(inputs=[input_layer], outputs=[soft_proba, v_output, reverse_observation],name=self.training_param.model_name)
        print(self.model.summary())

    def predict_movement(self, data, epsilon):
        data = np.expand_dims(data, axis=0)
        # dummy_n = np.zeros((1, self.action_size))  #pa为网络输入时用
        # dummy_1 = np.zeros((1, self.training_param.reward_size))   #adv为网络输入时
        a_prob = self.model_policy_head.predict(data)
        opt_policy = np.random.choice(range(a_prob.shape[1]),p=a_prob.ravel())  # select action w.r.t the actions prob
        # epsilon-greedy
        # rand_val = np.random.random(1)
        # if (rand_val < epsilon):
        #     opt_policy = np.random.randint(0, self.action_size)
        pa = np.squeeze(a_prob)    #降维回（45，）
        return int(opt_policy), pa

    def random_move(self,data):
        opt_policy = np.random.randint(0, self.action_size)
        data = np.expand_dims(data, axis=0)
        a_prob = self.model_policy_head.predict(data)
        pa = np.squeeze(a_prob)
        return opt_policy,pa

    def get_advantages(self, states, dones, rewards, last_state):   #对每一回合计算所有步的adv,dones已经转换成done为0，非done为1
        values = self.model_critic_head.predict(states)
        Advantage = []
        gae = 0
        v = self.model_critic_head.predict(last_state)
        values = np.append(values,v,axis=0)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.training_param.gama * values[i + 1] * dones[i] - values[i]
            gae = delta + self.training_param.gama * self.training_param.lmbda * dones[i] * gae
            Advantage.insert(0, gae)

        adv = np.array(Advantage)
        adv_normalized = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return adv_normalized

    def calculate_returns(self,rewards, dones):
        result = np.empty_like(rewards)
        result[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            result[t] = rewards[t] + self.training_param.gama * dones[t] * result[t + 1]
        return result

    # def train(self, s_batch, a_batch, r_batch, s2_batch, d_batch, adv_batch, pa_batch, w_batch, idx_batch, batch_size):  #采样更新
    def train(self, s_batch, a_batch, r_batch, s2_batch, d_batch, pa_batch):
        """Trains networks to fit given parameters"""
        #求GAE
        donee      = tf.Variable(d_batch, dtype=tf.float32)
        done       = tf.expand_dims((1.0 - donee), axis=1)  # done为0，非done为1，用于筛选非done
        last_state = np.expand_dims(s2_batch[-1], axis=0)
        adv_batch  = self.get_advantages(s_batch,done,r_batch,last_state)
        self.advs  = adv_batch

        #求V_target
        V_last = self.model_critic_head.predict(last_state)
        r_batch[-1] += self.training_param.gama * done[-1] * V_last
        V_target     = self.calculate_returns(r_batch, done)
        act_batch    = tf.one_hot(a_batch, self.action_size)
        a_prob       = self.model_policy_head.predict(s_batch)

        #编译
        self.model.compile(optimizer=Adam(learning_rate=self.lr_, clipnorm=0.5),
                            loss=[self.custom_loss(), mean_squared_error, mean_squared_error],
                            loss_weights=self.training_param.loss_weight,  # 选critic_loss小的，entropy大的，adv正且大的，即actor_loss  的
                            run_eagerly=True)

        combined = tf.concat([act_batch, adv_batch, pa_batch], axis=1)
        reverse_obs_batch = s_batch
        h = self.model.fit(x=s_batch, y={'layer_actor_head_output':combined, 'layer_critic_head_output':V_target, 'layer_AE_head_output':reverse_obs_batch},
                            batch_size=32, epochs=3, shuffle=true)
        # loss = self.model.evaluate(x=s_batch, y=[np.append(act_batch,adv_batch,axis = 1), V_target,a_prob])
        # loss_actor = np.mean(cb.losses)
        # loss_v = 0
        loss_actor = h.history["layer_actor_head_output_loss"]
        # loss_v = cb.losses["layer_critic_head_output_loss"]
        loss_v = h.history["layer_critic_head_output_loss"]
        loss_AE = h.history["layer_AE_head_output_loss"]
        # loss_AE = 0

        #训练：Train_on_batch
        # loss = self.model.train_on_batch(x=s_batch, y=[np.append(act_batch,adv_batch,axis = 1), V_target, a_prob])
        return tf.reduce_mean(loss_actor),tf.reduce_mean(loss_v),tf.reduce_mean(loss_AE)

    @staticmethod
    def _get_path_model(path, name=None):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        path_policy_model = "{}_policy_head".format(path_model)
        path_critic_model = "{}_critic_head".format(path_model)
        return path_model, path_policy_model, path_critic_model

    def save_network(self, path, name=None, ext="h5"):
        # Saves model at specified path as h5 file
        # nothing has changed
        path_model, path_policy_model, path_critic_model = self._get_path_model(path, name)
        self.model.save('{}.{}'.format(path_model, ext))
        self.model_policy_head.save('{}.{}'.format(path_policy_model, ext))
        self.model_critic_head.save('{}.{}'.format(path_critic_model, ext))
        print("Successfully saved network at: {}".format(path))

    def load_network(self, path):
        # Load from a model.h5 file
        self.model.load_weights(path)
        print("Successfully loaded network from: {}".format(path))
