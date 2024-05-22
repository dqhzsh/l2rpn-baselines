import os
import numpy as np
from sqlalchemy import true
import tensorflow as tf
import time
# tf.compat.v1.disable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)
from l2rpn_baselines.FairRL_L2RPN.Common.trainingParam import TrainingParam

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import load_model, Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, subtract, add, Reshape
    from tensorflow.keras.layers import Input, Lambda, Concatenate
    from tensorflow.keras.losses import mean_squared_error,categorical_crossentropy
    from tensorflow.keras.activations import relu
    from tensorflow.keras.callbacks import TensorBoard

class A2C(object):
    """Constructs the desired actor critic network"""

    def __init__(self, action_size, observation_size, lr=1e-5,
                training_param=TrainingParam()):
        self.action_size = action_size
        self.observation_size = observation_size
        self.lr_ = lr
        self.training_param = training_param

        self.n_user = self.training_param.n_user
        # self.nn_archi_sizes = self.training_param.kwargs_archi["sizes"]  # 新增参数
        # self.nn_archi_activs = self.training_param.kwargs_archi['activs']  # 新增参数
        # self.policy_archi_size = self.training_param.kwargs_archi["Policysizes"]  # 新增参数
        # self.policy_archi_activs = self.training_param.kwargs_archi["Policyactivs"]  # 新增参数
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
            y_true = data[:,:self.action_size]
            adv = data[:,self.action_size:]

            newpolicy_probs = K.sum(y_true * y_pred, axis=1)

            # #老师
            # newpolicy_logits = -K.log(newpolicy_probs + 1e-10)
            # newpolicy_logits = tf.tile(tf.expand_dims(newpolicy_logits, axis=1),
            #                         multiples=[1, self.training_param.reward_size])
            # p1 = adv * newpolicy_logits
            # weight_coef1 = tf.constant(self.training_param.weight_coef, dtype=tf.float32)
            # loss_actor = tf.tensordot(p1, weight_coef1, axes=1)
            # print(loss_actor)
            # loss_actor = tf.reduce_mean(loss_actor)


            #我
            loss_actor2 = -(adv * K.log(newpolicy_probs+ 1e-10))
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
        # #建立Policy网络
        # input_layer = Input(shape=(self.observation_size,),name="observation")
        # lay = Dense(128, name="layer_shared_hidden1")(input_layer)  # put at self.action_size全连接层
        # lay = Activation('relu')(lay)  # 激活层
        # lay = Dense(128, name="layer_shared_hidden2")(lay)  # put at self.action_size全连接层
        # lay = Activation('relu')(lay)  # 激活层
        #
        # lay_actor = Dense(64, name="layer_actor_head_hidden3")(lay)  # put at self.action_size全连接层
        # lay_actor = Activation('relu')(lay_actor)  # 激活层
        # lay_actor = Dense(45, name="layer_actor_head_hidden4")(lay_actor)  # put at self.action_size全连接层
        # lay_actor = Activation('relu')(lay_actor)  # 激活层
        # soft_proba = Dense(self.action_size,name="layer_actor_head_output", activation="softmax", kernel_initializer='uniform')(lay_actor)
        # # predict_a_p = soft_proba
        # # entropy = -K.sum(soft_proba * K.log(soft_proba + 1e-10))
        #
        # lay_critic = Dense(128, name="layer_critic_head_hidden3")(lay)  # put at self.action_size全连接层
        # lay_critic = Activation('relu')(lay_critic)  # 激活层
        # v_output = Dense(self.training_param.reward_size,name="layer_critic_head_output")(lay_critic)
        # # v_out = Reshape((-1,self.training_param.reward_size))(v_output)  # 将结果reshape为batchsize*|R|的矩阵
        #共享层
        input_layer = Input(shape=(self.observation_size,), name="observation")
        lay_share = input_layer
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["sharesizes"], self.training_param.kwargs_archi["shareactivs"])):
            lay_share = Dense(size, name="layer_shared_hidden{}".format(lay_num))(lay_share)  # put at self.action_size全连接层
            lay_share = Activation(act)(lay_share)  # 激活层

        #actor网络
        lay_actor = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Actorsizes"], self.training_param.kwargs_archi["Actoractivs"])):
            lay_actor = Dense(size, name="layer_actor_head_hidden{}".format(lay_num))(lay_actor)  # put at self.action_size全连接层
            lay_actor = Activation(act)(lay_actor)  # 激活层
        soft_proba = Dense(self.action_size, name="layer_actor_head_output", activation="softmax",
                           kernel_initializer='uniform')(lay_actor)

        #critic网络
        lay_critic = lay_share
        for lay_num, (size, act) in enumerate(zip(self.training_param.kwargs_archi["Criticsizes"], self.training_param.kwargs_archi["Criticactivs"])):
            lay_critic = Dense(size, name="layer_critic_head_hidden{}".format(lay_num))(lay_critic)  # put at self.action_size全连接层
            lay_critic = Activation(act)(lay_critic)  # 激活层
        v_output = Dense(1, name="layer_critic_head_output")(lay_critic)

        self.model_policy_head = Model(inputs=[input_layer], outputs=[soft_proba],name="Actor-{}".format(int(time.time())))
        self.model_critic_head = Model(inputs=[input_layer], outputs=[v_output],name="Critic-{}".format(int(time.time())))
        self.model             = Model(inputs=[input_layer], outputs=[soft_proba, v_output],name="A2C-{}".format(int(time.time())))
        # self.model.compile(optimizer=Adam(lr=self.lr_,clipnorm = 0.5),
        #                           loss=self.actor_loss(),
        #                           run_eagerly=True)
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
    def train(self, s_batch, a_batch, r_batch, s2_batch, d_batch, pa_batch,logdir):
        """Trains networks to fit given parameters"""
        #求GAE
        donee = tf.Variable(d_batch, dtype=tf.float32)
        done = tf.expand_dims((1.0 - donee), axis=1)  # done为0，非done为1，用于筛选非done
        last_state = np.expand_dims(s2_batch[-1], axis=0)
        adv_batch = self.get_advantages(s_batch,done,r_batch,last_state)
        self.advs = adv_batch

        #求V_target
        V_last = self.model_critic_head.predict(last_state)
        r_batch[-1] += self.training_param.gama * done[-1] * V_last
        V_target = self.calculate_returns(r_batch, done)
        act_batch = tf.one_hot(a_batch, self.action_size)
        a_prob = self.model_policy_head.predict(s_batch)

        #编译
        self.model.compile(optimizer=Adam(learning_rate=self.lr_, clipnorm=0.5),
                            loss=[self.custom_loss(), mean_squared_error],
                            loss_weights=self.training_param.loss_weight,  # 选critic_loss小的，entropy大的，adv正且大的，即actor_loss  的
                            run_eagerly=True)

        # 定义callback类
        class MyCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.losses = []
                return

            def on_batch_end(self, batch, logs={}):  # batch 为index, logs为当前batch的日志acc, loss...
                self.losses.append(logs.get('loss'))
                return
        #训练：fit
        # loss = self.model.fit(x=s_batch,y={'layer_actor_head_output':act_batch, 'tf_op_layer_Reshape':V_target,'layer_actor_head_output_1':a_prob},batch_size=32,epochs=600,shuffle=true)
        cb = MyCallback()
        tensorboard = TensorBoard(log_dir=logdir)
        combined = np.append(act_batch, adv_batch, axis=1)
        h = self.model.fit(x=s_batch, y={'layer_actor_head_output':np.append(act_batch,adv_batch,axis = 1), 'layer_critic_head_output':V_target},
                            batch_size=32, epochs=3, shuffle=true,callbacks=[cb])
        # loss = self.model.evaluate(x=s_batch, y=[np.append(act_batch,adv_batch,axis = 1), V_target,a_prob])
        # loss_actor = np.mean(cb.losses)
        # loss_v = 0
        loss_actor = h.history["layer_actor_head_output_loss"]
        # loss_v = cb.losses["layer_critic_head_output_loss"]
        loss_v = h.history["layer_critic_head_output_loss"]

        #训练：Train_on_batch
        # loss = self.model.train_on_batch(x=s_batch, y=[np.append(act_batch,adv_batch,axis = 1), V_target, a_prob])
        return tf.reduce_mean(loss_actor),tf.reduce_mean(loss_v)

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

    def load_network(self, path, name=None, ext="h5"):
        # nothing has changed
        path_model, path_policy_model, path_critic_model = self._get_path_model(path, name)
        self.model = load_model('{}.{}'.format(path_model, ext))
        self.model_policy_head = load_model('{}.{}'.format(path_policy_model, ext))
        self.model_critic_head = load_model('{}.{}'.format(path_critic_model, ext))
        print("Successfully loaded network from: {}".format(path))