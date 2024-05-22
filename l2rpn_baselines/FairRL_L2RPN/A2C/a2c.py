import os
import tensorflow as tf
from tqdm import tqdm
import warnings
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct



from l2rpn_baselines.FairRL_L2RPN.Common.trainingParam import TrainingParam
from l2rpn_baselines.FairRL_L2RPN.A2C.a2c_NN import A2C

class Agent(AgentWithConverter):
    def convert_obs(self, observation):
        li_vect = []
        for obs_attr_name in self._training_param.list_attr_obs:
            # v = observation._get_array_from_attr_name(obs_attr_name).astype(np.float32)#作用同下，模拟器的函数
            v = getattr(observation, obs_attr_name)
            li_vect.append(v)
        return np.concatenate((li_vect),axis=0)

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation.reshape(1, -1), epsilon=0.0)
        return int(predict_movement_int)

    def convert_act(self, action):
        return super().convert_act(action)

    # def _filter_action(self, action):
    #     MAX_ELEM = 3
    #     act_dict = action.impact_on_objects()
    #     elem = 0
    #     elem += act_dict["force_line"]["reconnections"]["count"]
    #     elem += act_dict["force_line"]["disconnections"]["count"]
    #     elem += act_dict["switch_line"]["count"]
    #     elem += len(act_dict["topology"]["bus_switch"])
    #     elem += len(act_dict["topology"]["assigned_bus"])
    #     elem += len(act_dict["topology"]["disconnect_bus"])
    #     elem += len(act_dict["redispatch"]["generators"])
    #
    #     if elem <= MAX_ELEM:
    #         return True
    #     return False

    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "DQN":
                cls = DeepQ
            elif self.mode == "A2C":
                cls = A2C
            elif self.mode == "SAC":
                cls = SAC
            else:
                raise RuntimeError("Unknown neural network named \"{}\". Supported types are \"DQN\", \"DDQN\" and "
                                    "\"SAC\"".format(self.mode))
            # for i in range(action_space.size()):
            #     print("\n第{}个动作为:".format(i))
            #     print(self.convert_act(i))
            self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[-1],lr=self.lr,training_param=self._training_param)

    def __init__(self, action_space, mode="A2C", lr=1e-5, training_param=TrainingParam(),**kwargs_converters):
        # this function has been adapted.

        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct,**kwargs_converters)
        #self.action_space.filter_action(self._filter_action)
        print("转换后动作空间大小为{}".format(self.action_space.size()))

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.deep_q = None
        self.mode = mode
        self.lr = lr
        self._training_param = training_param
        self.epoch_rewards = []  #存储平均每条轨迹reward
        self.epoch_alive = []    #存储平均每条存活步数
        self.loss_actor_list = []      #存储每轮loss
        self.loss_critic_list = []
        self.GGF_score_list = [] #存储每轮GGF_score
        self.Reward_list = []
        self.TQ_list = []

    def load(self, path):
        # not modified compare to original implementation
        self.deep_q.load_network(path)

    def save(self, path):
        if path is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            nm_conv = "action_space.npy"
            conv_path = os.path.join(path, nm_conv)
            if not os.path.exists(conv_path):
                self.action_space.save(path=path, name=nm_conv)

            self._training_param.save_as_json(path, name="training_params.json")
            # self._nn_archi.save_as_json(tmp_me, "nn_architecture.json")
            self.deep_q.save_network(path, name=self.mode)

    def GGF_score(self,total_reward):
        avg_Reward = tf.expand_dims(tf.Variable(total_reward, dtype=tf.float32), axis=0)
        GGF_score = self.deep_q.GGF(avg_Reward[:,:self._training_param.n_user], self._training_param.n_user, 2)
        # print(tf.reduce_mean(GGF_score))
        return tf.reduce_mean(GGF_score)

    def train(self,
            env,
            epochs,
            iterations,
            save_path,
            logdir=None,
            training_param=None,
            verbose=True
            ):

        if training_param is None:
            training_param = TrainingParam()

        if self._training_param is None:
            self._training_param = training_param
        else:
            self.training_param = self._training_param

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        env.set_chunk_size(int(max(100, nb_ts_one_day)))#优化数据读取过程，等于下面一行的函数，读一天大小的数据
        # self._set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        if hasattr(env, "nb_env"):
            nb_env = env.nb_env
            warnings.warn("Training using {} environments".format(nb_env))
            self.__nb_env = nb_env
        else:
            self.__nb_env = 1

        self.epsilon = self._training_param.initial_epsilon

        # Compute dimensions from intial spaces
        self.observation_size = self._training_param.get_obs_size(env, self._training_param.list_attr_obs)
        self.action_size = self.action_space.size()

        #循环中的计数器初始化
        epoch_actul = []

        with tqdm(total=epochs*iterations, disable=False, miniters=1, mininterval=3) as pbar:
            for epoch in range(epochs):
                obs = env.reset()
                state = self.convert_obs(obs)
                if self.deep_q is None:
                    self.init_deep_q(state)
                states = []      #buffer
                actions = []
                new_states = []
                dones = []
                rewards = []
                actions_probs = []
                alive_steps = 0
                ambiguous_steps = 0
                # total_reward = 0    #计数iteration步所有reward总和
                num_done = 0        #计数iteration步有多少个完整轨迹
                total_reward = np.zeros(self._training_param.reward_size)

                for step in range(iterations):
                    a, a_prob = self.deep_q.predict_movement(state, self.epsilon)
                    act = self.convert_act(a)
                    new_obs, reward, done, info = env.step(act)

                    new_state = self.convert_obs(new_obs)
                    if info["is_illegal"] or info["is_ambiguous"] or \
                            info["is_dispatching_illegal"] or info["is_illegal_reco"]:
                        ambiguous_steps += 1
                        if verbose:
                            print(a, info)

                    # 记录一条轨迹
                    states.append(state)
                    actions.append(a)
                    rewards.append(reward)
                    dones.append(done)
                    new_states.append(new_state)
                    actions_probs.append(a_prob)
                    alive_steps += 1
                    if done:
                        obs = env.reset()
                        state = self.convert_obs(obs)
                        num_done +=1
                    else:
                        state = new_state
                    total_reward += reward
                    pbar.update(1)
                #完成一个epoch
                #统计该epoch有多少个轨迹
                if not done:
                    num_done += 1
                total_reward_per_epoch = total_reward/num_done
                self.epoch_rewards.append(total_reward_per_epoch)
                self.epoch_alive.append(alive_steps/num_done)
                actul_step = alive_steps - ambiguous_steps
                epoch_actul.append(actul_step/num_done)

                # 回合更新
                combined_rewards = np.dot(np.array(rewards), self._training_param.weight_coef)
                loss_actor, loss_critic = self.deep_q.train(np.array(states), np.array(actions), np.array(combined_rewards),
                                                            np.array(new_states), np.array(dones),
                                                            np.array(actions_probs),logdir)
                self.loss_actor_list.append(loss_actor)
                self.loss_critic_list.append(loss_critic)

                # GGFscore = self.GGF_score(total_reward_per_epoch)
                # self.GGF_score_list.append([float(GGFscore)])
                Reward = np.dot(total_reward_per_epoch, self._training_param.weight_coef)
                self.Reward_list.append(Reward)
                # self.TQ_list.append([float(GGFscore + Reward)])
                if verbose:
                    print("loss_actor =", loss_actor)
                    print("loss_critic =", loss_critic)
                    #print("GGF_score =", GGFscore)
                # 保存结果，便于用tensorboard查看
                train_summary_writer = tf.summary.create_file_writer(logdir)
                with train_summary_writer.as_default():
                    tf.summary.scalar('actor_loss', loss_actor, step=epoch)
                    tf.summary.scalar('critic_loss', loss_critic, step=epoch)
                    #tf.summary.scalar('GGF_score', GGFscore, step=epoch)
                    tf.summary.scalar('Reward',Reward, step=epoch)
                    tf.summary.scalar('epoch_alive', alive_steps/num_done, step=epoch)
                    tf.summary.scalar('epoch_actul', actul_step/num_done, step=epoch)
                    # tf.summary.scalar('TQ', GGFscore + Reward, step=epoch)
                # 保存模型
                if epoch % self._training_param.SAVING_NUM == 0:
                    self.save(save_path)



    #     # 画图
    #     fig = plt.figure(1, figsize=(8, 8))
    #     # 画Loss图和GGF-score图
    #     ax1 = plt.subplot(3, 1, 1)
    #     # x1 = range(self._training_param.MINIBATCH_SIZE, len(loss_list)+self._training_param.MINIBATCH_SIZE)   #预训练随机探索同时训练网络
    #     # x_step = range(num_pre_training_steps, len(self.loss_list) + num_pre_training_steps)  # 预训练仅探索不训练网络
    #     x = range(len(self.loss_actor_list))
    #     # x2 = range(iterations)
    #     # y = np.array(loss_list)
    #     y_actor_loss = self.loss_actor_list
    #     # y_sum_loss = np.array(self.loss_actor_list)+np.array(self.loss_critic_list)
    #     #y_GGFScore = self.GGF_score_list
    #     plt.plot(x, y_actor_loss, label='actor_loss')
    #     #plt.plot(x, y_GGFScore, label='y_GGFScore')
    #     # plt.plot(x, y_sum_loss, label='Sumloss')
    #     # plt.plot(x, y_Reward, label='Reward')
    #     # plt.plot(x, y_TQ, label='y_TQ')
    #     # plt.legend(['loss', 'actor_loss', 'critic_loss', 'entropy', 'GGF_score'],
    #     #            loc="best", fontsize=8)
    #     plt.legend(loc="best", fontsize=8)
    #     plt.grid(True)
    #     # plt.legend(['loss value','GGF score'])
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.title('50*500,loss1:0.01:1,Loss Value')
    #
    #     ax1 = plt.subplot(3, 1, 2)
    #     x = range(len(self.loss_actor_list))
    #     y_critic_loss = self.loss_critic_list
    #     plt.plot(x, y_critic_loss, label='critic_loss')
    #     plt.legend(loc="best", fontsize=8)
    #     plt.grid(True)
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #
    #     # 画每个epoch的reward图和存活步数图
    #     ax2 = plt.subplot(3, 1, 3)
    #     x_epoch = range(epochs)
    #     # y_reward  = self.epoch_rewards
    #     y_alive = self.epoch_alive
    #     y_epoch_actul = epoch_actul
    #     y_Reward = self.Reward_list
    #     # y_TQ = self.TQ_list
    #     # plt.plot(x_epoch, y_reward, label='epoch_rewards')
    #     plt.plot(x_epoch, y_Reward, label='Reward')
    #     # plt.plot(x_epoch, y_TQ, label='y_TQ')
    #     plt.plot(x_epoch, y_alive, "r--", linewidth="2", label='epoch_alive')
    #     plt.plot(x_epoch, y_epoch_actul, color="black", linestyle="dashdot", label='epoch_actul')
    #     # plt.legend(['reward 1','reward 2','reward 3','reward 4','reward 5','reward 6','reward 7','epoch_alive','actul_alive'],loc="best",fontsize=8)
    #     plt.legend(loc="best")
    #     plt.grid(True)
    #     plt.xlabel('epoch')
    #     plt.ylabel('rewards')
    #     plt.title('rewards Value')
    #     plt.show()
    #     # 输出epoch平均reward和平均alive
    #     # mean_loss = np.mean(self.loss_list)            #用于调参
    #     # mean_GGF = np.mean(self.GGF_score_list)        #用于调参
    #     mean_reward = np.mean(self.epoch_rewards)
    #     mean_alive = np.mean(self.epoch_alive)
    #     mean_actual_alive = np.mean(epoch_actul)
    #     if len(self.epoch_rewards) >= 100:
    #         mean_reward_100 = np.mean(self.epoch_rewards[-100:])
    #         mean_alive_100 = np.mean(self.epoch_alive[-100:])
    #         mean_actual_alive_100 = np.mean(epoch_actul[-100:])
    #     else:
    #         mean_reward_100 = mean_reward
    #         mean_alive_100 = mean_alive
    #         mean_actual_alive_100 = mean_actual_alive
    #     print("mean_reward: {}".format(mean_reward))
    #     print("mean_alive: {}".format(mean_alive))
    #     print("mean_reward_100: {}".format(mean_reward_100))
    #     print("mean_alive_100: {}".format(mean_alive_100))
    #     print("mean_actual_alive_100: {}".format(mean_actual_alive_100))
    #     # print("loss_actor_list:{}".format(self.loss_actor_list))
    #
    # # return mean_loss,mean_GGF