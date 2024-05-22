import os
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct



from Common.trainingParam import TrainingParam
from FairDQN.replayBuffer import ReplayBuffer
from FairDQN.prioritized_replay_buffer import PrioritizedReplayBuffer
from FairDQN.prioritized_replay_buffer import RankPrioritizedReplayBuffer
from FairDQN.fairDQN_NN import DeepQ

class DeepQAgent(AgentWithConverter):
    def convert_obs(self, observation):
        li_vect = []
        for obs_attr_name in self._training_param.list_attr_obs:
            # v = observation._get_array_from_attr_name(obs_attr_name).astype(np.float32)#作用同下，模拟器的函数
            v = getattr(observation, obs_attr_name)
            li_vect.append(v)

        # evaluate the danger degree of line, and append it to line state vector as the 6th/last feature
        danger = 0.9
        self.thermal_limit_under400 = tf.Variable(self.env._thermal_limit_a < 400)
        obsrho = getattr(observation, "rho")
        danger_ = ((obsrho >= (danger - 0.05)) & self.thermal_limit_under400) | (obsrho >= danger)
        d_vect = tf.cast(danger_, dtype=float)
        li_vect.append(d_vect)

        return np.concatenate((li_vect),axis=0)

    def my_act(self, transformed_observation, reward, done=False):
        if self.deep_q is None:
            self.init_deep_q(transformed_observation)
        predict_movement_int, *_ = self.deep_q.predict_movement(transformed_observation.reshape(1, -1), epsilon=0.0)
        return int(predict_movement_int)

    def convert_act(self, action):
        return super().convert_act(action)

    def init_deep_q(self, transformed_observation):
        if self.deep_q is None:
            # the first time an observation is observed, I set up the neural network with the proper dimensions.
            if self.mode == "DQN":
                cls = DeepQ
            # elif self.mode == "DDQN":
            #     cls = DuelQ
            # elif self.mode == "SAC":
            #     cls = SAC
            else:
                raise RuntimeError("Unknown neural network named \"{}\". Supported types are \"DQN\", \"DDQN\" and "
                                   "\"SAC\"".format(self.mode))
            print("建立Q网络时的动作空间大小为{}".format(self.action_space.size()))
            observation_size = transformed_observation.shape[-1]
            print("状态空间大小为{}".format(observation_size))
            # for i in range(action_space.size()):
            #     print("\n第{}个动作为:".format(i))
            #     print(self.convert_act(i))
            self.deep_q = cls(self.action_space.size(), observation_size=observation_size,lr=self.lr,training_param=self._training_param)

    def __init__(self, env, action_space, mode="DQN", lr=1e-5, training_param=TrainingParam(),**kwargs_converters):
        # this function has been adapted.
        self.env = env
        self.deep_q = None
        self.mode = mode
        self.lr = lr
        self._training_param = training_param
        self.epoch_rewards = []  # 存储每轮reward
        self.epoch_alive = []  # 存储每轮存活步数
        self.loss_list = []  # 存储每步loss
        self.AE_loss_list = []
        self.GGF_score_list = []  # 存储每步GGF_score
        self.Reward_list = []  # 存储每步Reward
        self.TQ_list = []
        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct,**kwargs_converters)

        # and now back to the origin implementation
        # self.replay_buffer = ReplayBuffer(training_param.BUFFER_SIZE)                  #普通buffer
        # self.replay_buffer = PrioritizedReplayBuffer(training_param.BUFFER_SIZE,0.7)   #第二个参数为buffer_ALPHA
        self.replay_buffer = RankPrioritizedReplayBuffer(training_param.BUFFER_SIZE, self._training_param.buffer_ALPHA)    #时序优先级缓冲池


    def load(self, env, path):
        self.observation_size = self._training_param.get_obs_size(env, self._training_param.list_attr_obs) + env.n_line
        # self.observation_size = env.observation_space.size_obs()
        self.deep_q = DeepQ(self.action_space.size(), observation_size=self.observation_size, lr=self.lr,
                          training_param=self._training_param)
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

    # def save_loss(self, path, loss_list):
    #     if path is not None:
    #         if not os.path.exists(path):
    #             os.mkdir(path)
    #         lossname = "loss.csv"
    #         losspath = os.path.join(path, lossname)
    #         file_csv = codecs.open(losspath, 'w+', 'utf-8')  # 追加
    #         writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    #         for data in loss_list:
    #             writer.writerow(data)
    #
    # def save_GGF(self, path, GGF_score_list):
    #     if path is not None:
    #         if not os.path.exists(path):
    #             os.mkdir(path)
    #         GGFname = "GGF_score.csv"
    #         GGFpath = os.path.join(path, GGFname)
    #         file_csv = codecs.open(GGFpath, 'w+', 'utf-8')  # 追加
    #         writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    #         for data in GGF_score_list:
    #             # data = list(map(lambda x: [x], data))
    #             writer.writerow(data)

    def GGF_score2(self,num):
        s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.samplepart(num)
        # r = tf.constant(r_batch, dtype=tf.float32, shape=[num, 1, self._training_param.reward_size])   #r作为GGFscore衡量指标
        Q = self.deep_q.model_Qnet_head.predict(s_batch)
        selected_actions = tf.one_hot(a_batch, self.action_size)
        valid_action = tf.reshape(selected_actions, shape=[-1, self.action_size, 1])
        q_sa = Q * valid_action

        GGF_score = self.deep_q.GGF(q_sa[:, :, :self._training_param.n_user], self._training_param.n_user, 2)
        Reward = self._training_param.weight_coef * q_sa
        # print(tf.reduce_mean(GGF_score))
        return tf.reduce_mean(GGF_score),tf.reduce_mean(Reward)

    def GGF_score(self,total_reward):
        avg_Reward = tf.expand_dims(tf.Variable(total_reward, dtype=tf.float32), axis=0)
        GGF_score = self.deep_q.GGF(avg_Reward[:,:self._training_param.n_user], self._training_param.n_user, 2)
        # print(tf.reduce_mean(GGF_score))
        return tf.reduce_mean(GGF_score)


    def train(self,
              env,
              iterations,
              num_pre_training_steps,
              save_path,
              logdir=None,
              training_param=None,
              verbose=True
              ):

        if training_param is None:
            training_param = TrainingParam()

        # self._train_lr = training_param.lr
        #agent._init_把lr传进来了并self.lr = lr

        if self._training_param is None:
            self._training_param = training_param
        else:
            training_param = self._training_param
        # self.init_deep_q(self._training_param, env)  #第一次执行的时候初始化
        # self._fill_vectors(self._training_param)

        # self._init_replay_buffer()     #agent._init_已经创建了

        # efficient reading of the data (read them by chunk of roughly 1 day
        nb_ts_one_day = 24 * 60 / 5  # number of time steps per day
        env.set_chunk_size(int(max(100, nb_ts_one_day)))#优化数据读取过程，等于下面一行的函数，读一天大小的数据
        # self._set_chunk(env, nb_ts_one_day)

        # Create file system related vars
        if save_path is not None:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok=True)

        # if logdir is not None:
        #     logpath = os.path.join(logdir, self.name)
        #     self._tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        # else:
        #     logpath = None
        #     self._tf_writer = None
        # UPDATE_FREQ = training_param.update_tensorboard_freq  # update tensorboard every "UPDATE_FREQ" steps
        # SAVING_NUM = training_param.save_model_each

        # 生成writer
        # train_summary_writer = tf.summary.create_file_writer(logdir)
        train_summary_writer = SummaryWriter(logdir)

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
        step = 0
        epoch = 0
        done = True
        alive_steps = 0
        ambiguous_steps = 0
        epoch_actul = []
        total_reward = np.zeros(self._training_param.reward_size)
        total_Q_loss = 0
        total_AE_loss = 0
        with tqdm(total=iterations - step, disable=False, miniters = 500) as pbar:
            while step < iterations:
                # Init first time or new episode
                if done:
                    epoch += 1
                    obs = env.reset()  # This shouldn't raise
                    state = self.convert_obs(obs)
                    if self.deep_q is None:
                        self.init_deep_q(state)
                if verbose and step % 10 == 0:
                    print("Epoch[{}]--Step [{}] -- Random [{}]".format(epoch, step, self.epsilon))

                # Choose an action
                if step < num_pre_training_steps:
                    a = self.deep_q.random_move()
                else:
                    a, _ ,_= self.deep_q.predict_movement(state, self.epsilon)

                # Convert it to a valid action
                # print("此次选择的动作是{}".format(a))
                act = self.convert_act(a)
                # print("动作具体是{}".format(act))
                # Execute action
                new_obs, reward, done, info = env.step(act)

                new_state = self.convert_obs(new_obs)
                if info["is_illegal"] or info["is_ambiguous"] or \
                        info["is_dispatching_illegal"] or info["is_illegal_reco"]:
                    ambiguous_steps += 1
                    if verbose :
                        print(a, info)

                # Save to experience buffer
                self.replay_buffer.add(state, a, reward, done, new_state)
                total_reward += reward
                # total_reward = reward + self._training_param.gama * total_reward

                # Perform training when we have enough experience in buffer
                # if step > self._training_param.MINIBATCH_SIZE:      #预训练随机探索同时训练网络
                if step > num_pre_training_steps:                    #预训练仅探索不训练网络
                    # #普通buffer时
                    # s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(
                    #     self._training_param.MINIBATCH_SIZE)
                    # loss = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch,
                    #                                      self._training_param.MINIBATCH_SIZE)
                    #PrioritizedReplayBuffer时
                    s_batch, a_batch, r_batch, d_batch, s2_batch, w_batch, idx_batch = self.replay_buffer.sample(
                        self._training_param.MINIBATCH_SIZE,0.5)  #0.5为采样时使用重要性权重的程度，0为不使用，1为全部使用，D3QN默认0.5，fair-RL由从0到1的线性插值得到
                    loss_Q, loss_AE, priorities = self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, w_batch, idx_batch,
                                             self._training_param.MINIBATCH_SIZE)
                    self.replay_buffer.update_priorities(idx_batch, priorities)    #只有时序时不需要

                    total_Q_loss += loss_Q
                    total_AE_loss += loss_AE
                    self.loss_list.append(loss_Q)
                    self.AE_loss_list.append(loss_AE)
                    if verbose and step % 5 == 0:
                        print("loss =", loss_Q)
                    # trial.report(loss, step)          #调参时剪枝优化
                    GGFscore,Reward = self.GGF_score2(20)
                    self.GGF_score_list.append([float(GGFscore)])
                    self.Reward_list.append([float(Reward)])
                    self.TQ_list.append([float(GGFscore+Reward)])
                    # if verbose and step % 5 == 0:
                    #     print("GGF_score =", GGFscore)
                    #     print("Reward =", Reward)
                    # # 保存结果，便于用tensorboard查看
                    # train_summary_writer = tf.summary.create_file_writer(logdir)
                    # with train_summary_writer.as_default():
                    #     tf.summary.scalar('loss', loss, step=step)
                    #     tf.summary.scalar('GGF_score', GGFscore, step=step)
                    #     tf.summary.scalar('Reward', Reward, step=step)
                    #     tf.summary.scalar('TQ', GGFscore + Reward, step=step)
                    train_summary_writer.add_scalar('Loss', np.array(loss_Q), global_step=step)
                    # train_summary_writer.add_scalar('Step-AE_loss', np.array(loss_AE), global_step=step)
                    train_summary_writer.add_scalar('GGF_score', np.array(GGFscore), global_step=step)
                    train_summary_writer.add_scalar('Reward', np.array(Reward), global_step=step)
                    # train_summary_writer.add_scalar('TQ', np.array(GGFscore + Reward), global_step=step)


                if done:                   #每次epoch结束输出结果
                    Epoch_GGF_score = self.GGF_score(total_reward)
                    # self.GGF_score_list.append([float(Epoch_GGF_score)])
                    epoch_Reward = np.dot(total_reward, self._training_param.weight_coef)
                    self.epoch_rewards.append(epoch_Reward)
                    self.epoch_alive.append(alive_steps)
                    actul_step = alive_steps-ambiguous_steps
                    # print(actul_step)
                    epoch_actul.append(actul_step)
                    epoch_Q_loss = total_Q_loss/(alive_steps+1e-8)
                    epoch_AE_loss = total_AE_loss/(alive_steps+1e-8)
                    if verbose:
                        print("Survived [{}] steps".format(alive_steps))
                        print("Total reward [{}]".format(epoch_Reward))
                        print("Epoch[{}]--Step [{}] -- Random [{}]".format(epoch, step, self.epsilon))
                    # # 保存结果，便于用tensorboard查看
                    # train_summary_writer = tf.summary.create_file_writer(logdir)
                    # with train_summary_writer.as_default():
                    #     tf.summary.scalar('epoch_alive', alive_steps, step=epoch)
                    #     tf.summary.scalar('epoch_actul', actul_step, step=epoch)
                    #     tf.summary.scalar('epoch_Reward', epoch_Reward, step=epoch)
                    # train_summary_writer.add_scalar('Q-loss', np.array(epoch_Q_loss), global_step=epoch)
                    # train_summary_writer.add_scalar('AE_loss', np.array(epoch_AE_loss), global_step=epoch)
                    # train_summary_writer.add_scalar('Loss', np.array(self._training_param.loss_weight[0] * epoch_Q_loss +
                    #                                                  self._training_param.loss_weight[1] * epoch_AE_loss ),global_step=epoch)
                    # train_summary_writer.add_scalar('GGF_score', np.array(Epoch_GGF_score), global_step=epoch)
                    # train_summary_writer.add_scalar('epoch_alive', np.array(alive_steps), global_step=epoch)
                    # train_summary_writer.add_scalar('epoch_actul', np.array(actul_step), global_step=epoch)
                    # train_summary_writer.add_scalar('Reward', np.array(epoch_Reward), global_step=epoch)
                    alive_steps = 0
                    ambiguous_steps = 0
                    total_reward = np.zeros(self._training_param.reward_size)
                    total_Q_loss = 0
                    total_AE_loss = 0
                else:
                    alive_steps += 1


                # Save the network every 100 iterations
                if step % self._training_param.SAVING_NUM == 0:
                    self.deep_q.target_train()
                    # self.save(save_path)


                # Iterate to next loop
                step += 1
                if step > num_pre_training_steps:
                    if self.epsilon > self._training_param.final_epsilon:
                        self.epsilon -= (self._training_param.initial_epsilon-self._training_param.final_epsilon)/self._training_param.epsilon_decay
                obs = new_obs
                state = new_state
                pbar.update(1)

            #迭代结束
            self.save(save_path)
            # self.save_loss(save_path, self.loss_list)
            # self.save_GGF(save_path, self.GGF_score_list)

            #画图
            fig = plt.figure(1,figsize=(8,8))
            #画Loss图和GGF-score图
            ax1 = plt.subplot(2, 1, 1)
            # x1 = range(self._training_param.MINIBATCH_SIZE, len(loss_list)+self._training_param.MINIBATCH_SIZE)   #预训练随机探索同时训练网络
            x_step = range(num_pre_training_steps, len(self.loss_list) + num_pre_training_steps)  # 预训练仅探索不训练网络
            # x2 = range(len(GGF_score_list))
            # x2 = range(iterations)
            # y = np.array(loss_list)
            y_loss = self.loss_list
            y_GGFScore = self.GGF_score_list
            y_Reward = self.Reward_list
            y_TQ = self.TQ_list
            plt.plot(x_step, y_loss, label='loss value')
            plt.plot(x_step, y_GGFScore, label='GGF score')
            plt.plot(x_step, y_Reward, label='Reward')
            plt.plot(x_step, y_TQ, label='y_TQ')
            plt.legend(loc="best")
            plt.grid(True)
            # plt.legend(['loss value','GGF score'])
            plt.xlabel('step')
            plt.ylabel('value')
            plt.title('Loss Value-RankPriority')

            # 画每个epoch的reward图和存活步数图
            ax1 = plt.subplot(2,1,2)
            x_epoch = range(len(epoch_actul))
            y_reward  = self.epoch_rewards
            y_alive = self.epoch_alive
            y_epoch_actul = epoch_actul
            plt.plot(x_epoch, y_reward, label='epoch_rewards')
            plt.plot(x_epoch, y_alive,"r--",linewidth = "2", label='epoch_alive')
            plt.plot(x_epoch, y_epoch_actul,color="black", linestyle="dashdot", label='epoch_alive')
            plt.legend(['reward 1','reward 2','reward 3','reward 4','reward 5','reward 6','reward 7','epoch_alive','actul_alive'],loc="best",fontsize=8)
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('rewards')
            plt.title('rewards Value')
            plt.show()
            # 输出epoch平均reward和平均alive
            mean_loss = np.mean(self.loss_list)            #用于调参
            mean_GGF = np.mean(self.GGF_score_list)        #用于调参
            mean_reward = np.mean(self.epoch_rewards)
            mean_alive = np.mean(self.epoch_alive)
            mean_actual_alive = np.mean(epoch_actul)
            if len(self.epoch_rewards) >= 100:
                mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                mean_alive_100 = np.mean(self.epoch_alive[-100:])
                mean_actual_alive_100 = np.mean(epoch_actul[-100:])
            else:
                mean_reward_100 = mean_reward
                mean_alive_100 = mean_alive
                mean_actual_alive_100 = mean_actual_alive
            print("mean_reward: {}".format(mean_reward))
            print("mean_alive: {}".format(mean_alive))
            print("mean_reward_100: {}".format(mean_reward_100))
            print("mean_alive_100: {}".format(mean_alive_100))
            print("mean_actual_alive_100: {}".format(mean_actual_alive_100))

        return self.loss_list,self.GGF_score_list
