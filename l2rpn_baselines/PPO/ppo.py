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
import sys
sys.path.append(r'/home/fly/zyh')


from Common.trainingParam import TrainingParam
from PPO.ppo_NN import PPO

class Agent(AgentWithConverter):
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
            elif self.mode == "A2C":
                cls = A2C
            elif self.mode == "SAC":
                cls = SAC
            elif self.mode == "PPO":
                cls = PPO
            else:
                raise RuntimeError("Unknown neural network named \"{}\". Supported types are \"DQN\", \"DDQN\" and "
                                    "\"SAC\"".format(self.mode))
            print("建立网络时的动作空间大小为{}".format(self.action_space.size()))
            # for i in range(action_space.size()):
            #     print("\n第{}个动作为:".format(i))
            #     print(self.convert_act(i))
            self.deep_q = cls(self.action_space.size(), observation_size=transformed_observation.shape[-1],lr=self.lr,training_param=self._training_param)

    def __init__(self, env, action_space, mode="PPO", lr=1e-5, training_param=TrainingParam(),**kwargs_converters):
        # this function has been adapted.

        # to built a AgentWithConverter, we need an action_space.
        # No problem, we add it in the constructor.
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct,**kwargs_converters)
        print("转换后动作空间大小为{}".format(self.action_space.size()))
        # and now back to the origin implementation

        # compare to original implementation, i don't know the observation space size.
        # Because it depends on the component of the observation we want to look at. So these neural network will
        # be initialized the first time an observation is observe.
        self.env = env
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



        #循环中的计数器初始化
        epoch_actul = []

        with tqdm(total=epochs*iterations, disable=False, miniters=1, mininterval=120) as pbar:
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
                # rewards = np.dot(np.array(rewards),self._training_param.weight_coef)
                loss_actor, loss_critic, loss_AE = self.deep_q.train(np.array(states), np.array(actions), np.array(combined_rewards),
                                                            np.array(new_states), np.array(dones),
                                                            np.array(actions_probs))
                self.loss_actor_list.append(loss_actor)
                self.loss_critic_list.append(loss_critic)

                GGFscore = self.GGF_score(total_reward_per_epoch)
                self.GGF_score_list.append([float(GGFscore)])
                Reward = np.dot(total_reward_per_epoch, self._training_param.weight_coef)
                self.Reward_list.append(Reward)
                # self.TQ_list.append([float(GGFscore + Reward)])
                if verbose:
                    print("loss_actor =", loss_actor)
                    print("loss_critic =", loss_critic)
                    print("loss_AE =", loss_AE)
                    print("GGF_score =", GGFscore)
                # 保存结果，便于用tensorboard查看
                # with train_summary_writer.as_default():
                #     tf.summary.scalar('actor_loss', loss_actor, step=epoch)
                #     tf.summary.scalar('critic_loss', loss_critic, step=epoch)
                #     tf.summary.scalar('AE_loss', loss_AE, step=epoch)
                #     tf.summary.scalar('Loss', self._training_param.loss_weight[0] * loss_actor +
                #                       self._training_param.loss_weight[1] * loss_critic +
                #                       self._training_param.loss_weight[2] * loss_AE, step=epoch)
                #     tf.summary.scalar('GGF_score', GGFscore, step=epoch)
                #     tf.summary.scalar('Reward',Reward, step=epoch)
                #     tf.summary.scalar('epoch_alive', alive_steps/num_done, step=epoch)
                #     tf.summary.scalar('epoch_actul', actul_step/num_done, step=epoch)

                train_summary_writer.add_scalar('actor_loss', np.array(loss_actor), global_step=epoch)
                train_summary_writer.add_scalar('critic_loss', np.array(loss_critic), global_step=epoch)
                train_summary_writer.add_scalar('AE_loss', np.array(loss_AE), global_step=epoch)
                train_summary_writer.add_scalar('Loss', np.array(self._training_param.loss_weight[0] * loss_actor +
                                                                self._training_param.loss_weight[1] * loss_critic +
                                                                self._training_param.loss_weight[2] * loss_AE), global_step=epoch)
                train_summary_writer.add_scalar('GGF_score', np.array(GGFscore), global_step=epoch)
                train_summary_writer.add_scalar('Reward', np.array(Reward), global_step=epoch)
                train_summary_writer.add_scalar('epoch_alive', np.array(alive_steps / num_done), global_step=epoch)
                train_summary_writer.add_scalar('epoch_actul', np.array(actul_step / num_done), global_step=epoch)

                #保存模型
                # if epoch % self._training_param.SAVING_NUM == 0:
                #     self.save(save_path)



        # 迭代结束
        self.save(save_path)
        # self.save_loss(save_path, self.loss_list)
        # # self.save_GGF(save_path, self.GGF_score_list)
        # 加平滑（样本足够多，不需要再平滑）
        # results = np.zeros(len(self.Reward_list))
        # for i in range(4, len(self.Reward_list)):
        #     m = self.Reward_list[(i - 4):i]
        #     results[i] = np.mean(m)



        
        # 输出epoch平均reward和平均alive
        # mean_loss = np.mean(self.loss_list)            #用于调参
        # mean_GGF = np.mean(self.GGF_score_list)        #用于调参
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
        # print("loss_actor_list:{}".format(self.loss_actor_list))

    # return mean_loss,mean_GGF
