try:
    import grid2op
    import threading
    import numpy as np
    import time
    import json
    import copy
    import os
    from grid2op import make
    from grid2op.Agent import MLAgent
    from grid2op.Environment import Environment
    from grid2op.Parameters import Parameters
    from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward
    from grid2op.Agent import AgentWithConverter
    from grid2op.Converter import IdToAct

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    import tensorflow

    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.python.keras.backend as K
    from l2rpn_baselines.AsynchronousActorCritic_init.user_environment_make import set_environement
    # import user_environment_make
except ImportError as exc_:
    raise ImportError("AsynchronousActorCritic baseline impossible to load the required dependencies for training the model. The error was: \n {}".format(exc_))


# import user_environment_make

# Create the Agent instance here that can used with the Runner to test the performance of the trained RL agent.
class A3CAgent(AgentWithConverter):
    # first change: An Agent must derived from grid2op.Agent (in this case MLAgent, because we manipulate vector instead
    # of classes) We will use this template to create our desired ML agent with unique neural network configuration.
    def __init__(self, state_size, action_space, env_name,
                 profiles_chronics, EPISODES_train2, time_step_end2, Hyperparameters, Thread_count, train_flag,save_path):
        AgentWithConverter.__init__(self, action_space, action_space_converter=IdToAct)
        # Parameter settings.
        # NOTE: MAKE SURE THE FOLLOWING SETTINGS ARE SAME AS THE TRAINED AGENT OR THE WEIGHTS WONT LOAD SUCCESSFULLY.
        # get size of state and action
        self.state_size = state_size
        self.action_size = self.action_space.size()
        self.action_space = self.action_space
        # get gym environment name
        self.env_name = env_name

        if train_flag:
            # these are hyper parameters for the A3C
            self.actor_lr = Hyperparameters["actor_learning_rate"]
            self.critic_lr = Hyperparameters["critic_learning_rate"]
            self.discount_factor = Hyperparameters["discount_factor"]
            self.threads = Thread_count
            self.save_path = os.path.abspath(save_path)
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
        else: #evaluating
            self.action_list = []

        self.hidden1, self.hidden2 = Hyperparameters["size_of_hidden_layer_1"], Hyperparameters["size_of_hidden_layer_2"]

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        if train_flag:
            # method for training actor and critic network
            self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

            # global variables for threading
            global scores
            scores = []
            global alive_step
            alive_step = []
            global time_step_end
            time_step_end = time_step_end2
            global EPISODES_train
            EPISODES_train = EPISODES_train2

            self.profiles_chronics = profiles_chronics

        # 使用TensorFlow库创建一个交互式会话，并配置了一个不包含GPU的计算设备。然后使用Keras的K.set_session()方法将该会话设置为Keras的默认会话，并运行了全局变量的初始化操作。
        #self.sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU': 0}))
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())


    def my_act(self, state, reward, done=False):
        # state = state.to_vect()  # 直接使用原始状态向量
        state = np.array(state)
        state = state.reshape([1, state.size])
        action = self.get_action(state)
        return action

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer. These neural networks should be same as the neural network
    # from "train.py".
    def build_model(self):
        state = Input(batch_shape=(None,  self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)
        # action_prob = K.softmax(action_intermediate)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        return actor, critic

    def act(self, state, reward, done=False):
        #state = state.to_vect()  # 直接使用原始状态向量
        state = np.array(state)
        state = state.reshape([1, state.size])
        action = self.get_action(state)
        return action

    def get_action(self, state):
        # 使用神经网络预测动作概率
        policy = self.actor.predict(state, batch_size=1).flatten()

        # 随机选择动作
        action_index = np.random.choice(self.action_size, p=policy)
        return action_index

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None,))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01 * entropy

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        gradients = tf.gradients(actor_loss, self.actor.trainable_weights)
        grads_and_vars = zip(gradients, self.actor.trainable_weights)
        optimizer.apply_gradients(grads_and_vars)
        train = K.function([self.actor.input, action, advantages], outputs=[K.constant([0,1])], updates=[])
        return train

    # 定义 Critic 的优化器
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        gradients = tf.gradients(loss, self.critic.trainable_weights)
        grads_and_vars = zip(gradients, self.critic.trainable_weights)
        optimizer.apply_gradients(grads_and_vars)
        train = K.function([self.critic.input, discounted_reward], outputs=[K.constant([0,1])], updates=[])
        return train

    # make agents(local) and start training
    def train(self, nn_weights_name):
        agents = [Agent(i, self.actor, self.critic, self.optimizer, self.env_name, self.discount_factor,
                        self.action_space, self.state_size, self.profiles_chronics, self.sess) for i in range(self.threads)]

        for agent in agents:
            agent.start()

        while (len(scores) < EPISODES_train):
            time.sleep(200) # main thread saves the model every 200 sec
            print("len(scores) = ", len(scores))
            if (len(scores)>10):
                self.save_model(nn_weights_name,self.save_path)
                print("_______________________________________________________________________________________________________")
                print("saved NN model at episode", episode, "\n")
                print("_______________________________________________________________________________________________________")

    def load_model(self, nn_weight_name, load_path):
            self.actor.load_weights(os.path.join(load_path,nn_weight_name + "_actor.h5"))
            self.critic.load_weights(os.path.join(load_path,nn_weight_name + "_critic.h5"))

    def save_model(self, nn_weight_name, save_path):
        self.actor.save_weights(os.path.join(save_path,nn_weight_name + "_actor.h5"))
        self.critic.save_weights(os.path.join(save_path,nn_weight_name + "_critic.h5"))

# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, actor, critic, optimizer, env_name, discount_factor, action_space, state_size, profiles_chronics, session):
        threading.Thread.__init__(self)

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_space.size()
        self.state_size = state_size
        self.session = session
        self.action_space = action_space

        self.profiles_chronics = profiles_chronics

        # Agent类的__init__方法中，创建一个独立的日志写入对象
        self.tf_writer = tf.compat.v1.summary.FileWriter("logs-train-gpu/thread_" + str(index))

    # Thread interactive with environment
    def run(self):
        global episode
        episode = 0
        env = set_environement(self.index,self.env_name,self.profiles_chronics)
        while episode < EPISODES_train:
            state = env.reset()
            # state = copy.deepcopy(np.reshape(state.to_vect(), [1, self.state_size]))
            state = self.convert_obs(state)
            state = state.reshape([1, state.size])
            score = 0
            time_step = 0
            max_action = 0
            non_zero_actions = 0
            epsilon = 0.5
            while True:
                # Decaying epsilon greedy. Not the best one. This agent needs a better exploration strategy to help
                # it learn to perform well.
                if np.random.random() < epsilon*(1/(episode/400+1)):
                    action_index = int(np.random.choice(self.action_size))
                    epison_flag = True
                else:
                    epison_flag = False
                    if time_step%1 == 0:# or max(state_as_dict.rho)>0.75:
                        action_index = self.get_action(state)
                    else:
                        action_index = 0

                convert_instance = convert(env.observation_space, self.action_space)
                action = convert_instance.convert_act(action_index)
                next_state, reward, done, flag = env.step(action)
                time_hour = next_state.day*10000 + next_state.hour_of_day * 100 + next_state.minute_of_hour
                # next_state = np.reshape(next_state.to_vect(), [1, self.state_size]) if not done else np.zeros([1, self.state_size])
                next_state = self.convert_obs(next_state)
                next_state = np.array(next_state)
                next_state = next_state.reshape([1,next_state.size])
                # next_state = observation_space.array_to_observation(next_state).as_minimalist().as_array()
                # score += (reward-0.1*(next_state[1]*next_state[1]+next_state[3]*next_state[3])) # reducing the reward based on speed...
                score += reward if not done else -100*(1+np.sqrt(episode)/10)
                non_zero_actions += 0 if action_index == 0 else 1

                self.memory(state, action_index, reward if not done else -100*(1+np.sqrt(episode)/10))

                state = copy.deepcopy(next_state) if not done else np.zeros([1, self.state_size])

                time_step += 1
                max_action = max(max_action, action_index)

                if done or time_step > time_step_end:
                    if done:
                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    if time_step > time_step_end:
                        print("End Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    # global scores
                    scores.append(score)
                    alive_step.append(time_step)
                    # print(len(scores))
                    # global episode
                    episode += 1
                    # if len(self.states) == 0:
                    #     k = 1
                    self.train_episode(True)  # max score = 80000
                    break
                else:
                    if time_step % 10 ==0:
                        print("Continue Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with recent time:", time_step, "/ with recent action", action_index,"/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ max_action so far:", max_action)
                        self.train_episode(False)

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            with self.session.as_default():
                with self.session.graph.as_default():
                    running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state[0])
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)
        with self.session.as_default():
            with self.session.graph.as_default():
                values = self.critic.predict(np.array(self.states))[0]
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        #mean_reward = np.mean(scores)
        if len(scores) >= 50:
            mean_reward_50 = np.mean(scores[-50:])
        else:
            mean_reward_50 = np.mean(scores)

        if len(alive_step) >= 50:
            mean_alive_step_50 = np.mean(alive_step[-50:])
        else:
            mean_alive_step_50 = np.mean(alive_step)

        # 创建摘要对象
        mean_reward_summary = tf.compat.v1.Summary()
        # 添加平均奖励到摘要
        #mean_reward_summary.value.add(tag="mean_reward", simple_value=mean_reward)
        mean_reward_summary.value.add(tag="mean_reward_50", simple_value=mean_reward_50)
        mean_reward_summary.value.add(tag="mean_alive_step_50", simple_value=mean_alive_step_50)

        # 将摘要添加到摘要写入器
        self.tf_writer.add_summary(mean_reward_summary, episode)

        with self.session.as_default():
            with self.session.graph.as_default():
                state_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.states))
                action_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.actions))
                advantage_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(advantages))
                # reshaped_state_as_tensor = tf.compat.v1.reshape(state_as_tensor,self.actor.input.shape.dims)

                self.optimizer[0]([state_as_tensor, action_as_tensor, advantage_as_tensor])
                self.optimizer[1]([state_as_tensor, discounted_rewards])
                self.states, self.actions, self.rewards = [], [], []



    def get_action(self, state):
        with self.session.as_default():
            with self.session.graph.as_default():
            # 使用神经网络预测动作概率
                policy = self.actor.predict(state, batch_size=1).flatten()
        # 以概率policy随机选择动作
        action_index = np.random.choice(self.action_size, p=policy)
        return action_index

    def convert_obs(self, observation):
        # Made a custom version to normalize per attribute
        #return observation.to_vect()
        li_vect=  []
        for el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(float)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)

class convert(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__):

        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)

        # Store constructor params
        self.observation_space = observation_space
        self.name = name
    def convert_act(self, action):
        return super().convert_act(action)

    def my_act(self, state, reward, done=False):
        # state = state.to_vect()  # 直接使用原始状态向量
        state = np.array(state)
        state = state.reshape([1, state.size])
        action = self.get_action(state)
        return action
