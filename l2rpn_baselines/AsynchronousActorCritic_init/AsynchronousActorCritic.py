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

    from tensorflow.keras.layers import Dense, Input, LSTM, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.python.keras.backend as K
    from l2rpn_baselines.AsynchronousActorCritic_init.user_environment_make import set_environement
    from l2rpn_baselines.AsynchronousActorCritic_init.experienceBuffer import ExperienceBuffer
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
        self.h_size = 512
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
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, dtype=tf.int32, name="trace_length")  # 序列长度。
        self.dropout_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False,
                                         name="dropout_rate")  # 用于控制输入的 dropout 比率（69行用到）。Dropout 是一种正则化技术，有助于减少过拟合。
        input_h_state = Input(dtype=tf.float32, shape=(self.h_size), name='input_h_state')
        # 这行代码创建了一个名为input_h_state的输入层，用于表示LSTM的初始隐藏状态。
        input_c_state = Input(dtype=tf.float32, shape=(self.h_size), name='input_c_state')
        # 这行代码创建了一个名为 input_c_state 的输入层，用于表示 LSTM 的初始细胞状态。
        input_layer = Input(dtype=tf.float32, shape=(None, self.state_size + self.action_size + 1),
                                name='input_obs')
        # 这行代码创建了一个名为 input_layer 的输入层，用于表示观测序列,这个输入层可以接受任何形状的张量作为输入，
        # 只要张量的最后一个维度与输入层的形状匹配即可。因此，不管输入的张量是一维、二维、三维或更高维度，都可以作为输入传递给模型。
        # dtype=tf.float32 指定了输入数据的数据类型为 32 位浮点数。
        # self.observation_size 表示每个时间步观测的特征维度。name='input_obs' 指定了该输入层的名称为 'input_obs'。
        # self.observation_size 是指每个时间步观测的特征维度，这意味着在每个时间步，有一个包含455个特征的观测值。

        # Get shapes from input_layer，这段代码逐行获取了输入层 input_layer 的形状信息，分别表示批次大小、序列长度和特征维度。
        batch_size = tf.shape(input_layer)[0]
        trace_len = tf.shape(input_layer)[1]
        data_size = tf.shape(input_layer)[-1]

        # Reshape for dense processing
        input_format = tf.reshape(input_layer, (-1, input_layer.shape[-1]), name="dense_reshape")

        # Bayesian NN simulate
        lay1 = Dropout(tf.convert_to_tensor(self.dropout_rate), name="bnn_dropout")(input_format)
        # Forward pass
        lay1 = Dense(512, name="fc_1")(lay1)
        lay1 = tf.nn.leaky_relu(lay1, alpha=0.01, name="leak_fc_1")
        lay2 = Dense(256, name="fc_2")(lay1)
        lay2 = tf.nn.leaky_relu(lay2, alpha=0.01, name="leak_fc_2")
        lay3 = Dense(128, name="fc_3")(lay2)
        lay3 = tf.nn.leaky_relu(lay3, alpha=0.01, name="leak_fc_3")
        lay4 = Dense(self.h_size, name="fc_4")(lay3)

        # Reshape to (batch_size, trace_len, data_size) for rnn
        rnn_format = tf.reshape(lay4, (batch_size, trace_len, self.h_size), name="rnn_reshape")
        # Recurring part
        lstm_layer = LSTM(self.h_size, return_state=True, name="lstm")
        # lstm_layer 使用 tfkl.LSTM 创建，其中 self.h_size 表示 LSTM 层神经元的大小。return_state 参数决定是否返回最后一个时间步的隐藏状态和细胞状态
        lstm_state = [input_h_state, input_c_state]
        # lstm_state 被定义为 [input_mem_state, input_carry_state]，它们是 LSTM 的隐藏和细胞状态的初始值。
        lstm_output, h_s, c_s = lstm_layer(rnn_format, initial_state=lstm_state)
        # 通过使用 rnn_format（重塑后的输入）和初始状态调用 lstm_layer，得到 lstm_output、h_s 和 c_s。
        # lstm_output 是每个样本在最后一个时间步的输出，而 h_s 和 c_s 是处理输入序列后更新的隐藏和细胞状态。

        # state = Input(batch_shape=(None,  self.state_size))
        # shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(lstm_output)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)
        # action_prob = K.softmax(action_intermediate)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(lstm_output)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        model_inputs = [input_h_state, input_c_state, input_layer]
        action_model_outputs = [action_prob, h_s, c_s]
        state_model_outputs = [state_value, h_s, c_s]


        actor = Model( inputs= model_inputs, outputs=action_model_outputs)
        critic = Model(inputs=model_inputs, outputs=state_model_outputs)

        # actor._make_predict_function()
        # critic._make_predict_function()

        return actor, critic

    # def act(self, state, h, c, prev_a, prev_r):
    #     self.trace_length.assign(1)
    #     self.dropout_rate.assign(0.0)
    #
    #     state = np.concatenate([state, prev_a, [prev_r]])
    #     state_input = state.reshape(1, 1, -1)
    #     h_input = h.reshape(1, -1)
    #     c_input = c.reshape(1, -1)
    #     model_input = [h_input, c_input, state_input]
    #
    #     # #state = state.to_vect()  # 直接使用原始状态向量
    #     # state = np.array(state)
    #     # state = state.reshape([1, state.size])
    #     # 使用神经网络预测动作概率
    #     policy = self.actor.predict(model_input, batch_size=1).flatten()
    #
    #     # 随机选择动作
    #     action_index = np.random.choice(self.action_size, p=policy)
    #     return action_index, h, c

    # def get_action(self, state):
    #     # 使用神经网络预测动作概率
    #     policy = self.actor.predict(state, batch_size=1).flatten()
    #
    #     # 随机选择动作
    #     action_index = np.random.choice(self.action_size, p=policy)
    #     return action_index

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None,))

        policy, h, c = self.actor.output

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

        value, h, c = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        gradients = tf.gradients(loss, self.critic.trainable_weights)
        grads_and_vars = zip(gradients, self.critic.trainable_weights)
        optimizer.apply_gradients(grads_and_vars)
        train = K.function([self.critic.input, discounted_reward], outputs=[K.constant([0,1])], updates=[])
        return train

    # make agents(local) and start training
    def train(self, nn_weights_name):
        agents = [Agent(i, self.actor, self.critic, self.trace_length, self.dropout_rate, self.optimizer, self.env_name, self.discount_factor,
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
    def __init__(self, index, actor, critic, trace_length, dropout_rate, optimizer, env_name, discount_factor, action_space, state_size, profiles_chronics, session):
        threading.Thread.__init__(self)

        # self.states = []
        # self.rewards = []
        # self.actions = []
        # self.pre_rewards = []
        # self.pre_actions = []
        self.exp_buffer = None
        self.batch_size = 32

        self.trace_length = trace_length
        self.dropout_rate = dropout_rate

        self.h_state = None
        self.c_state = None
        self.prev_action = None
        self.prev_reward = None

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
            #state = state.reshape([1, state.size])
            score = 0
            time_step = 0
            max_action = 0
            non_zero_actions = 0
            epsilon = 0.5
            self.h_state = np.zeros(512)
            self.c_state = np.zeros(512)
            self.prev_action = 0
            self.prev_reward = 0.0
            episode_exp = []
            while True:
                # Decaying epsilon greedy. Not the best one. This agent needs a better exploration strategy to help
                # it learn to perform well.
                #if np.random.random() < epsilon*(1/(episode/400+1)):
                if epsilon <= 1024:
                    action_index, h, c = self.random_move(state, self.h_state, self.c_state, self.prev_action, self.prev_reward)
                    #epison_flag = True
                # else:
                #     epison_flag = False
                #     if time_step%1 == 0:# or max(state_as_dict.rho)>0.75:
                #         action_index, h, c = self.act(state, self.h_state, self.c_state, self.prev_action, self.prev_reward)
                #     else:
                #         action_index = 0
                #         h = self.h_state
                #         c = self.c_state
                elif len(episode_exp) < self.trace_length:
                    action_index, h, c = self.random_move(state, self.h_state, self.c_state, self.prev_action,
                                                          self.prev_reward)
                    action_index = 0
                else:
                    action_index, h, c = self.act(state, self.h_state, self.c_state, self.prev_action, self.prev_reward)

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
                #score += reward if not done else -100*(1+np.sqrt(episode)/10)
                score += reward
                non_zero_actions += 0 if action_index == 0 else 1

                #self.memory(state, action_index, reward if not done else -100*(1+np.sqrt(episode)/10))
                # Save to current episode experience
                episode_exp.append((state, action_index, reward, done, next_state, self.prev_action, self.prev_reward))

                state = copy.deepcopy(next_state) if not done else np.zeros([1, self.state_size])
                self.h_state = copy.deepcopy(h)
                self.c_state = copy.deepcopy(c)
                self.prev_action = action_index
                self.prev_reward = reward


                time_step += 1
                max_action = max(max_action, action_index)

                if done or time_step > time_step_end:
                    # global episode
                    episode += 1
                    if done:
                        # global episode
                        # print("----STOPPED Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                        #       "/ with final time:", time_step, "/ with final action", action_index,
                        #       "/Random action: ",epison_flag,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode, "/ instant reward",
                              int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/ number of non-zero actions", non_zero_actions,
                              "/ day_hour_min:", time_hour)
                        self._register_experience(episode_exp, episode)
                    if time_step > time_step_end:
                        print("End Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with final time:", time_step, "/ with final action", action_index,
                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)
                    # global scores
                    scores.append(score)
                    alive_step.append(time_step)
                    # print(len(scores))

                    # if len(self.states) == 0:
                    #     k = 1
                    if self.exp_buffer.can_sample():
                        # Sample from experience buffer
                        batch = self.exp_buffer.sample()
                        # Perform training
                        #self._batch_train(batch, step, training_step)
                        self.train_episode(batch, True)  # max score = 80000
                    episode_exp = []
                    break
                else:
                    if time_step % 10 ==0 and self.exp_buffer.can_sample():
                        print("Continue Thread:", self.index, "/ train episode: ", episode,  "/ instant reward",int(reward), "/ score : ", int(score),
                              "/ with recent time:", time_step, "/ with recent action", action_index,"/ number of non-zero actions", non_zero_actions, "/ max_action so far:", max_action)
                        # Sample from experience buffer
                        batch = self.exp_buffer.sample()
                        self.train_episode(batch, False)

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, states, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            with self.session.as_default():
                with self.session.graph.as_default():
                    running_add = self.critic.predict(states)[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    # def memory(self, state, action, reward):
    #     self.states.append(state[0])
    #     act = np.zeros(self.action_size)
    #     act[action] = 1
    #     self.actions.append(act)
    #     self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, batch, done):
        policy = np.zeros((self.batch_size, self.action_size))
        value = np.zeros((self.batch_size, 1))
        batch_h = np.zeros((self.batch_size, self.actor.h_size))
        batch_c = np.zeros((self.batch_size, self.actor.h_size))

        input_size = self.state_size + self.action_size + 1
        m_data = np.vstack(batch[:, 0])
        # 添加当前时刻的上一步动作和上一步奖励
        m_data_prev_a = batch[:, 5].astype(int)  # 第六列数据
        m_data_prev_a_one_hot = np.eye(self.action_size)[m_data_prev_a]
        m_data_prev_r = batch[:, 6].astype(float)  # 第七列数据
        # m_data = np.hstack((m_data, m_data_prev_a_one_hot, m_data_prev_r[:, np.newaxis]))
        m_data = np.concatenate((m_data, m_data_prev_a_one_hot, m_data_prev_r[:, np.newaxis]), axis=1)
        m_data = m_data.reshape(self.batch_size, self.trace_length, input_size)

        m_data_a = batch[:, 1].astype(int)  # 第六列数据
        m_data_a_one_hot = np.eye(self.action_size)[m_data_a]

        input = [
            copy.deepcopy(batch_h),
            copy.deepcopy(batch_c),
            copy.deepcopy(m_data)
        ]

        # 拿到状态的最后一个值
        input_last = [
            copy.deepcopy(batch_h),
            copy.deepcopy(batch_c),
            copy.deepcopy(m_data(-1))
        ]

        # t_data = np.vstack(batch[:, 4])
        # # 添加下一状态的上一步动作和上一步奖励
        # t_data_prev_a = batch[:, 1].astype(int)  # 第二列数据
        # t_data_prev_a_one_hot = np.eye(self.action_size)[t_data_prev_a]
        # t_data_prev_r = batch[:, 2].astype(float)  # 第三列数据
        # # t_data = np.hstack((t_data, t_data_prev_a[:, np.newaxis], t_data_prev_r[:, np.newaxis]))
        # t_data = np.concatenate((t_data, t_data_prev_a_one_hot, t_data_prev_r[:, np.newaxis]), axis=1)
        # t_data = t_data.reshape(self.batch_size, self.trace_length, input_size)

        # Batch predict
        self.actor.trace_length.assign(self.trace_length)
        self.actor.dropout_rate.assign(0.0)
        self.critic.trace_length.assign(self.trace_length)
        self.critic.dropout_rate.assign(0.0)

        discounted_rewards = self.discount_rewards(input_last, batch[:, 2], done)
        with self.session.as_default():
            with self.session.graph.as_default():
                value = self.critic.predict(input)[0]
        value = np.reshape(value, len(value))

        advantages = discounted_rewards - value

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
                # state_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.states))
                # action_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(self.actions))
                # advantage_as_tensor = tf.compat.v1.convert_to_tensor(np.asarray(advantages))
                # reshaped_state_as_tensor = tf.compat.v1.reshape(state_as_tensor,self.actor.input.shape.dims)
                input_tensor = tf.convert_to_tensor(input)  # 将 input 转换为 TensorFlow 张量
                m_data_a_one_hot_tensor = tf.convert_to_tensor(m_data_a_one_hot)  # 将 m_data_a_one_hot 转换为 TensorFlow 张量
                advantages_tensor = tf.convert_to_tensor(advantages)  # 将 advantages 转换为 TensorFlow 张量

                # 使用转换后的张量调用优化器
                self.optimizer[0]([input_tensor, m_data_a_one_hot_tensor, advantages_tensor])
                # 对于第二个优化器，discounted_rewards 是一个 NumPy 数组，可以直接传递它，不需要转换为 TensorFlow 张量
                self.optimizer[1]([input_tensor, discounted_rewards])


    def act(self, state, h, c, prev_a, prev_r):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        # 将 prev_a 转换为 one-hot 向量
        prev_a_one_hot = np.eye(self.action_size)[prev_a]
        state = np.concatenate([state, prev_a_one_hot, [prev_r]])
        state_input = state.reshape(1, 1, -1)
        h_input = h.reshape(1, -1)
        c_input = c.reshape(1, -1)
        model_input = [h_input, c_input, state_input]

        # #state = state.to_vect()  # 直接使用原始状态向量
        # state = np.array(state)
        # state = state.reshape([1, state.size])
        # 使用神经网络预测动作概率
        policy, h, c = self.actor.predict(model_input, batch_size=1).flatten()

        # 随机选择动作
        action_index = np.random.choice(self.action_size, p=policy)
        return action_index, h, c

    def random_move(self, state, h, c, prev_a, prev_r):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        # 将 prev_a 转换为 one-hot 向量
        prev_a_one_hot = np.eye(self.action_size)[prev_a]
        state = np.concatenate([state, prev_a_one_hot, [prev_r]])
        state_input = state.reshape(1, 1, -1)
        h_input = h.reshape(1, -1)
        c_input = c.reshape(1, -1)
        model_input = [h_input, c_input, state_input]

        _, h, c = self.actor.predict(model_input, batch_size = 1)
        move = np.random.randint(0, self.action_size)

        return move, h, c

    # def get_action(self, state):
    #     with self.session.as_default():
    #         with self.session.graph.as_default():
    #         # 使用神经网络预测动作概率
    #             policy = self.actor.predict(state, batch_size=1).flatten()
    #     # 以概率policy随机选择动作
    #     action_index = np.random.choice(self.action_size, p=policy)
    #     return action_index

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

    def _register_experience(self, episode_exp, episode):
        missing_obs = self.trace_length - len(episode_exp)

        if missing_obs > 0: # We are missing exp to make a trace
            exp = episode_exp[0] # Use inital state to fill out
            for missing in range(missing_obs):
                # Use do_nothing action at index 0
                #self.exp_buffer.add(exp[0], 0, exp[2], exp[3], exp[4], episode)
                self.exp_buffer.add(exp[0], 0, exp[2], exp[3], exp[4], exp[5], exp[6], episode)

        # Register the actual experience
        for exp in episode_exp:
            #self.exp_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4], episode)
            self.exp_buffer.add(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], exp[6], episode)

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
