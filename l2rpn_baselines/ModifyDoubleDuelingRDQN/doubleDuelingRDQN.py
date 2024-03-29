# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import copy
import numpy as np
import random
try:
    import tensorflow as tf
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False


from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from l2rpn_baselines.ModifyDoubleDuelingRDQN.doubleDuelingRDQNConfig import DoubleDuelingRDQNConfig as cfg
from l2rpn_baselines.ModifyDoubleDuelingRDQN.experienceBuffer import ExperienceBuffer
from l2rpn_baselines.ModifyDoubleDuelingRDQN.doubleDuelingRDQN_NN import DoubleDuelingRDQN_NN

class DoubleDuelingRDQN(AgentWithConverter):
    """
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 is_training=False):
                
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        
        # Call parent constructor
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)

        # Store constructor params
        self.observation_space = observation_space
        self.name = name
        self.trace_length = cfg.TRACE_LENGTH
        self.batch_size = cfg.BATCH_SIZE
        self.is_training = is_training
        self.lr = cfg.LR
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.mem_state = None
        self.carry_state = None
        self.prev_action = None
        self.prev_reward = None

        # Declare training vars
        self.exp_buffer = None
        self.done = False
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None

        # Compute dimensions from intial state
        self.observation_size = self.observation_space.size_obs()
        self.action_size = self.action_space.size()

        # Load network graph
        self.Qmain = DoubleDuelingRDQN_NN(self.action_size,
                                          self.observation_size,
                                          learning_rate = self.lr)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

        # 初始化评估时当前状态的上一步动作值
        self.prev_action_evaluate = 0



    def _init_training(self):
        self.exp_buffer = ExperienceBuffer(cfg.REPLAY_BUFFER_SIZE,
                                           self.batch_size,
                                           self.trace_length)
        self.done = True
        self.epoch_rewards = []
        self.epoch_alive = []
        self.Qtarget = DoubleDuelingRDQN_NN(self.action_size,
                                            self.observation_size,
                                            learning_rate = self.lr)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False
        self.mem_state = np.zeros(self.Qmain.h_size)
        self.carry_state = np.zeros(self.Qmain.h_size)
        # 添加初始化当前状态的上一步动作和奖励值
        self.prev_action = 0
        self.prev_reward = 0.0

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

    def _save_hyperparameters(self, logpath, env, steps):
        try:
            # change of name in grid2op >= 1.2.3
            r_instance = env._reward_helper.template_reward
        except AttributeError as nm_exc_:
            r_instance = env.reward_helper.template_reward
        hp = {
            "lr": cfg.LR,
            "batch_size": cfg.BATCH_SIZE,
            "trace_len": cfg.TRACE_LENGTH,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "buffer_size": cfg.REPLAY_BUFFER_SIZE,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
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

    def convert_act(self, action):
        return super().convert_act(action)

    def reset(self, observation):
        self._reset_state(observation)

    def my_act(self, state, reward,  done=False):
        #state = np.concatenate([state, [self.prev_action_evaluate, reward]])
        data_input = np.array(state)
        data_input.reshape(1, 1, self.observation_size)
        a, _, m, c = self.Qmain.predict_move(data_input,
                                             self.mem_state,
                                             self.carry_state,
                                             self.prev_action_evaluate,
                                             reward)
        self.mem_state = m
        self.carry_state = c
        self.prev_action_evaluate = a

        return a
    
    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    def random_task(env, N_task):
        tasks = []
        # Loop for N_task times
        for _ in range(N_task):
            # Randomly select an environment
            mix_names = list(env.keys())
            random_env_name = random.choice(mix_names)
            random_env = env[random_env_name]

            # Get all chronic names
            all_chronics_paths = random_env.chronics_handler.subpaths
            all_chronics_names = [path.split("\\")[-1] for path in all_chronics_paths]

            # Randomly select a chronic
            random_chronic_name = random.choice(all_chronics_names)

            # Set the environment to the selected chronic
            random_env.set_id(random_chronic_name)

            # Reset the environment
            random_env.reset()

            tasks.append(random_env)
            return tasks

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps = 0,
              logdir = "logs"):

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        epsilon = cfg.INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        episode = 0
        episode_exp = []

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".tf")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)
        
        # Training loop
        print(self.action_size)
        self._reset_state(env.current_obs)
        while step < num_steps:
            # New episode
            if self.done:
                new_obs = env.reset() # This shouldn't raise
                self._reset_state(new_obs)
                self.prev_action = 0 #初始化
                self.prev_reward = 0
                # Push current episode experience to experience buffer
                self._register_experience(episode_exp, episode)
                # Reset current episode experience
                episode += 1
                episode_exp = []

            if cfg.VERBOSE and step % 1000 == 0:
                print("Step [{}] -- Dropout [{}]".format(step, epsilon))

            # Choose an action
            if step <= num_pre_training_steps:
                a, m, c = self.Qmain.random_move(self.state,
                                                 self.mem_state,
                                                 self.carry_state,
                                                 self.prev_action,
                                                 self.prev_reward)
            elif len(episode_exp) < self.trace_length:
                a, m, c = self.Qmain.random_move(self.state,
                                                 self.mem_state,
                                                 self.carry_state,
                                                 self.prev_action,
                                                 self.prev_reward)
                a = 0 # Do Nothing
            else:
                a, _, m, c = self.Qmain.bayesian_move(self.state,
                                                      self.mem_state,
                                                      self.carry_state,
                                                      self.prev_action,
                                                      self.prev_reward,
                                                      epsilon)

            # Update LSTM state
            self.mem_state = m
            self.carry_state = c

            # Convert it to a valid action
            act = self.convert_act(a)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            new_state = self.convert_obs(new_obs)


            # Save to current episode experience
            episode_exp.append((self.state, a, reward, self.done, new_state, self.prev_action, self.prev_reward))

            # 更新上一步的a,r
            self.prev_action = a;
            self.prev_reward = reward;

            # Train when pre-training is over
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Slowly decay dropout rate
                if epsilon > cfg.FINAL_EPSILON:
                    epsilon -= cfg.STEP_EPSILON
                if epsilon < cfg.FINAL_EPSILON:
                    epsilon = cfg.FINAL_EPSILON

                # Perform training at given frequency
                if step % cfg.UPDATE_FREQ == 0 and \
                   self.exp_buffer.can_sample():
                    # Sample from experience buffer
                    batch = self.exp_buffer.sample()
                    # Perform training
                    self._batch_train(batch, step, training_step)
                    # Update target network towards primary network
                    if cfg.UPDATE_TARGET_SOFT_TAU > 0:
                        tau = cfg.UPDATE_TARGET_SOFT_TAU
                        self.Qmain.update_target_soft(self.Qtarget.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings,
                # update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            print(total_reward)
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                if cfg.VERBOSE:
                    print("Survived [{}] steps".format(alive_steps))
                    print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.save(modelpath)

            # Iterate to next loop
            step += 1
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)

    def _batch_train(self, batch, step, training_step):
        """Trains network to fit given parameters"""
        Q = np.zeros((self.batch_size, self.action_size))
        batch_mem = np.zeros((self.batch_size, self.Qmain.h_size))
        batch_carry = np.zeros((self.batch_size, self.Qmain.h_size))

        input_size = self.observation_size+self.action_size+1
        m_data = np.vstack(batch[:, 0])
        #添加当前时刻的上一步动作和上一步奖励
        m_data_prev_a = batch[:, 5].astype(int)  # 第六列数据
        m_data_prev_a_one_hot = np.eye(self.action_size)[m_data_prev_a]
        m_data_prev_r = batch[:, 6].astype(float)  # 第七列数据
        #m_data = np.hstack((m_data, m_data_prev_a_one_hot, m_data_prev_r[:, np.newaxis]))
        m_data = np.concatenate((m_data, m_data_prev_a_one_hot, m_data_prev_r[:, np.newaxis]), axis=1)
        m_data = m_data.reshape(self.batch_size, self.trace_length, input_size)

        t_data = np.vstack(batch[:, 4])
        # 添加下一状态的上一步动作和上一步奖励
        t_data_prev_a = batch[:, 1].astype(int)  # 第二列数据
        t_data_prev_a_one_hot = np.eye(self.action_size)[t_data_prev_a]
        t_data_prev_r = batch[:, 2].astype(float)  # 第三列数据
        #t_data = np.hstack((t_data, t_data_prev_a[:, np.newaxis], t_data_prev_r[:, np.newaxis]))
        t_data = np.concatenate((t_data, t_data_prev_a_one_hot, t_data_prev_r[:, np.newaxis]), axis=1)
        t_data = t_data.reshape(self.batch_size, self.trace_length, input_size)
        q_input = [
            copy.deepcopy(batch_mem),
            copy.deepcopy(batch_carry),
            copy.deepcopy(m_data)
        ]
        q1_input = [
            copy.deepcopy(batch_mem),
            copy.deepcopy(batch_carry),
            copy.deepcopy(t_data)
        ]
        q2_input = [
            copy.deepcopy(batch_mem),
            copy.deepcopy(batch_carry),
            copy.deepcopy(t_data)
        ]

        # Batch predict
        self.Qmain.trace_length.assign(self.trace_length)
        self.Qmain.dropout_rate.assign(0.0)
        self.Qtarget.trace_length.assign(self.trace_length)
        self.Qtarget.dropout_rate.assign(0.0)

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q, _, _ = self.Qmain.model.predict(q_input,
                                           batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1, _, _ = self.Qmain.model.predict(q1_input,
                                            batch_size=self.batch_size)
        Q2, _, _ = self.Qtarget.model.predict(q2_input,
                                              batch_size=self.batch_size)

        # Compute batch Double Q update to Qtarget
        for i in range(self.batch_size):
            idx = i * (self.trace_length - 1)
            doubleQ = Q2[i, np.argmax(Q1[i])]
            a = batch[idx][1]
            r = batch[idx][2]
            d = batch[idx][3]
            Q[i, a] = r
            if d == False:
                Q[i, a] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        batch_x = [batch_mem, batch_carry, m_data]
        batch_y = [Q, batch_mem, batch_carry]
        loss = self.Qmain.model.train_on_batch(batch_x, batch_y)
        loss = loss[0]

        # Log some useful metrics
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            if cfg.VERBOSE:
                print("loss =", loss)

            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)

