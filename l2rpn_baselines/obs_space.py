
import numpy as np
import copy
import os
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward
import sys
import grid2op

# def useful_state(obs,value_multiplier):
#     selected_obs = np.hstack((obs.topo_vect,obs.line_status))
#     selected_obs = np.hstack((selected_obs,obs.load_p/100))#
#     selected_obs = np.hstack((selected_obs,obs.load_q/100))
#     selected_obs = np.hstack((selected_obs,obs.prod_p/100))
#     selected_obs = np.hstack((selected_obs,obs.prod_v/value_multiplier))
#     selected_obs = np.hstack((selected_obs,obs.rho))
#     # selected_obs = np.hstack((selected_obs,obs.day))
#     selected_obs = np.hstack((selected_obs,obs.hour_of_day/24))
#     selected_obs = np.hstack((selected_obs,obs.minute_of_hour/60))
#     # selected_obs = np.hstack((selected_obs,obs.day_of_week/7))
#     return selected_obs
#
#
#
# env = grid2op.make("l2rpn_neurips_2020_track2_small")
#
# # Define the size of state space and action space.
# do_nothing_act = env._helper_action_player({})
#
# obs, reward, done, info = env.step(do_nothing_act)
# #conversion parameter
# value_multiplier = env.backend.prod_pu_to_kv
# state_trimmed = useful_state(obs,value_multiplier)
# state_trimmed = state_trimmed.reshape([1,state_trimmed.size])
# state_size = state_trimmed.size



# def useful_state(obs, value_multiplier,value_multiplier_load):
#     # 对已存在的属性使用原有的归一化方法
#     selected_obs = np.hstack((obs.topo_vect, obs.line_status,
#                               obs.load_p / 100, obs.load_q / 100, obs.load_v/value_multiplier_load,
#                               obs.prod_p / 100,  obs.prod_q / 100, obs.prod_v / value_multiplier,
#                               obs.rho, obs.month/12,
#                               obs.hour_of_day / 24, obs.minute_of_hour / 60))
#
#     # 对新增的属性使用新的归一化方法
#     # 首先获取新增属性的值
#     obs_add = ['load_theta', 'gen_theta', 'timestep_overflow', 'p_or', 'q_or', 'v_or', 'a_or', 'theta_or']
#     new_attributes = convert_obs(obs_add,obs)
#
#     # 将归一化后的新属性添加到状态向量中
#     selected_obs = np.hstack((selected_obs, new_attributes))
#
#     return selected_obs

obs_add = ['topo_vect','line_status','load_p','load_q','load_v','prod_p','prod_q','prod_v','rho','load_theta',
           'gen_theta', 'timestep_overflow', 'p_or', 'q_or', 'v_or', 'a_or', 'theta_or','month','day', 'hour_of_day', 'minute_of_hour', 'day_of_week']


def convert_obs(observation,obs):
    # Made a custom version to normalize per attribute
    # return observation.to_vect()
    li_vect = []
    for el in observation:
        v = obs._get_array_from_attr_name(el).astype(float)
        v_fix = np.nan_to_num(v)
        v_norm = np.linalg.norm(v_fix)
        if v_norm > 1e6:
            v_res = (v_fix / v_norm) * 10.0
        else:
            v_res = v_fix
        li_vect.append(v_res)
    return np.concatenate(li_vect)


env = grid2op.make("l2rpn_neurips_2020_track2_small")

# Define the size of state space and action space.
do_nothing_act = env._helper_action_player({})

obs, reward, done, info = env.step(do_nothing_act)
# print(obs.attr_list_vect)
# Conversion parameter
# value_multiplier = env.backend.prod_pu_to_kv
# value_multiplier_load = env.backend.load_pu_to_kv
state_trimmed = convert_obs(obs_add,obs)
state_trimmed = state_trimmed.reshape([1, state_trimmed.size])
state_size = state_trimmed.size

print(state_trimmed)
print(state_size)


