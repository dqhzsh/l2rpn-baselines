try:
    import grid2op
    import matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    import os
    from grid2op.MakeEnv import make
    import itertools
    import os.path

    import pickle
except ImportError as exc_:
    raise ImportError("AsynchronousActorCritic baseline impossible to load the required dependencies. The error was: \n {}".format(exc_))


def prune_action_space(bus_no):
    # Initialize the env."case5_example",chronics_path=os.path.join("public_data", "chronics_5bus_example")
    if bus_no == "14":
        environment = make('l2rpn_case14_sandbox')
        num_substations = 14 # 14 for 14 bus
    elif bus_no == "5":
        environment = make("case5_example",chronics_path=os.path.join("public_data", "chronics_5bus_example"))
        num_substations = 5
    else:
        print("Unexpected bus size of the power system. Exiting code in Action_reduced_list.py")
        exit()

    from pandapower.plotting.plotly import simple_plotly
    # simple_plotly(environment.backend._grid)
    # Load ids
    print("\nInjection information:")
    load_to_subid = environment.action_space.load_to_subid
    print ('There are {} loads connected to substations with id: {}'.format(len(load_to_subid), load_to_subid))  #对于每个负载，给出它所连接的变电站的 ID: [ 1  2  3  4  5  8  9 10 11 12 13]

    # Generators irds
    gen_to_subid = environment.action_space.gen_to_subid
    print ('There are {} generators, connected to substations with id: {}'.format(len(gen_to_subid), gen_to_subid))  #There are 6 generators, connected to substations with id: [1 2 5 5 7 0]

    # Line id sender
    print("\nPowerline information:")
    lines_or_to_subid = environment.action_space.line_or_to_subid
    #lines_or_to_subid = environment.action_space.lines_or_to_subid
    lines_ex_to_subid = environment.action_space.line_ex_to_subid

    print ('There are {} transmissions lines on this grid.'.format(len(lines_or_to_subid)))  #Powerline information:There are 20 transmissions lines on this grid.


    gen_at_sub = []  #对于每个变电站ID来说，连接到变电站ID的发电机ID
    load_at_sub = []   #对于每个变电站ID来说，连接到变电站ID的负载ID
    lines_or_at_sub = []   #对于每个变电站ID来说，起点连接到变电站ID的电线ID
    lines_ex_at_sub = []   #对于每个变电站ID来说，终点连接到变电站ID的电线ID
    for k in range(num_substations):
        gen_at_sub.append(np.arange(len(gen_to_subid))[gen_to_subid==k])
        load_at_sub.append(np.arange(len(load_to_subid))[load_to_subid==k])
        lines_or_at_sub.append(np.arange(len(lines_or_to_subid))[lines_or_to_subid==k])
        lines_ex_at_sub.append(np.arange(len(lines_ex_to_subid))[lines_ex_to_subid==k])
    #1;
    # Num of elements per SE
    print("\nSubstations information:")
    for i, nb_el in enumerate(environment.action_space.sub_info):
        print("On susbtation {} there are {} elements.".format(i, nb_el))  #电网的第i个变电站有 nb_el 个元件与其连接
    # Substations information:
    # On susbtation 0 there are 3 elements.
    # On susbtation 1 there are 6 elements.
    # On susbtation 2 there are 4 elements.
    # On susbtation 3 there are 6 elements.
    # On susbtation 4 there are 5 elements.
    # On susbtation 5 there are 7 elements.
    # On susbtation 6 there are 3 elements.
    # On susbtation 7 there are 2 elements.
    # On susbtation 8 there are 5 elements.
    # On susbtation 9 there are 3 elements.
    # On susbtation 10 there are 3 elements.
    # On susbtation 11 there are 3 elements.
    # On susbtation 12 there are 4 elements.
    # On susbtation 13 there are 3 elements.

    # adding the change_actions at for 1 substation at a time . Initializing with no action
    gen_action_list =[[]] # list of gens acted
    load_action_list = [[]]
    line_or_action_list = [[]]
    line_ex_action_list = [[]]
    substation_acted = [[]] # substation acted
    gen_action_list_bus_2 = [[]]
    load_action_list_bus_2 = [[]]
    line_or_action_list_bus_2 = [[]]
    line_ex_action_list_bus_2 = [[]]
    do_nothing_act = environment._helper_action_player({})
    obs, reward, done, info = environment.step(do_nothing_act)
    for sub_id in range(num_substations):
        num_gen_at_sub = len(gen_at_sub[sub_id] )  #连接到sub_id变电站的的发电机数量
        num_load_at_sub = len(load_at_sub[sub_id] )  #连接到sub_id变电站的的负载数量
        num_line_or_at_sub = len(lines_or_at_sub[sub_id] )  #起点连接到sub_id变电站的的电线数量
        num_line_ex_at_sub = len(lines_ex_at_sub[sub_id] )  #终点连接到sub_id变电站的的电线数量
        switching_patterns = ["".join(seq) for seq in itertools.product("01",repeat=environment.action_space.sub_info[sub_id] - 1)]  # reduce by 1 bit due to compliment being the same
        switching_patterns = [[int(sw_i_k) for sw_i_k in '0' + sw_i] for sw_i in switching_patterns]  # adding back the '0' at the beginning as we are fizing this bit
        switching_patterns.pop(0) # deleting first action as it is a no-action
        for sw_action in switching_patterns:
            switching_patterns_split = np.split(np.array(sw_action), np.cumsum([num_gen_at_sub,num_load_at_sub,num_line_or_at_sub,num_line_ex_at_sub]))
            gen_action_list.append(gen_at_sub[sub_id][switching_patterns_split[0] == 1])
            load_action_list.append(load_at_sub[sub_id][switching_patterns_split[1] == 1])
            line_or_action_list.append(lines_or_at_sub[sub_id][switching_patterns_split[2] == 1])
            line_ex_action_list.append(lines_ex_at_sub[sub_id][switching_patterns_split[3] == 1])
            substation_acted.append(sub_id)


            # gen_action_list_bus_2.append(gen_at_sub[sub_id][switching_patterns_split[0] == 0])
            # load_action_list_bus_2.append(load_at_sub[sub_id][switching_patterns_split[1] == 0])
            # line_or_action_list_bus_2.append(lines_or_at_sub[sub_id][switching_patterns_split[2] == 0])
            # line_ex_action_list_bus_2.append(lines_ex_at_sub[sub_id][switching_patterns_split[3] == 0])
            # do_act = environment.action_space(
            #     {"set_bus": {"generators_id": [(g,1) for g in gen_action_list[-1]]+[(g,2) for g in gen_action_list_bus_2[-1]],
            #                  "loads_id": [(lo,1) for lo in load_action_list[-1]]+[(lo,2) for lo in load_action_list_bus_2[-1]],
            #                  "lines_or_id": [(li,1) for li in line_or_action_list[-1]]+[(li,2) for li in line_or_action_list_bus_2[-1]],
            #                  "lines_ex_id": [(li,1) for li in line_ex_action_list[-1]]+[(li,2) for li in line_ex_action_list_bus_2[-1]]}})
            # print(do_act)
            # obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_act)
            # print(obs_sim)
            # print(is_done_sim)
            # if is_done_sim:
            #     print("----------------------")
            # print(info_sim)
            # h_letters = [(letter, 2) for letter in 'human']
            # print(h_letters)

            # do_nothing_act = environment.action_space({"change_bus":{"generators_id": [0],"loads_id": [1],"lines_or_id":[3],"lines_ex_id":[7]}})
            # obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_nothing_act)
    #
    # # 在生成 gen_action_list 之后添加以下 print 语句
    # print("Generated gen_action_list:", gen_action_list)
    # print(load_action_list)
    # print(line_or_action_list)
    # print(line_ex_action_list)

    action_index = 0 # max is 60 for 5 bus
    do_act = environment.action_space(
        {"change_bus": {"generators_id": gen_action_list[action_index], "loads_id": load_action_list[action_index], "lines_or_id": line_or_action_list[action_index], "lines_ex_id": line_ex_action_list[action_index]}})
    print(do_act)
    obs_sim, reward_sim, is_done_sim, info_sim = obs.simulate(do_act)
    k = 1
    # environment.action_space({"change_bus":{"generators_id": [0],"loads_id": [1],"lines_or_id":[3],"lines_ex_id":[7]}})
    #np.save("gen_action_list.npy", gen_action_list)
    # 保存 gen_action_list 到文件
    with open("gen_action_list.pkl", "wb") as f:
        pickle.dump(gen_action_list, f)
    with open("load_action_list.pkl", "wb") as f:
        pickle.dump(load_action_list, f)
    with open("line_or_action_list.pkl", "wb") as f:
        pickle.dump(line_or_action_list, f)
    with open("line_ex_action_list.pkl", "wb") as f:
        pickle.dump(line_ex_action_list, f)
    # np.save("load_action_list.npy", load_action_list)
    # np.save("line_or_action_list.npy", line_or_action_list)
    # np.save("line_ex_action_list.npy", line_ex_action_list)
    return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list

def main_function(bus_no):
    # There are a total of 4 files.
    file_names = ["gen_action_list","load_action_list","line_or_action_list","line_ex_action_list"]
    flag_check = False
    for index_item, item in enumerate(file_names):
        #flag_check = os.path.isfile(item+".npy")
        flag_check = os.path.isfile(item + ".pkl")
        if flag_check == False:
            # run the action pruning code!
            gen_action_list, load_action_list, line_or_action_list, line_ex_action_list = prune_action_space(bus_no)
            return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list
    if flag_check == True:
        # load the data from the saved files.
        # gen_action_list = np.load("gen_action_list.npy",allow_pickle=True)
        # load_action_list = np.load("load_action_list.npy",allow_pickle=True)
        # line_or_action_list = np.load("line_or_action_list.npy",allow_pickle=True)
        # line_ex_action_list = np.load("line_ex_action_list.npy",allow_pickle=True)
        gen_action_list = np.load("gen_action_list.pkl", allow_pickle=True)
        load_action_list = np.load("load_action_list.pkl", allow_pickle=True)
        line_or_action_list = np.load("line_or_action_list.pkl", allow_pickle=True)
        line_ex_action_list = np.load("line_ex_action_list.pkl", allow_pickle=True)
    return gen_action_list, load_action_list, line_or_action_list, line_ex_action_list
