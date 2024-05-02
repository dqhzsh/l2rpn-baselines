from grid2op import make
from grid2op.Parameters import Parameters
from grid2op.Reward import *
from grid2op.Action import *
from lightsim2grid import LightSimBackend

def set_environement(start_id,env_name,profiles_chronics):
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True

    #env = make(env_name,chronics_path= profiles_chronics, reward_class=CombinedScaledReward, action_class= TopologyChangeAndDispatchAction,param=param)
    backend = LightSimBackend()
    env = make(env_name, chronics_path=profiles_chronics, reward_class=CombinedReward, param=param, backend=backend)

    # Register custom reward for training
    cr = env._reward_helper.template_reward
    #cr.addReward("reco", LinesReconnectedReward(), 50.0)
    cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    cr.addReward("game", GameplayReward(), 100.0)
    #cr.addReward("redisp", RedispReward(), 1e-3)
    #cr.set_range(-1.0, 1.0)
    # Initialize custom rewards
    cr.initialize(env)

    # env = make(env_name,chronics_path= profiles_chronics, reward_class=CombinedReward,param=param)
    # # Register custom reward for training
    # cr = env._reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 50.0)
    # cr.addReward("game", GameplayReward(), 100.0)
    # cr.initialize(env)

    # Debug prints
    print("Debug prints --->:")
    print("Chronics location that being used:", env.chronics_handler.path)
    print("Grid location being used:", env._init_grid_path)
    print("Reward class that is being used:", env._rewardClass)
    print("Action type class being used:", env._actionClass)
    print("Observation type class being used:", env._observationClass)
    print("Backend CSV file key names:", env._names_chronics_to_backend)
    print("Legal action class being used:", env._legalActClass)
    print("Voltage controller class being used:", env._voltagecontrolerClass)

    if start_id != None:
        env.chronics_handler.tell_id(start_id)
        print(dir(env.chronics_handler.real_data))
        print("Thread number:",start_id,", ID of chronic current folder:",env.chronics_handler.real_data.get_id())
    return env
