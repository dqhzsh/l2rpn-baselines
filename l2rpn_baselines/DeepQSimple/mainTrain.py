import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DeepQSimple import train

# define the environment
env = grid2op.make("l2rpn_case14_sandbox",
                   reward_class=L2RPNReward)

# use the default training parameters
tp = TrainingParam()

# this will be the list of what part of the observation I want to keep
# more information on https://grid2op.readthedocs.io/en/latest/observation.html#main-observation-attributes
li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                 "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                 "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]

# neural network architecture
observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
sizes = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
kwargs_archi = {'observation_size': observation_size,
                'sizes': sizes,
                'activs': ["relu" for _ in sizes],  # all relu activation function
                "list_attr_obs": li_attr_obs_X}

# select some part of the action
# more information at https://grid2op.readthedocs.io/en/latest/converter.html#grid2op.Converter.IdToAct.init_converter
kwargs_converters = {"all_actions": None,
                     "set_line_status": False,
                     "change_bus_vect": True,
                     "set_topo_vect": False
                     }
# define the name of the model
nm_ = "AnneOnymous"
save_path = "./models"
logs_dir = "./logs"
try:
    train(env,
          name=nm_,
          iterations=10000,
          save_path=save_path,
          load_path=None,
          logs_dir=logs_dir,
          training_param=tp,
          kwargs_converters=kwargs_converters,
          kwargs_archi=kwargs_archi)
finally:
    env.close()
