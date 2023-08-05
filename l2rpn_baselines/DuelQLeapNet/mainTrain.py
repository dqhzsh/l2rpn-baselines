import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam
from l2rpn_baselines.DuelQLeapNet import train, LeapNet_NNParam

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
li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                 "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                 "time_before_cooldown_sub", "timestep_overflow", "line_status", "rho"]
# compared to the other baseline, we have different inputs at different place, this is how we split it
li_attr_obs_Tau = ["rho", "line_status"]
sizes = [800, 800, 800, 494, 494, 494]

# nn architecture
x_dim = LeapNet_NNParam.get_obs_size(env, li_attr_obs_X)
tau_dims = [LeapNet_NNParam.get_obs_size(env, [el]) for el in li_attr_obs_Tau]

kwargs_archi = {'sizes': sizes,
                'activs': ["relu" for _ in sizes],
                'x_dim': x_dim,
                'tau_dims': tau_dims,
                'tau_adds': [0.0 for _ in range(len(tau_dims))],  # add some value to taus
                'tau_mults': [1.0 for _ in range(len(tau_dims))],  # divide by some value for tau (after adding)
                "list_attr_obs": li_attr_obs_X,
                "list_attr_obs_tau": li_attr_obs_Tau
                }

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