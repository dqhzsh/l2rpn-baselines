import numpy as np
import time
import tensorflow as tf
from grid2op.Parameters import Parameters
from grid2op import make
from grid2op.Action import *
import re
from l2rpn_baselines.utils import cli_train

from l2rpn_baselines.FairRL_L2RPN.Common.reward import MyReward
from l2rpn_baselines.FairRL_L2RPN.Common.trainingParam import TrainingParam
from l2rpn_baselines.FairRL_L2RPN.A2C.a2c import Agent

DEFAULT_NAME = "A2C"
def train(env,
        name=DEFAULT_NAME,
        epochs=1,
        iterations=1,
        save_path=None,
        load_path=None,
        logs_dir=None,
        training_param=None,
        filter_action_fun=None,
        verbose=True,
        kwargs_converters={},
        ):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if training_param is None:
        training_param = TrainingParam()


    my_agent = Agent(action_space=env.action_space,
                        mode=name,
                        lr=1e-5,
                        training_param=training_param,
                        **kwargs_converters
                        )

    if load_path is not None:
        if verbose:
            print("INFO: Reloading a model, training parameters will be ignored")
        my_agent.load(load_path)
        training_param = my_agent._training_param

    my_agent.train(env,
                epochs,
                iterations,
                save_path,
                logs_dir,
                training_param,
                verbose
                )
    # my_agent.save(save_path)


if __name__ == "__main__":

    try:
        from lightsim2grid.lightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    args = cli_train().parse_args()

    # Create grid2op game environement
    env_init = None
    try:
        from grid2op.Chronics import MultifolderWithCache
    except:
        from grid2op.Chronics import MultiFolder
        MultifolderWithCache = MultiFolder

    game_param = Parameters()
    game_param.NB_TIMESTEP_COOLDOWN_SUB = 2
    game_param.NB_TIMESTEP_COOLDOWN_LINE = 2
    #env_name = "rte_case14_redisp "
    env_name = "l2rpn_neurips_2020_track2_small"
    env = make(env_name,
            param=game_param,
            reward_class=MyReward,
            action_class=TopologyChangeAndDispatchAction,
            backend=backend
            #chronics_class=MultifolderWithCache
            )
    print("最初动作空间大小为{}".format(env.action_space.size()))
    #env.chronics_handler.set_max_iter(7*288)
    # try:
    #     env.chronics_handler.real_data.set_filter(lambda x: re.match(".*december.*", x) is not None)
    #     env.chronics_handler.real_data.reset()
    # except RuntimeError as exc_:
    #     raise exc_
    # except AttributeError as exc_:
    #     # not available in all grid2op version
    #     pass
    # env.chronics_handler.real_data.
    env_init = env
    if args.nb_env > 1:
        from l2rpn_baselines.utils import make_multi_env
        env = make_multi_env(env_init=env_init, nb_env=int(args.nb_env))

    tp = TrainingParam()
    #tp参数：
    # decay_rate = 0.9,
    # BUFFER_SIZE = 40000,
    # MINIBATCH_SIZE = 64,
    # TOT_FRAME = 3000000,
    # epsilon_decay = 10000,
    # MIN_OBSERVATION = 50,  # 5000
    # final_epsilon = 1 / 300,  # have on average 1 random action per scenario of approx 287 time steps
    # initial_epsilon = 0.1,
    # TAU = 0.01,
    # ALPHA = 1,
    # NUM_FRAMES = 1,
    # gama = 1,
    tp.BUFFER_SIZE = 50000
    tp.n_user = env.n_gen
    tp.reward_size = tp.n_user + 2
    tp.SAVING_NUM = 10
    tp.gama = 0.99
    tp.epsilon_decay = 1000
    tp.MINIBATCH_SIZE = 64
    tp.train_V_iter = 40
    userweight = 0.2
    sys1weight = 1.0
    sys2weight = 0.9
    tp.weight_coef = []
    for _ in range(tp.n_user):
        tp.weight_coef.append(userweight)
    tp.weight_coef.append(sys1weight)
    tp.weight_coef.append(sys2weight)
    tp.loss_weight = []
    actor_weight = 0.5
    critic_weight = 1
    tp.loss_weight.append(actor_weight)
    tp.loss_weight.append(critic_weight)
    tp.model_name ="A2C-{}".format(int(time.time()))

    li_attr_obs_X = ["prod_p", "prod_v", "load_p", "load_q",
                    "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                    "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]
    # observation_size = tp.get_obs_size(env_init, li_attr_obs_X)
    tp.list_attr_obs = li_attr_obs_X
    # sizes = [800, 800, 800, 494, 494, 494]  # sizes of each hidden layers
    # PolicySize = [640, 640, 576, 1024, 960]
    sizes = [800, 800, 494, 494]  # sizes of each hidden layers
    PolicySize = [800, 576, 460]
    CriticSize = [800, 64]
    tp.kwargs_archi = {'sharesizes': sizes,
                    'shareactivs': ["relu" for _ in sizes],  # all relu activation function
                    'Actorsizes':PolicySize,
                    'Actoractivs':["relu" for _ in PolicySize],
                    'Criticsizes': CriticSize,
                    'Criticactivs': ["relu" for _ in PolicySize]
                    }

    # which actions i keep
    tp.kwargs_converters = {"all_actions": None,
                        "set_line_status": False,
                        "change_bus_vect": False,
                        "set_topo_vect": False,
                        "redispatch": True
                        }
    save_path = "../Outputs/Results/A2C/{}".format(tp.model_name)
    logs_dir = "../Outputs/logs/A2C/{}".format(tp.model_name)
    load_path = None
    num_epochs = 50
    num_train_steps = 2000
    num_pre_training_steps = 0

    try:
        train(env,
            name=DEFAULT_NAME,
            epochs=num_epochs,
            iterations=num_train_steps,
            save_path=save_path,
            load_path=load_path,
            logs_dir=logs_dir,
            training_param=tp,
            filter_action_fun=None,
            verbose=False,
            kwargs_converters=tp.kwargs_converters
            )
    finally:
        env.close()
        if args.nb_env > 1:
            env_init.close()