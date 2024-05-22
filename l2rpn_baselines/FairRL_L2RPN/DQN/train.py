import time
import tensorflow as tf
from grid2op.Parameters import Parameters
from grid2op import make
from grid2op.Action import *
import re
from l2rpn_baselines.utils import cli_train

from Common.reward import MyReward
from Common.trainingParam import TrainingParam
from DQN.dqn import DeepQAgent

DEFAULT_NAME = "DQN"
def train(env,
          name=DEFAULT_NAME,
          iterations=1,
          num_pre_training_steps=0,
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

    my_agent = DeepQAgent(env=env,
                          action_space=env.action_space,
                          mode=name,
                          lr=1e-4,
                          training_param=training_param,
                          **kwargs_converters,
                         )

    if load_path is not None:
        if verbose:
            print("INFO: Reloading a model, training parameters will be ignored")
        my_agent.load(load_path)
        training_param = my_agent._training_param

    my_agent.train(env,
                   iterations,
                   num_pre_training_steps,
                   save_path,
                   logs_dir,
                   training_param,
                   verbose
                   )
    # my_agent.save(save_path)


def main():
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
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
    env_name = "rte_case14_redisp"
    env = make(env_name,
               param=game_param,
               reward_class=MyReward,
               action_class=TopologyChangeAndDispatchAction,
               backend=backend,
               chronics_class=MultifolderWithCache
               )
    print("最初动作空间大小为{}".format(env.action_space.size()))
    # env.chronics_handler.set_max_iter(7*288)
    try:
        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*((03)|(72)|(57))$", x) is not None)
        env.chronics_handler.real_data.reset()
    except RuntimeError as exc_:
        raise exc_
    except AttributeError as exc_:
        # not available in all grid2op version
        pass
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
    tp.BUFFER_SIZE    = 6000
    tp.n_user         = env.n_gen
    tp.reward_size    = tp.n_user+2

    tp.SAVING_NUM     = 1000
    tp.gama           = 0.99
    tp.alpha          = 2.6
    tp.epsilon_decay  = 1000
    tp.MINIBATCH_SIZE = 256
    tp.buffer_ALPHA   = 0.8

    userweight     = 0.2
    sys1weight     = 1.0
    sys2weight     = 0.9
    tp.weight_coef = []
    for _ in range(tp.n_user):
        tp.weight_coef.append(userweight)
    tp.weight_coef.append(sys1weight)
    tp.weight_coef.append(sys2weight)

    tp.loss_weight = []
    Q_weight = 0.8
    ae_weight = 0.2
    tp.loss_weight.append(Q_weight)
    tp.loss_weight.append(ae_weight)

    tp.model_name = "DQN-{}-10500-500-1_10".format(int(time.time()))

    # Brand-new design of state space
    li_attr_obs_X = ["prod_p", "load_p", "p_or", "p_ex",
                     "topo_vect", "rho", "timestep_overflow",
                     "time_next_maintenance"
                     ]
    tp.list_attr_obs = li_attr_obs_X
    sizes = [128,128,64]  # sizes of each hidden layers
    tp.kwargs_archi = {'sizes': sizes,
                    'activs': ["relu" for _ in sizes],  # all relu activation function
                    }

    # which actions i keep
    # tp.kwargs_converters = {"all_actions": None,
    #                      "set_line_status": False,
    #                      "change_bus_vect": False,
    #                      "set_topo_vect": False
    #                      }
    # save_path = "D:\PycharmProjects\l2rpn_baselines\FairDQN\saved_GGFDQN3.3_Log带GGFscore带优先级\DQN-两千-200"
    save_path = "../Outputs/Results/DQN/{}".format(tp.model_name)
    logs_dir = "../Outputs/logs/DQN/{}".format(tp.model_name)
    load_path = None
    num_train_steps        = 10500
    num_pre_training_steps = 500

    try:
        train(env,
              name=DEFAULT_NAME,
              iterations=num_train_steps,
              num_pre_training_steps = num_pre_training_steps,
              save_path=save_path,
              load_path=load_path,
              logs_dir=logs_dir,
              training_param=tp,
              filter_action_fun=None,
              verbose=True,
              kwargs_converters=tp.kwargs_converters
              )
    finally:
        env.close()
        if args.nb_env > 1:
            env_init.close()


if __name__ == "__main__":
    main()