import os
import tensorflow as tf

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Action import *
from grid2op.Episode import EpisodeReplay
from grid2op.Reward import L2RPNReward,GameplayReward,CloseToOverflowReward,RedispReward,EconomicReward

from Common.reward import MyEvaluateReward,GGFEvaluateReward
from Common.trainingParam import TrainingParam
from FairA2C.fairA2C import Agent


def evaluate(env,
             load_path=None,
             logs_path=None,
             training_param=None,
             nb_episode= 1,
             nb_process= 1,
             max_steps= -1,
             # TODO 2020第一max_steps为864
             num_frames= 4,
             verbose=False,
             save_gif=False,
             ):
    # Set config
    # D3QNConfig.N_FRAMES = num_frames
    # D3QNConfig.VERBOSE = verbose

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    if training_param is None:
        training_param = TrainingParam()

    # Create agent
    agent = Agent(env=env,
        action_space=env.action_space,
                          mode="A2C",
                          lr=1e-5,
                          training_param=training_param,
                          **training_param.kwargs_converters,
                          is_training=False
                          )
    # agent = DeepQAgent(env.observation_space,
    #                   env.action_space,
    #                   is_training=False)

    # Load weights from file

    agent.load(env, load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)
    print("1")

    # Print model summary
    if verbose:
        stringlist = []
        agent.deep_q.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)

    print("2")
    print(verbose)
    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=verbose)
    print("3")
    # Print summary
    total_reward = 0
    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step,
                                                            max_ts)
            total_reward += cum_reward
            print(msg_tmp)

        mean_reward = total_reward/nb_episode
        print("\tmean reward: {:.6f}".format(mean_reward))
    print("4")
    if save_gif:
        save_log_gif(logs_path, res)
    print("5")
    return res

def save_log_gif(path_log, res, gif_name=None):
    """
    Output a gif named (by default "episode.gif") that is the replay of the episode in a gif format,
    for each episode in the input.

    Parameters
    ----------
    path_log: ``str``
        Path where the log of the agents are saved.

    res: ``list``
        List resulting from the call to `runner.run`

    gif_name: ``str``
        Name of the gif that will be used.

    """
    init_gif_name = gif_name
    ep_replay = EpisodeReplay(path_log)
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        if gif_name is None:
            gif_name = chron_name
        gif_path = os.path.join(path_log, chron_name, gif_name)
        print("Creating {}.gif".format(gif_name))
        ep_replay.replay_episode(episode_id=chron_name, gif_name=gif_name, display=False)
        print("Wrote {}.gif".format(gif_path))
        gif_name = init_gif_name


if __name__ == "__main__":
    # Parse command line
    # args = cli()
    # Create dataset env
    env_name = "rte_case14_redisp_test"
    env = make(env_name,
               reward_class=GGFEvaluateReward,
               action_class=TopologyChangeAndDispatchAction
               )

    load_file = "../../Outputs/paint/FairA2C-1666683956-50-2000-1_10"
    load_path = os.path.join(load_file, "A2C.h5")
    param_path = os.path.join(load_file, "training_params.json")
    tp = TrainingParam.from_json(param_path)

    # logs_dir = "../Outputs/logs-evals/FairA2C/{}".format(tp.model_name)
    logs_dir = "../Outputs/logs-evals/FairA2C/FairA2C-1666683956-50-2000-1_10-test1"
    nb_episode = 3
    # Call evaluation interface
    try:
        evaluate(env,
                 load_path=load_path,
                 logs_path=logs_dir,
                 training_param=tp,
                 nb_episode=nb_episode,
                 verbose=True,
                 save_gif=False
                 )
    finally:
        env.close()
