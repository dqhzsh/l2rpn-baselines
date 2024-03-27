try:
    import numpy as np
    import copy
    import os
    from grid2op.Parameters import Parameters
    from grid2op.Reward import L2RPNReward, CombinedReward, CloseToOverflowReward, GameplayReward
    import sys
    from l2rpn_baselines.AsynchronousActorCritic_init.AsynchronousActorCritic import A3CAgent
    from l2rpn_baselines.AsynchronousActorCritic_init.user_environment_make import set_environement
    # import pytorch
except ImportError as exc_:
    raise ImportError("AsynchronousActorCritic baseline impossible to load the required dependencies for training the model. The error was: \n {}".format(exc_))


def train(env,name,iterations,save_path,load_path,env_name,profiles_chronics,time_step_end,Hyperparameters,Thread_count):
    """
    Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environmnent on which the baseline will be trained
        name: ``str``
            Fancy name you give to this baseline that is being trained.
        iterations: ``int``
            Number of training iterations to perform
        save_path: ``str``
            The path where the baseline will be saved at the end of the training procedure.
        load_path: ``str``
            Path where to look for reloading the model. Use ``None`` if no model should be loaded.
        kwargs:
            Other key-word arguments that you might use for training.
    """

    # NB: "env" is not used. Instead the A3C agent will call function from "user_environment_make.py" in
    # "each unique thread" to create duplicate multiple environments in parallel. By no means this is perfect but this
    # works and we recommend using MultiEnv class from Grid2Op for future implementations that uses multiple threads.

    env_temp = set_environement(None, env_name, profiles_chronics)

    # Define the size of state space and action space.
    state_size = env_temp.observation_space.size()

    action_space = env_temp.action_space
    print(action_space.size())

    # Create an agent architecture.
    global_agent = A3CAgent(state_size, action_space, env_name,
                            profiles_chronics,iterations,time_step_end,Hyperparameters,Thread_count,train_flag=True,
                            save_path=save_path)

    # Print model summary
    global_agent.actor.summary()
    global_agent.critic.summary()

    if load_path is not None:
        # Load the agent with trained neural network weights.
        try:
            global_agent.load_model(nn_weight_name=name,load_path=load_path)
            print("Loaded saved NN model parameters \n")
        except:
            print("Issue with loading the NN weights/No existing model is found or saved model sizes do not match. Exiting...\n")
            exit()
    else:
        print("Training the A3C agent from scratch i.e., without loading the previously trained neural network weights.")

    global_agent.train(name)

if __name__ == "__main__":
    # Name of the ".h5" files that stores the actor and critic neural network weights.
    #name = "grid2op_14_a3c"
    name = "l2rpn_2020_track2"
    # Total number of episodes to train the A3C agent.
    EPISODES_train = 100000
    # Maximum number of time steps or iterations taken in each episode during the training.
    time_step_end = 3000
    # Number of parallel workers or threads to train the A3C agent.
    Thread_count = 7
    # location where ".h5" files to be saved.
    save_path = os.path.join(os.getcwd(),"nn_weight_folder")
    # location to load pre-trained ".h5" files (neural network weights) OR provide "None" as input.
    load_path = None #os.path.join(os.getcwd(),"nn_weight_folder")

    # Neural network architecture hyper parameters. For more details on structure see the ".summary()" that will be
    # printed when running this code.
    Hyperparameters = {}
    Hyperparameters["actor_learning_rate"] = 0.0003
    Hyperparameters["critic_learning_rate"] = 0.0003
    Hyperparameters["discount_factor"] = 0.95
    Hyperparameters["size_of_hidden_layer_1"] = 200
    Hyperparameters["size_of_hidden_layer_2"] = 100

    # Create the environment
    # env = user_environment_make.set_environement(None, env_name, profiles_chronics)

    # Name of the Grid2Op environment to train the A3C RL agent on.
    #env_name = 'l2rpn_case14_sandbox'
    env_name = 'l2rpn_neurips_2020_track2_small'
    # Location of the chronic files. Change this to your chronics location.
    #profiles_chronics = r"C:\Users\dqh\data_grid2op\l2rpn_case14_sandbox\chronics"
    profiles_chronics = r"C:\Users\dqh\data_grid2op\l2rpn_neurips_2020_track2_small\l2rpn_neurips_2020_track2_x1\chronics"
    print("_____________________________________________")
    print("NOTE: PLEASE MAKE CHANGES TO THE FILE user_environment_make.py TO MAKE YOUR ENVIRONMENT. This "
          "user_environment_make.py will be called wherever it is necessary when training the A3C agent")
    print("_____________________________________________")

    # Train function that trains the A3C agent.
    train(env=None,name=name,iterations=EPISODES_train,save_path=save_path,load_path=load_path,
          env_name=env_name,profiles_chronics=profiles_chronics,time_step_end=time_step_end,
          Hyperparameters=Hyperparameters,Thread_count=Thread_count)


