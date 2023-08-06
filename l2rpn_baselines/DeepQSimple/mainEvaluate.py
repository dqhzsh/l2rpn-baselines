from grid2op import make
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
from l2rpn_baselines.DeepQSimple import evaluate

# Create dataset env
env = make("l2rpn_case14_sandbox",
           reward_class=L2RPNSandBoxScore,
           other_rewards={
               "reward": L2RPNReward
           })

# Call evaluation interface
evaluate(env,
         name="AnneOnymous",
         load_path="./models/",
         logs_path="./logs-eval/do-nothing-baseline",
         nb_episode=10,
         nb_process=1,
         max_steps=-1,
         verbose=True,
         save_gif=False)
