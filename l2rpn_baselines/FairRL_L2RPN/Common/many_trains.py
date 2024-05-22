from FairA2C.train import main as FairA2C_train
# from A2C.train import main as A2C_train
from FairDQN.train import main as FairDQN_train
from DQN.train import main as DQN_train
from FairPPO.train import main as FairPPO_train
from PPO.train import main as PPO_train

if __name__ == '__main__':
    num = 15
    for i in range(num):
        print("第{}次实验".format(i+1))
        FairA2C_train()


