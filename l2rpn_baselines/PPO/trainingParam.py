import os
import json
import numpy as np


class TrainingParam(object):
    """
    A class to store the training parameters of the models. It was hard coded in the notebook 3.
    self.decay_rate: 以前按乘进行epsilon衰减时的参数, 现在无用
    self.BUFFER_SIZE: 经验缓冲池大小
    self.MINIBATCH_SIZE: train时的batch大小
    self.TOT_FRAME = TOT_FRAME
    self.epsilon_decay: epsilon衰减的总步数
    self.MIN_OBSERVATION = MIN_OBSERVATION   # 5000无用
    self.final_epsilon : epsilon的最终值 # have on average 1 random action per scenario of approx 287 time steps
    self.initial_epsilon = INITIAL_EPSILON
    self.TAU = TAU
    self.NUM_FRAMES = NUM_FRAMES  #q_network隐藏层个数的参数, 无用了
    self.SAVING_NUM = SAVING_NUM
    self.alpha: GGF前的权重, 公式中的λ
    self.gama: 更新Q时的折扣因子
    self.lmbda: #GAE的λ, 一般->0,远小于gama; λ=0, 为A=r+γV(s')-V(s), λ=1, 为A=sum(r)-V(s)
    self.weight_coef 奖励向量的权重
    self.reward_size: 奖励向量的大小
    self.list_attr_obs: 状态空间参数
    self.n_user : 用户个数
    self.kwargs_archi: 网络结构
    self.buffer_ALPHA: 优先级缓冲池控制优先级程度
    MINIBATCH_SIZE,alpha,gama,weight_coef,kwargs_archi()
    """
    _all_attr = ["decay_rate" ,"BUFFER_SIZE","MINIBATCH_SIZE","TOT_FRAME",
                 "epsilon_decay","final_epsilon","initial_epsilon",
                 "MIN_OBSERVATION","TAU","SAVING_NUM","alpha","buffer_ALPHA",
                 "gama","lmbda","weight_coef","loss_weight","reward_size",
                 "n_user","list_attr_obs","kwargs_archi","kwargs_converters",
                 "train_V_iter","model_name"]
    def __init__(self,
                 DECAY_RATE        = 0.9,
                 BUFFER_SIZE       = 4000,
                 MINIBATCH_SIZE    = 64,
                 TOT_FRAME         = 3000000,
                 EPSILON_DECAY     = 300,
                 MIN_OBSERVATION   = 5000,                              #5000
                 FINAL_EPSILON     = 1/300,                             # have on average 1 random action per scenario of approx 287 time steps
                 INITIAL_EPSILON   = 0.5,
                 TAU               = 0.01,
                 ALPHA             = 2,
                 buffer_ALPHA      = 0.7,
                 NUM_FRAMES        = 1,
                 SAVING_NUM        = 100,
                 GAMA              = 0.9,
                 lmbda             = 0.2,
                 weight_coef       = [0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.9],
                 loss_weight       = [1, 1],
                 reward_size       = 1,
                 n_user            = 1,
                 list_attr_obs     = [],
                 kwargs_archi      = {},
                 kwargs_converters = {"all_actions": None},
                 train_V_iter      = 40,
                 model_name        = "NULL"
    ):
        self.decay_rate      = DECAY_RATE
        self.BUFFER_SIZE     = BUFFER_SIZE
        self.TOT_FRAME       = TOT_FRAME
        self.MIN_OBSERVATION = MIN_OBSERVATION   # 5000
        self.final_epsilon   = FINAL_EPSILON  # have on average 1 random action per scenario of approx 287 time steps
        self.initial_epsilon = INITIAL_EPSILON
        self.TAU             = TAU
        # self.NUM_FRAMES = NUM_FRAMES  #q_network隐藏层个数的参数
        self.SAVING_NUM    = SAVING_NUM
        self.reward_size   = reward_size
        self.list_attr_obs = [str(el) for el in list_attr_obs]
        self.n_user        = n_user


        self.MINIBATCH_SIZE    = MINIBATCH_SIZE
        self.epsilon_decay     = EPSILON_DECAY
        self.alpha             = ALPHA   #GGF的权重, 公式中的λ
        self.buffer_ALPHA      = buffer_ALPHA
        self.gama              = GAMA
        self.lmbda             = lmbda   #GAE的λ
        self.weight_coef       = weight_coef
        self.loss_weight       = loss_weight
        self.kwargs_archi      = kwargs_archi
        self.kwargs_converters = kwargs_converters
        self.train_V_iter      = train_V_iter
        self.model_name        = model_name

    def get_obs_size(self, env, list_attr_name):
        """get the size of the flatten observation"""
        res = 0
        for obs_attr_name in list_attr_name:
            beg_, end_, dtype_ = env.observation_space.get_indx_extract(obs_attr_name)
            res += end_ - beg_  # no "+1" needed because "end_" is exclude by python convention
        return res

    def to_dict(self):
        """serialize this instance to a dictionnary."""
        res = {}
        # for attr_nm in self._int_attr:
        #     tmp = getattr(self, attr_nm)
        #     if tmp is not None:
        #         res[attr_nm] = int(tmp)
        #     else:
        #         res[attr_nm] = None
        # for attr_nm in self._float_attr:
        #     tmp = getattr(self, attr_nm)
        #     if tmp is not None:
        #         res[attr_nm] = float(tmp)
        #     else:
        #         res[attr_nm] = None
        for attr_nm in self._all_attr:
            tmp = getattr(self, attr_nm)
            if tmp is not None:
                res[attr_nm] = tmp
            else:
                res[attr_nm] = None
        return res

    def save_as_json(self, path, name=None):
        """save this instance as a json"""
        res = self.to_dict()
        if name is None:
            name = "training_parameters.json"
        if not os.path.exists(path):
            raise RuntimeError("Directory \"{}\" not found to save the training parameters".format(path))
        if not os.path.isdir(path):
            raise NotADirectoryError("\"{}\" should be a directory".format(path))
        path_out = os.path.join(path, name)
        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(res, fp=f, indent=4, sort_keys=True)

    def from_dict(tmp):
        """initialize this instance from a dictionary"""
        if not isinstance(tmp, dict):
            raise RuntimeError("TrainingParam from dict must be called with a dictionary, and not {}".format(tmp))
        res = TrainingParam()
        # for attr_nm in TrainingParam._int_attr:
        #     if attr_nm in tmp:
        #         tmp_ = tmp[attr_nm]
        #         if tmp_ is not None:
        #             setattr(res, attr_nm, int(tmp_))
        #         else:
        #             setattr(res, attr_nm, None)
        #
        # for attr_nm in TrainingParam._float_attr:
        #     if attr_nm in tmp:
        #         tmp_ = tmp[attr_nm]
        #         if tmp_ is not None:
        #             setattr(res, attr_nm, float(tmp_))
        #         else:
        #             setattr(res, attr_nm, None)
        for attr_nm in TrainingParam._all_attr:
            if attr_nm in tmp:
                tmp_ = tmp[attr_nm]
                if tmp_ is not None:
                    setattr(res, attr_nm, tmp_)
                else:
                    setattr(res, attr_nm, None)
        # res.update_nb_iter = res._update_nb_iter
        # res.initial_epsilon = res._initial_epsilon
        # res._compute_exp_facto()
        return res

    def from_json(json_path):
        """initialize this instance from a json"""
        if not os.path.exists(json_path):
            raise FileNotFoundError("No path are located at \"{}\"".format(json_path))
        with open(json_path, "r") as f:
            dict_ = json.load(f)
        return TrainingParam.from_dict(dict_)