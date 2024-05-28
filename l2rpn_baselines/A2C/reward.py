from grid2op.Reward import L2RPNReward,BaseReward,GameplayReward,CloseToOverflowReward,RedispReward,EconomicReward
import numpy as np
from grid2op.dtypes import dt_float
class MyReward(L2RPNReward):
    def __init__(self, logger=None):
        L2RPNReward.__init__(self, logger=logger)
        self.last_reward = RedispReward(logger=logger)

    def initialize(self, env):
        self.last_reward.initialize(env)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        
        self.reward_min_vect = np.zeros(env.n_gen+2)
        self.reward_max_vect = np.zeros(env.n_gen+2)
        for i in range(env.n_gen):
            # self.reward_max[i] = np.float(1.0)
            self.reward_min_vect[i] = self.reward_min
            self.reward_max_vect[i] = self.reward_max
        self.reward_min_vect[-2] = 0.0
        self.reward_max_vect[-2] = 1.0
        self.reward_min_vect[-1] = self.last_reward.reward_min
        self.reward_max_vect[-1] = 1.0

 

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        # if has_error or is_illegal or is_ambiguous:
        #     # previous action was bad
        #     res = np.self.reward_min
        # elif is_done:
        #     # really strong reward if an episode is over without game over
        #     res = self.reward_max
        # else:
        if has_error:
            res = self.reward_min_vect
            # severe punishment on the action leading to blackout
            # res = self.reward_min - 1.0

        elif is_illegal or is_ambiguous:
            # Did not respect the rules
            res = self.reward_min_vect
            res[-1] = self.last_reward.reward_illegal_ambiguous
            # res = self.reward_max/5

            # mild punishment to disencourage DoNothing action
            # res = self.reward_min - 0.5
        elif is_done:
            # really strong reward if an episode is over without game over
            res = self.reward_max_vect
        else:
            res = np.zeros(env.n_gen + 2)
            res_user = self.__get_gen_capacity_usage(env)
            for i in range(env.n_gen):
                res[i] = res_user[i]

            # res[0] = res_user[0]          #nuclear
            # res[1] = res_user[1]          #thermal
            # res[2] = res_user[2]          #wind
            # res[3] = res_user[3]          #solar
            # res[4] = res_user[4]          #thermal

            # lineflow_ratio = env.current_obs.rho
            # sum = np.float(0.0)
            # for ratio in lineflow_ratio:
            #     sum += min(ratio,1)
            # res[-2] = 1 - sum/env.n_line

            line_cap = self.__get_lines_capacity_usage(env)
            res[-2] = np.sum(line_cap)/env.n_line

            # res[-2] = CloseToOverflowReward(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # res[5] = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # res[5] /= env.n_line
            # if not np.isfinite(res):
            #     res[5] = self.reward_min
            # res[6] = GameplayReward(action, env, has_error, is_done, is_illegal, is_ambiguous)

            # if has_error:
            #     res[6] =self.reward_min
            # elif is_illegal or is_ambiguous:
            # # Did not respect the rules
            #     res[6] =self.reward_min /float(2)
            # else:

            # last_reward = RedispReward()
            # last_reward = EconomicReward()
            # last_reward.initialize(env)
            # res[-1] = 0.0005 * last_reward(action, env, has_error, is_done, is_illegal, is_ambiguous)   #权值0.0005使得各reward分量的数量级一致
            res[-1] = self.last_reward(action, env, has_error, is_done, is_illegal, is_ambiguous) / self.last_reward.reward_max
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
            ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
            thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
            thermal_limits += 1e-1  # for numerical stability
            relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

            x = np.minimum(relative_flow, dt_float(1.0))
            lines_capacity_usage_score = np.maximum(
                dt_float(1.0) - x**2, np.zeros(x.shape, dtype=dt_float)
            )
            return lines_capacity_usage_score
            
    @staticmethod
    def __get_gen_capacity_usage(env):
        gen_p_obs = env.current_obs.gen_p
        gen_p_max = env.gen_pmax + 1e-1 # for numerical stability
        gen_capacity_usage = np.array(gen_p_obs/gen_p_max)
        # clip the gen_capacity_usage between [0, 1]
        gen_capacity_usage = np.minimum(gen_capacity_usage, dt_float(1.0)) 
        gen_capacity_usage = np.maximum(gen_capacity_usage, dt_float(0.0)) 
        return gen_capacity_usage


class MyEvaluateReward(L2RPNReward):
    def __init__(self, max_lines=5):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        # if has_error or is_illegal or is_ambiguous:
        #     # previous action was bad
        #     res = np.self.reward_min
        # elif is_done:
        #     # really strong reward if an episode is over without game over
        #     res = self.reward_max
        # else:
        self.reward_min = np.zeros(env.n_gen + 2)
        self.reward_max = np.zeros(env.n_gen + 2)
        for i in range(env.n_gen + 2):
            self.reward_max[i] = np.float(1.0)
        if has_error:
            res = self.reward_min
        elif is_illegal or is_ambiguous:
        # Did not respect the rules
            res = self.reward_max/5
        elif is_done:
        # really strong reward if an episode is over without game over
            res = self.reward_max
        else:
            res = np.zeros(env.n_gen + 2)
            res_user = np.array(env.current_obs.gen_p/env.gen_pmax)
            for i in range(env.n_gen):
                res[i] = res_user[i]

            # res[0] = res_user[0]          #nuclear
            # res[1] = res_user[1]          #thermal
            # res[2] = res_user[2]          #wind
            # res[3] = res_user[3]          #solar
            # res[4] = res_user[4]          #thermal

            lineflow_ratio = env.current_obs.rho
            sum = np.float(0.0)
            for ratio in lineflow_ratio:
                sum += min(ratio,1)
            res[-2] = 1 - sum/env.n_line
            # res[-2] = CloseToOverflowReward(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # res[5] = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
            # res[5] /= env.n_line
            # if not np.isfinite(res):
            #     res[5] = self.reward_min
            # res[6] = GameplayReward(action, env, has_error, is_done, is_illegal, is_ambiguous)

            # if has_error:
            #     res[6] =self.reward_min
            # elif is_illegal or is_ambiguous:
            # # Did not respect the rules
            #     res[6] =self.reward_min /float(2)
            # else:
            last_reward = RedispReward()
            # last_reward = EconomicReward()
            last_reward.initialize(env)
            res[-1] = last_reward.__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)

        weight = [0.200, 0.200, 0.200, 0.200, 0.200, 1.00, 1.00]
        reward = dt_float(np.dot(weight, res))
        return reward

# class MyEvaluateReward(MyReward):
#     def initialize(self, env):
#         pass
#
#     def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
#         result = MyReward().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
#         weight = [0.2, 0.2, 0.3, 0.2, 0.2, 1, 1]
#         res = float(np.dot(weight , result))
#
#         return res

class GGFEvaluateReward(BaseReward):
    def __init__(self, max_lines=5):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.max_overflowed = dt_float(max_lines)

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error,
                 is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        res_user        = np.array(env.current_obs.gen_p / env.gen_pmax)
        sorted_q_values = np.sort(res_user, axis=-1)[::-1]
        omega           = [1 / 2 ** n for n in range(len(res_user))]
        w               = np.dot(sorted_q_values, omega)

        if w > 0.9:
            reward = w
        else:
            reward = self.reward_min

        # thermal_limits = env.backend.get_thermal_limit()
        # lineflow_ratio = env.current_obs.rho
        #
        # close_to_overflow = dt_float(0.0)
        # for ratio, limit in zip(lineflow_ratio, thermal_limits):
        #     # Seperate big line and small line
        #     if (limit < 400.00 and ratio >= 0.95) or ratio >= 0.975:
        #         close_to_overflow += dt_float(1.0)
        #
        # close_to_overflow = np.clip(close_to_overflow,
        #                             dt_float(0.0), self.max_overflowed)
        # reward = np.interp(close_to_overflow,
        #                    [dt_float(0.0), self.max_overflowed],
        #                    [self.reward_max, self.reward_min])
        return reward