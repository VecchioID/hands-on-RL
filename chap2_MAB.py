# -*- coding:utf-8 -*-
# __author__ = 'Vecchio'
import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit(object):
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        # 获奖概率最大的拉杆和概率
        self.max_K = np.argmax(self.probs)
        self.max_prob = self.probs[self.max_K]
        self.K = K

    def step(self, k):
        if np.random.uniform() < self.probs[k]:
            return 1
        else:
            return 0


# random seed
np.random.seed(1)
K = 10
bandit = BernoulliBandit(K)
print("随机生成一个%d赌博机" % K)
print("最大概率奖励的拉杆号是%d,概率是%.4f" % (bandit.max_K, bandit.max_prob))

# 用一个solver基础类实现上述的多臂老虎机的求解方案。
'''
需要实现的功能如下：
根据策略选择动作、根据动作获取奖励、更新期望奖励估值，更新累计懊悔和计数器
'''


class Solver(object):
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(bandit.K)
        # 当前懊悔
        self.regret = 0.
        # 定义累积regret
        self.regrets = []
        # 维护一个动作列表
        self.actions = []

    def update_regret(self, k):
        self.regret = self.regret + self.bandit.max_prob - self.bandit.probs[k]  # 对应action的期望奖励，因为奖励是1，所以等于概率值
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 具体的选择由相应的策略实现，可以采用单纯的贪心策略，只利用，不探索
        raise NotImplementedError

    def run(self, num_steps):
        # 运行次数的循环
        for _ in range(num_steps):
            # 运行一次
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


# 探索与利用的平衡，设置探索的概率是0.01, T = 5000
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的奖励值
        self.estimates = np.ones(bandit.K) * init_prob

    def run_one_step(self):
        if np.random.uniform() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=0.1)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


# 接下来尝试epsilon随时间衰减的算法
class DecayingEpsilonGreedy(Solver):
    '''随时间衰减的算法，继承Solver类'''

    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1. / self.total_count:  # 这里这个贪心值从1开始衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


# 上置信界算法，核心是引入不确定度量U(a)
class UCB(Solver):
    """ UCB算法,继承Solver类 """

    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    # 原来的是根据概率p进行随机或最大的策略选择，而今是计算UCB进行策略选择，因为p= 1/ t，所以蕴含在了UCB之中
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])


class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])