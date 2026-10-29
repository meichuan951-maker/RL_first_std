# agent.py

class QLearningAgent:

    def __init__(self, alpha, gamma, epsilon):
        """初始化超参数和 Q 表"""
        pass

    def choose_action(self, state, valid_actions):
        """使用 ε-greedy 选择动作"""
        pass

    def update(self, state, action, reward,next_state, next_valid_actions, done):
        """使用 Q-Learning 公式更新 Q 值"""
        pass

    def save(self, path):
        """保存 Q 表"""
        pass

    def load(self, path):
        """加载 Q 表"""
        pass