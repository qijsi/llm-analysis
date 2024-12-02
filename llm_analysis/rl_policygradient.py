import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

# 模拟性能模型函数
def performance_model(config):
    """
    模拟性能模型，根据配置预测训练吞吐量（越大越好）。
    """
    base_throughput = 1000
    tp_factor = 1 + np.log2(config['tp'])
    dp_factor = 1 + np.log2(config['dp'])
    pp_factor = 1 + np.log2(config['pp'])
    recompute_factor = 0.9 if config['activation_recompute'] else 1.0
    grad_accum_factor = 1 / config['grad_accum_steps']
    batch_size_factor = config['batch_size'] / 32

    throughput = base_throughput * tp_factor * dp_factor * pp_factor * recompute_factor * grad_accum_factor * batch_size_factor
    return throughput

# 定义 Actor-Critic 网络，结合 PPO 的思想
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value

# 定义高层代理（Manager）
class HighLevelAgent:
    def __init__(self, total_gpus, gpus_per_node, learning_rate=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.gamma = gamma
        self.eps_clip = eps_clip  # PPO 中的剪辑参数
        self.K_epochs = K_epochs  # PPO 中的更新次数

        # 高层动作空间（并行策略组合）
        self.high_level_actions = self.generate_high_level_actions()
        self.action_size = len(self.high_level_actions)
        self.state_size = 3  # 可以根据需要调整

        # Actor-Critic 网络和优化器
        self.policy = ActorCritic(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 旧策略网络，用于 PPO 的策略更新
        self.policy_old = ActorCritic(self.state_size, self.action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # 经验存储
        self.memory = []

    def generate_high_level_actions(self):
        # 生成并行策略组合
        tp_options = [1, 2, 4, 8]
        dp_options = [1, 2, 4, 8]
        pp_options = [1, 2, 4, 8]

        high_level_actions = []
        for tp, dp, pp in product(tp_options, dp_options, pp_options):
            total_parallel_size = tp * dp * pp
            if total_parallel_size <= self.total_gpus and tp <= self.gpus_per_node:
                action = {'tp': tp, 'dp': dp, 'pp': pp}
                high_level_actions.append(action)
        return high_level_actions

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        action = self.high_level_actions[action_idx.item()]

        # 存储
        self.memory.append({
            'state': state,
            'action': action_idx,
            'log_prob': m.log_prob(action_idx)
        })

        return action

    def update(self):
        # 准备数据
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.tensor([m['action'] for m in self.memory])
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        rewards = [m['reward'] for m in self.memory]

        # 计算折扣回报
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 标准化

        for _ in range(self.K_epochs):
            # 计算新策略下的动作概率和状态值
            action_probs, state_values = self.policy(states)
            m = torch.distributions.Categorical(action_probs)
            new_log_probs = m.log_prob(actions)

            # 计算优势函数
            advantages = returns - state_values.detach().squeeze()

            # 计算 PPO 损失
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), returns)

            # 更新策略
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空经验
        self.memory = []

    def get_state(self):
        # 定义高层状态，这里简化为全零向量
        return [0.0, 0.0, 0.0]

    def train(self, episodes):
        for episode in range(episodes):
            state = self.get_state()
            action = self.select_action(state)
            print(f"\nHigh Level Episode {episode + 1}/{episodes}")
            print(f"Selected High Level Action: {action}")

            # 创建低层代理，并传递高层动作
            worker = LowLevelAgent(action)
            reward = worker.train()

            # 存储奖励
            self.memory[-1]['reward'] = reward
            print(f"High Level Reward (Throughput): {reward}")

            # 更新策略网络
            self.update()

# 定义低层代理（Worker）
class LowLevelAgent:
    def __init__(self, high_level_action, learning_rate=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.tp = high_level_action['tp']
        self.dp = high_level_action['dp']
        self.pp = high_level_action['pp']
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 低层动作空间
        self.low_level_actions = self.generate_low_level_actions()
        self.action_size = len(self.low_level_actions)
        self.state_size = 3  # 可以根据需要调整

        # Actor-Critic 网络和优化器
        self.policy = ActorCritic(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 旧策略网络，用于 PPO 的策略更新
        self.policy_old = ActorCritic(self.state_size, self.action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # 经验存储
        self.memory = []

    def generate_low_level_actions(self):
        activation_recompute_options = [True, False]
        grad_accum_steps_options = [1, 2, 4, 8]
        batch_size_options = [1, 2, 4, 8, 16, 32]

        low_level_actions = []
        for ar, ga, bs in product(activation_recompute_options, grad_accum_steps_options, batch_size_options):
            action = {
                'activation_recompute': ar,
                'grad_accum_steps': ga,
                'batch_size': bs
            }
            low_level_actions.append(action)
        return low_level_actions

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        action = self.low_level_actions[action_idx.item()]

        # 存储
        self.memory.append({
            'state': state,
            'action': action_idx,
            'log_prob': m.log_prob(action_idx)
        })

        return action

    def update(self):
        # 准备数据
        states = torch.stack([m['state'] for m in self.memory])
        actions = torch.tensor([m['action'] for m in self.memory])
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        rewards = [m['reward'] for m in self.memory]

        # 计算折扣回报
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 标准化

        for _ in range(self.K_epochs):
            # 计算新策略下的动作概率和状态值
            action_probs, state_values = self.policy(states)
            m = torch.distributions.Categorical(action_probs)
            new_log_probs = m.log_prob(actions)

            # 计算优势函数
            advantages = returns - state_values.detach().squeeze()

            # 计算 PPO 损失
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), returns)

            # 更新策略
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空经验
        self.memory = []

    def get_state(self):
        # 定义低层状态，这里简化为全零向量
        return [0.0, 0.0, 0.0]

    def train(self):
        state = self.get_state()
        action = self.select_action(state)
        print(f"  Selected Low Level Action: {action}")

        # 评估配置，获取奖励
        config = {
            'tp': self.tp,
            'dp': self.dp,
            'pp': self.pp,
            'activation_recompute': action['activation_recompute'],
            'grad_accum_steps': action['grad_accum_steps'],
            'batch_size': action['batch_size']
        }

        # 使用性能模型预测吞吐量
        predicted_throughput = performance_model(config)

        # 实际运行进行校正（可选）
        if random.random() < 0.1:  # 10%的概率进行实际运行
            actual_throughput = self.run_actual_training(config)
            reward = actual_throughput
        else:
            reward = predicted_throughput

        # 存储奖励
        self.memory[-1]['reward'] = reward
        print(f"  Low Level Reward (Throughput): {reward}")

        # 更新策略网络
        self.update()

        return reward

    def run_actual_training(self, config):
        """
        实际运行单层模型训练，返回实际吞吐量。
        """
        print("  Running actual training for calibration...")
        actual_throughput = performance_model(config) * random.uniform(0.9, 1.1)
        return actual_throughput

# 主函数
if __name__ == '__main__':
    total_gpus = 32
    gpus_per_node = 8
    episodes = 10

    high_level_agent = HighLevelAgent(total_gpus, gpus_per_node)
    high_level_agent.train(episodes)
