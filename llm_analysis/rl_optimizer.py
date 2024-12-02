import os
import random
import time
import subprocess
from itertools import product
import numpy as np

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha  # 优先级的指数，控制采样概率的程度

    def add(self, error, sample):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = sample
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5  # 避免优先级为0

class HighLevelAgent:
    """
    高层代理(Manager), 负责选择总体的并行策略组合。
    """
    def __init__(self, high_level_actions, total_gpus, buffer_capacity=1000, learning_rate=0.1, discount_factor=0.9):
        self.q_table = {}  # 高层状态-动作值表
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.high_level_actions = high_level_actions
        self.total_gpus = total_gpus
        self.previous_state = None
        self.previous_action = None
        self.reward_cache = {}  # 缓存已经评估过的配置的奖励值

        # 初始化优先经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

    def get_state(self):
        """
        获取高层状态，可以定义为上一个高层动作的组合。
        """
        if self.previous_action is None:
            state = 'initial_state'
        else:
            action = self.previous_action
            state = f"tp_{action['tp']}_dp_{action['dp']}_pp_{action['pp']}"
        return state

    def choose_action(self, state):
        """
        使用上置信界(UCB)策略选择高层动作
        """
        state_actions = self.q_table.get(state, {})
        total_counts = sum([info['count'] for info in state_actions.values()]) + 1

        ucb_values = {}
        for action in self.high_level_actions:
            action_str = str(action)
            q_value = state_actions.get(action_str, {'value': 0, 'count': 0})['value']
            count = state_actions.get(action_str, {'value': 0, 'count': 0})['count']
            # UCB 公式
            ucb_value = q_value + np.sqrt(2 * np.log(total_counts) / (count + 1e-5))
            ucb_values[action_str] = ucb_value

        # 选择具有最大 UCB 值的动作
        max_action_str = max(ucb_values, key=ucb_values.get)
        action = eval(max_action_str)
        return action

    def update_q_value(self, state, action, reward, next_state):
        """
        更新高层Q值，使用优先经验回放
        """
        action_str = str(action)
        state_actions = self.q_table.setdefault(state, {})
        action_info = state_actions.setdefault(action_str, {'value': 0, 'count': 0})

        # 计算 TD 误差
        old_value = action_info['value']
        next_max = 0
        next_state_actions = self.q_table.get(next_state, {})
        if next_state_actions:
            next_max = max([info['value'] for info in next_state_actions.values()])

        td_error = reward + self.discount_factor * next_max - old_value

        # 更新 Q 值
        new_value = old_value + self.learning_rate * td_error
        action_info['value'] = new_value
        action_info['count'] += 1

        # 将经验添加到回放缓冲区
        sample = (state, action, reward, next_state)
        self.replay_buffer.add(abs(td_error), sample)

    def replay_experiences(self, batch_size):
        """
        从经验回放缓冲区采样经验，更新 Q 值
        """
        if len(self.replay_buffer.buffer) < batch_size:
            return

        samples, indices, weights = self.replay_buffer.sample(batch_size)
        errors = []

        for i, (state, action, reward, next_state) in enumerate(samples):
            action_str = str(action)
            state_actions = self.q_table.setdefault(state, {})
            action_info = state_actions.setdefault(action_str, {'value': 0, 'count': 0})

            old_value = action_info['value']
            next_max = 0
            next_state_actions = self.q_table.get(next_state, {})
            if next_state_actions:
                next_max = max([info['value'] for info in next_state_actions.values()])

            td_error = reward + self.discount_factor * next_max - old_value
            errors.append(abs(td_error))

            # 使用重要性采样权重更新 Q 值
            new_value = old_value + self.learning_rate * weights[i] * td_error
            action_info['value'] = new_value

        # 更新优先级
        self.replay_buffer.update_priorities(indices, errors)

    def train(self, episodes, batch_size=32):
        """
        训练高层代理
        """
        for episode in range(episodes):
            print(f"\nEpisode {episode+1}/{episodes} - High Level")
            state = self.get_state()
            action = self.choose_action(state)

            # 创建低层代理，在高层动作的范围内进行训练
            worker = LowLevelAgent(action, self.total_gpus)
            reward = worker.train()

            next_state = self.get_state()
            self.update_q_value(state, action, reward, next_state)

            # 更新上一个动作和状态
            self.previous_action = action
            self.previous_state = next_state

            # 经验回放
            self.replay_experiences(batch_size)

            # 日志记录
            print(f"High Level Selected Action: {action}")
            print(f"High Level Reward: {reward}")

class LowLevelAgent:
    """
    低层代理（Worker），在高层策略确定的范围内，进一步选择具体的配置。
    """
    def __init__(self, high_level_action, total_gpus, buffer_capacity=1000, learning_rate=0.1, discount_factor=0.9):
        self.tp = high_level_action['tp']
        self.dp = high_level_action['dp']
        self.pp = high_level_action['pp']
        self.total_gpus = total_gpus

        self.action_space = self.generate_low_level_actions()
        self.q_table = {}  # 低层状态-动作值表
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.previous_state = None
        self.previous_action = None
        self.reward_cache = {}  # 缓存已经评估过的配置的奖励值

        # 初始化优先经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

    def generate_low_level_actions(self):
        """
        生成低层动作空间，包括sequence_parallel, activation_recompute, micro_batch_size的组合
        """
        sequence_parallel_options = [True, False]
        activation_recompute_options = [True, False]
        micro_batch_sizes = [4, 8, 16, 32]

        action_space = []
        for config in product(sequence_parallel_options, activation_recompute_options, micro_batch_sizes):
            action = {
                'sequence_parallel': config[0],
                'activation_recompute': config[1],
                'micro_batch_size': config[2]
            }
            action_space.append(action)
        return action_space

    def get_state(self):
        """
        获取低层状态
        """
        if self.previous_action is None:
            state = 'initial_state'
        else:
            action = self.previous_action
            state = f"seq_{int(action['sequence_parallel'])}_recompute_{int(action['activation_recompute'])}_micro_{action['micro_batch_size']}"
        return state

    def choose_action(self, state):
        """
        使用上置信界（UCB）策略选择低层动作
        """
        state_actions = self.q_table.get(state, {})
        total_counts = sum([info['count'] for info in state_actions.values()]) + 1

        ucb_values = {}
        for action in self.action_space:
            action_str = str(action)
            q_value = state_actions.get(action_str, {'value': 0, 'count': 0})['value']
            count = state_actions.get(action_str, {'value': 0, 'count': 0})['count']
            # UCB 公式
            ucb_value = q_value + np.sqrt(2 * np.log(total_counts) / (count + 1e-5))
            ucb_values[action_str] = ucb_value

        # 选择具有最大 UCB 值的动作
        max_action_str = max(ucb_values, key=ucb_values.get)
        action = eval(max_action_str)
        return action

    def update_q_value(self, state, action, reward, next_state):
        """
        更新低层Q值，使用优先经验回放
        """
        action_str = str(action)
        state_actions = self.q_table.setdefault(state, {})
        action_info = state_actions.setdefault(action_str, {'value': 0, 'count': 0})

        # 计算 TD 误差
        old_value = action_info['value']
        next_max = 0
        next_state_actions = self.q_table.get(next_state, {})
        if next_state_actions:
            next_max = max([info['value'] for info in next_state_actions.values()])

        td_error = reward + self.discount_factor * next_max - old_value

        # 更新 Q 值
        new_value = old_value + self.learning_rate * td_error
        action_info['value'] = new_value
        action_info['count'] += 1

        # 将经验添加到回放缓冲区
        sample = (state, action, reward, next_state)
        self.replay_buffer.add(abs(td_error), sample)

    def replay_experiences(self, batch_size):
        """
        从经验回放缓冲区采样经验，更新 Q 值
        """
        if len(self.replay_buffer.buffer) < batch_size:
            return

        samples, indices, weights = self.replay_buffer.sample(batch_size)
        errors = []

        for i, (state, action, reward, next_state) in enumerate(samples):
            action_str = str(action)
            state_actions = self.q_table.setdefault(state, {})
            action_info = state_actions.setdefault(action_str, {'value': 0, 'count': 0})

            old_value = action_info['value']
            next_max = 0
            next_state_actions = self.q_table.get(next_state, {})
            if next_state_actions:
                next_max = max([info['value'] for info in next_state_actions.values()])

            td_error = reward + self.discount_factor * next_max - old_value
            errors.append(abs(td_error))

            # 使用重要性采样权重更新 Q 值
            new_value = old_value + self.learning_rate * weights[i] * td_error
            action_info['value'] = new_value

        # 更新优先级
        self.replay_buffer.update_priorities(indices, errors)

    def train(self, batch_size=32):
        """
        训练低层代理，返回累计奖励（高层的奖励）
        """
        total_reward = 0
        for episode in range(1):  # 低层可以只运行一次，也可以根据需要调整
            print(f"  Low Level Episode {episode+1}")
            state = self.get_state()
            action = self.choose_action(state)

            # 执行动作，获取奖励
            reward = self.execute_action(action)
            total_reward += reward

            next_state = self.get_state()
            self.update_q_value(state, action, reward, next_state)

            # 经验回放
            self.replay_experiences(batch_size)

            # 更新上一个动作和状态
            self.previous_action = action
            self.previous_state = next_state

            # 日志记录
            print(f"  Low Level Selected Action: {action}")
            print(f"  Low Level Reward: {reward}")

        return total_reward  # 返回累计奖励，供高层代理更新Q值

    def execute_action(self, action):
        """
        根据选择的低层动作执行相应的配置，返回奖励
        """
        # 先检查是否已有缓存的奖励值
        action_str = str(action)
        full_action_str = f"tp_{self.tp}_dp_{self.dp}_pp_{self.pp}_" + action_str
        if full_action_str in self.reward_cache:
            print("  Using cached reward.")
            return self.reward_cache[full_action_str]

        sequence_parallel = action['sequence_parallel']
        activation_recompute = action['activation_recompute']
        micro_batch_size = action['micro_batch_size']

        # 检查总的并行度是否超过GPU总数
        total_parallel_size = self.tp * self.dp * self.pp
        if total_parallel_size > self.total_gpus:
            print("  Invalid configuration: total parallel size exceeds available GPUs.")
            reward = -1000  # 给一个大的惩罚
            self.reward_cache[full_action_str] = reward
            return reward

        # 构建Megatron-LM的运行命令
        command = [
            'python', 'pretrain_gpt.py',
            '--tensor-model-parallel-size', str(self.tp),
            '--pipeline-model-parallel-size', str(self.pp),
            '--num-layers', '24',
            '--hidden-size', '1024',
            '--num-attention-heads', '16',
            '--micro-batch-size', str(micro_batch_size),
            '--global-batch-size', str(micro_batch_size * self.dp),
            '--seq-length', '1024',
            '--max-position-embeddings', '1024',
            '--train-iters', '10',
            '--data-path', 'my_data',
            '--vocab-file', 'vocab.json',
            '--merge-file', 'merges.txt',
            '--save-interval', '10000',
            '--save', 'checkpoints',
            '--log-interval', '1'
        ]

        if self.dp > 1:
            command.extend(['--distributed-backend', 'nccl'])

        if sequence_parallel:
            command.append('--sequence-parallel')

        if activation_recompute:
            command.append('--recompute-activations')

        # 环境变量，设置数据并行大小
        env_vars = os.environ.copy()
        env_vars['WORLD_SIZE'] = str(total_parallel_size)

        # 记录开始时间
        start_time = time.time()

        # 执行训练命令
        try:
            result = subprocess.run(command, env=env_vars, capture_output=True, text=True, check=True)
            # 可以在这里解析result.stdout或result.stderr获取更多信息
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  Command failed with return code {e.returncode}")
            # 如果训练失败，可以给一个大的惩罚
            training_time = 1000  # 设定一个较大的时间表示失败
            reward = -training_time
            self.reward_cache[full_action_str] = reward
            return reward

        # 记录结束时间
        end_time = time.time()

        # 计算训练时间
        training_time = end_time - start_time

        # 奖励可以是负的训练时间，表示越快越好
        reward = -training_time

        # 缓存奖励值
        self.reward_cache[full_action_str] = reward

        return reward

def generate_high_level_actions(total_gpus, gpus_per_node):
    # 计算2的幂次，确保它们小于等于总GPU数和每节点的GPU数
    def powers_of_two(limit):
        p = 1
        powers = []
        while p <= limit:
            powers.append(p)
            p *= 2
        return powers

    # 张量并行配置必须小于等于每个节点的GPU数
    tp_options = powers_of_two(gpus_per_node)

    # 数据并行和流水线并行的选项也是2的幂次，但限制为小于等于总GPU数
    dp_options = powers_of_two(total_gpus)
    pp_options = powers_of_two(total_gpus)

    high_level_actions = []
    for tp, dp, pp in product(tp_options, dp_options, pp_options):
        total_parallel_size = tp * dp * pp
        if total_parallel_size <= total_gpus and tp <= gpus_per_node:
            action = {
                'tp': tp,
                'dp': dp,
                'pp': pp
            }
            high_level_actions.append(action)
    return high_level_actions

if __name__ == '__main__':
    # 假设总共有32个GPU，每个节点有8个GPU
    total_gpus = 32
    gpus_per_node = 8

    # 生成高层动作空间
    high_level_actions = generate_high_level_actions(total_gpus, gpus_per_node)

    # 创建高层代理
    agent = HighLevelAgent(high_level_actions, total_gpus)

    # 开始训练
    agent.train(episodes=10)

    # 打印最终的高层Q表
    print("\nFinal High Level Q-Table:")
    for state, actions in agent.q_table.items():
        print(f"State: {state}")
        for action_str, info in actions.items():
            print(f"  Action: {action_str}, Q-value: {info['value']}")
