import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from llm_analysis import analysis

class PerformanceModel:
    def __init__(self):
        self.performance_model = analysis

    def train(self, config):
        # 使用 llm_analysis 模块进行性能预测
        result = analysis.train(
            model_name=config['model'],
            total_num_tokens=config['total_num_tokens'],
            gpu_name=config['gpu_name'],
            flops_efficiency=config['flops_efficiency'],
            hbm_memory_efficiency=config['hbm_efficiency'],
            activation_recomputation=config['activation_recompute'],
            tp_size=config['tp'],
            sp_size=config['sp'],
            pp_size=config['pp'],
            ds_zero=config['ds_zero'],
            batch_size_per_gpu=config['micro_batch'],
            global_batch_size=config['global_batch_size'],
            total_num_gpus=config['gpus'],
            seq_len=config['seq_len'],
            output_dir=config['output_dir']
        )
        return result.get('latency_per_iter'), result.get('estimated_peak_memory_per_gpu'), result.get('memory_left')

class TrainingEnvironment:
    def __init__(self):
        self.current_state = initial_state
        self.performance_model = PerformanceModel
        self.actual_run_frequency = 0.2
        self.history = {}

    def initial_state(self):
        gpu_utilization = 0.5
        compute_time_ratio = 0.5
        communication_time_ratio = 0.5
        memory_usage = 0.5
        return np.array([gpu_utilization, compute_time_ratio, communication_time_ratio, memory_usage])
    
    def update_state(self, result):
        return np.array([result['compute_ratio'], result['memory_ratio'], result['comm_ratio'], result['peak_mem']])
    
    def need_for_actual_run(self, predicted_throughput, config):
        if random.random() < self.actual_run_frequency:
            return True
        
        err_threshold = np.mean(self.history) if self.history else 0
        if err_threshold > 10: #FIXME
            return True
        
        if self.compute_config_impact(config) > 0.5:
            return True
        
        return False
    
    def compute_config_impact(self, config):
        return config['batch_size'] / max(config.values())
            

    def step(self, action):
        config = self.action_to_config(action)
        predicted_throughput, predicted_peak_memory, predicted_latency = self.performance_model(config)

        if self.need_for_actual_run(predicted_throughput, config):
            actual_throughput, actual_peak_memory, actual_latency = self.run_actual_training(config)
            reward = actual_throughput
            state['compute_ratio'] = actual_latency['compute']
            state['mem_ratio'] = actual_latency['memory']
            state['comm_ratio'] = actual_latency['comm']
            state['peak_mem'] = actual_peak_memory
            self.history.append(abs(predicted_throughput - actual_throughput))
        else:
            reward = predicted_throughput
            state['compute_ratio'] = predicted_latency['compute']
            state['mem_ratio'] = predicted_latency['memory']
            state['comm_ratio'] = predicted_latency['comm']
            state['peak_mem'] = predicted_peak_memory

        return state, reward

    def action_to_config(self, action):
        # 将动作转换为配置
        return {
            'tp': action['tp'],
            'dp': action['dp'],
            'pp': action['pp'],
            'activation_recompute': action['activation_recompute'],
            'grad_accum_steps': action['grad_accum_steps'],
            'batch_size': action['batch_size']
        }

    def run_actual_training(self, config):
        # 模拟实际训练的函数，返回实际吞吐量
        actual_throughput = config['tp'] * config['dp'] * config['pp'] * 10  # 简化的吞吐量计算
        return actual_throughput
    
class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []

    def store(self, state, action, reward, log_prob, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []

# 定义 Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)  # 策略网络
        self.critic = nn.Linear(hidden_size, 1)          # 值网络

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class Agent:
    def __init__(self, total_gpus, gpus_per_node, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.learning_rate = learning_rate

        # 定义动作空间
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.tp_options = [2**i for i in range(int(np.log2(gpus_per_node)) + 1)]
        self.dp_options = [2**i for i in range(int(np.log2(total_gpus)) + 1)]
        self.pp_options = [2**i for i in range(int(np.log2(total_gpus)) + 1)]

        self.high_level_actions = self.generate_high_level_actions()
        self.action_size = len(self.high_level_actions)
        self.state_size = 4  # 状态维度

        # Actor-Critic 网络和优化器
        self.actor_critic = ActorCritic(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

        self.low_level_agent = LowLevelAgent(self.state_size + self.action_size, learning_rate, gamma)

    def generate_high_level_actions(self):
        high_level_actions = []
        for tp, dp, pp in product(self.tp_options, self.dp_options, self.pp_options):
            total_parallel_size = tp * dp * pp
            if total_parallel_size <= self.total_gpus:
                action = {'tp': tp, 'dp': dp, 'pp': pp}
                high_level_actions.append(action)
        return high_level_actions

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state_tensor.unsqueeze(0))
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()

        self.saved_log_probs.append(m.log_prob(action_idx))
        self.saved_values.append(state_value)

        action = self.high_level_actions[action_idx.item()]
        return action

    def update(self):
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

       for reward, log_prob, value in zip(self.rewards, self.saved_log_probs, self.saved_values):
           R = reward
           advantage = R - value.item()
           
           policy_losses.append(-log_prob*advantage)
           value_losses.append(nn.functional.mse_loss(value.squeeze(), torch.tensor([R])))
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_log_probsp[:]
        del self.saved_values[:]

    def train(self, environment, episodes):
        for episode in range(episodes):
            state = environment.reset()
            done = False
            while not done:
                # 高层代理选择动作
                high_action = self.select_action(state)
                print(f"\nEpisode {episode + 1}")
                print(f"High Level Action: {high_action}")

                low_action= self.low_level_agent.select_action(state, high_action)
                print(f"Low Level Action: {low_action}")
                next_state, reward, done = environment.step(low_action)

                # 存储高层代理的奖励（可以将低层代理的奖励也纳入）
                self.rewards.append(reward)

                # 更新低层代理
                self.low_level_agent.rewards.append(reward)
                self.low_level_agent.update()

                state = next_state

            # 更新高层代理
            self.update()
            
class LowLevelAgent:
    def __init__(self, learning_rate, gamma):
        self.gamma = gamma
        self.learning_rate = learning_rate
        # Define the action space for the low-level agent based on the action received from the high-level agent
        self.state_size = 4  # Similar state dimensions as the high-level agent for simplicity

        self.actor_critic = ActorCritic(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        # 经验数据库，存储已评估的配置及其实际运行结果
        self.evaluated_configs = {}  # key: config tuple, value: actual throughput

        # 初始实际运行频率
        self.actual_run_frequency = 0.5  # 可以动态调整

    def execute(self, high_level_action):
        # Execute the action determined by the high-level agent and return the observed reward and the action taken
        # For simplicity, let's say we just simulate this process
        simulated_reward = random.random()  # Simulated reward
        low_level_action = None  # Determined action
        return simulated_reward, low_level_action
    
    def get_batch_size_and_gradient_accumulation(
        self,
        batch_size_per_gpu: int = None,
        gradient_accumulation_steps: int = None,
        global_batch_size: int = None,
    ) -> tuple:
        """Configure batch_size_per_gpu, gradient_accumulation_steps and
        global_batch_size (effective batch size). If none is given, find a
        maximum batch_size_per_gpu while satisfying the constraint `global_batch_size ==
        batch_size_per_gpu * gradient_accumulation_steps * dp_size`.

        Args:
            max_batch_size_per_gpu (int): the max batch size per gpu before OOM
            batch_size_per_gpu (int, optional): batch size per GPU. Defaults to None.
            gradient_accumulation_steps (int, optional): gradient accumulation steps. Defaults to None.
            global_batch_size (int, optional): global batch size (effective batch size). Defaults to None.

        Returns:
            tuple: (batch_size_per_gpu, gradient_accumulation_steps, global_batch_size)
        """
        assert_msg = (f"note that global_batch_size == batch_size_per_gpu *"
                      f" gradient_accumulation_steps * dp_size")
        #dp_size = self.parallelism_config.dp_size * self.parallelism_config.rdp_size
        dp_size = self.parallelism_config.dp_size
        if (global_batch_size and batch_size_per_gpu
                and gradient_accumulation_steps):
            assert (global_batch_size == batch_size_per_gpu *
                    gradient_accumulation_steps * dp_size), assert_msg
        elif global_batch_size and batch_size_per_gpu:
            # gradient_accumulation_steps is None, the other two are not None
            gradient_accumulation_steps = global_batch_size // (
                batch_size_per_gpu * dp_size)
            assert (global_batch_size % (batch_size_per_gpu * dp_size) == 0
                    and gradient_accumulation_steps > 0
                    ), "no valid gradient_accumulation_steps, {assert_msg}"
        elif global_batch_size and gradient_accumulation_steps:
            # batch_size_per_gpu is None, the other two are not None
            batch_size_per_gpu = global_batch_size // (
                gradient_accumulation_steps * dp_size)
            assert (global_batch_size %
                    (gradient_accumulation_steps * dp_size) == 0
                    and batch_size_per_gpu > 0
                    ), "no valid batch_size_per_gpu, {assert_msg}"
        elif batch_size_per_gpu and gradient_accumulation_steps or batch_size_per_gpu:
            # batch_size_per_gpu is not None
            if batch_size_per_gpu > max_batch_size_per_gpu:
                logger.warning(
                    f"batch_size_per_gpu {batch_size_per_gpu} must be <= max_batch_size_per_gpu {max_batch_size_per_gpu}, {assert_msg}"
                )
            if gradient_accumulation_steps is None:
                gradient_accumulation_steps = 1
            global_batch_size = (batch_size_per_gpu *
                                 gradient_accumulation_steps * dp_size)
        elif global_batch_size:
            # batch_size_per_gpu and gradient_accumulation_steps are None
            assert (
                global_batch_size % dp_size == 0
            ), f"global_batch_size must be divisible by dp_size, {assert_msg}"

            if max_batch_size_per_gpu >= global_batch_size // dp_size:
                batch_size_per_gpu = global_batch_size // dp_size
                gradient_accumulation_steps = 1
            else:
                prod = global_batch_size // dp_size
                batch_size_per_gpu = next(d for d in range(
                    prod,
                    0,
                    -1,
                ) if prod % d == 0 and d <= max_batch_size_per_gpu)
                gradient_accumulation_steps = global_batch_size // (
                    batch_size_per_gpu * dp_size)
            logger.info("batch_size_per_gpu not set, using batch_size_per_gpu"
                        f" {batch_size_per_gpu} (max_batch_size_per_gpu ="
                        f" {max_batch_size_per_gpu})")
        else:
            # (global_batch_size and batch_size_per_gpu are None) or (all are None)
            batch_size_per_gpu = max_batch_size_per_gpu
            gradient_accumulation_steps = (1 if
                                           gradient_accumulation_steps is None
                                           else gradient_accumulation_steps)
            global_batch_size = (batch_size_per_gpu *
                                 gradient_accumulation_steps * dp_size)
            logger.info("batch_size_per_gpu not set, using batch_size_per_gpu"
                        f" {batch_size_per_gpu} (max_batch_size_per_gpu ="
                        f" {max_batch_size_per_gpu})")
        return (
            batch_size_per_gpu,
            gradient_accumulation_steps,
            global_batch_size,
        )

    def select_action(self, state):
        state_with_high_action = np.concatenate((state, self.high_level_action))
        state_tensor = torch.tensor(state_with_high_action, dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state_tensor)
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()

        self.saved_log_probs.append(m.log_prob(action_idx))
        self.saved_values.append(state_value)

        action = self.low_level_actions[action_idx.item()]
        return action

    def compute_config_difference(self, config):
        """
        计算新配置与已评估配置之间的最小差异
        """
        min_diff = float('inf')
        for evaluated_config in self.evaluated_configs.keys():
            diff = sum([abs(config[k] - evaluated_config[k]) if isinstance(config[k], int) else config[k] != evaluated_config[k] for k in config])
            if diff < min_diff:
                min_diff = diff
        return min_diff

    def compute_config_impact(self, config):
        """
        计算配置对性能的影响程度（简单示例）
        """
        # 假设 batch_size 对性能影响较大
        impact = config['batch_size'] * 0.5 + config['grad_accum_steps'] * 0.3 + (1 if config['activation_recompute'] else 0) * 0.2
        return impact

    def adjust_actual_run_frequency(self):
        """
        根据模型性能动态调整实际运行频率
        """
        # 这里可以根据模型的预测误差、收敛情况等调整
        # 简单示例：如果最近的模型预测误差较大，则增加实际运行频率
        recent_errors = [abs(self.rewards[i] - self.saved_values[i].item()) for i in range(len(self.rewards))]
        avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0
        if avg_error > threshold:
            self.actual_run_frequency = min(self.actual_run_frequency + 0.1, 1.0)
        else:
            self.actual_run_frequency = max(self.actual_run_frequency - 0.1, 0.1)

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
            'batch_size': action['batch_size'],
            # 添加其他必要的参数
            'model': 'your_model_name',
            'total_num_tokens': 1000000000,
            'gpu_name': 'V100',
            'flops_efficiency': 0.5,
            'hbm_efficiency': 0.8,
            'sp': 1,  # 假设 sp 为 1
            'ds_zero': False,
            'micro_batch': action['batch_size'],
            'global_batch_size': action['batch_size'] * self.dp,
            'gpus': self.tp * self.dp * self.pp,
            'seq_len': 1024,
            'output_dir': './output'
        }

        # 配置的唯一标识，用于经验数据库
        config_key = tuple(config.items())

        # 使用性能模型预测吞吐量
        predicted_latency, predicted_peak_mem, predicted_left_mem = performance_model(config)
        predicted_throughput = self.dp * 1 / predicted_latency  # 简单假设吞吐量与延迟的倒数成正比

        # 判断是否需要实际运行
        need_actual_run = False

        # 方法11：基于差异更新
        if self.compute_config_difference(config) > difference_threshold:
            need_actual_run = True

        # 方法12：优先验证高影响配置
        if self.compute_config_impact(config) > impact_threshold:
            need_actual_run = True

        # 方法6：自适应采样策略
        if random.random() < self.actual_run_frequency:
            need_actual_run = True

        # 方法10：代理辅助优化
        # 如果预测吞吐量高于当前最优值，进行实际运行验证
        if predicted_throughput > self.best_predicted_throughput:
            need_actual_run = True

        # 执行实际运行或使用预测值
        if need_actual_run:
            actual_throughput = self.run_actual_training(config)
            reward = actual_throughput

            # 更新经验数据库
            self.evaluated_configs[config_key] = actual_throughput

            # 更新性能模型（可选，根据实际运行结果调整模型）
            # ...

            # 更新当前最优预测吞吐量
            if predicted_throughput > self.best_predicted_throughput:
                self.best_predicted_throughput = predicted_throughput

            # 动态调整实际运行频率
            self.adjust_actual_run_frequency()
        else:
            reward = predicted_throughput

        self.rewards.append(reward)
        print(f"  Low Level Reward (Throughput): {reward}")

        # 更新策略网络
        self.update()

        return reward

    def run_actual_training(self, config):
        """
        实际运行单层模型训练，返回实际吞吐量。
        """
        print("  Running actual training for calibration...")
        # 实际运行代码，根据需要修改
        latency, peak_mem, left_mem = performance_model(config)
        actual_throughput = 1 / latency * random.uniform(0.9, 1.1)  # 模拟实际吞吐量有一定波动
        return actual_throughput

    def get_state(self):
        # 定义低层状态，这里简化为全零向量
        return [0.0, 0.0, 0.0]

# 主函数
if __name__ == '__main__':
    total_gpus = 32
    gpus_per_node = 8
    episodes = 10
    global_batch = 1024

    high_level_agent = HighLevelAgent(total_gpus, gpus_per_node)
    high_level_agent.train(episodes)
