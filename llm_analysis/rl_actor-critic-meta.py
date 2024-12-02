import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from collections import deque

import matplotlib.pyplot as plt
import time

# Install the higher library for MAML
# Ensure that 'higher' is installed in your environment
# !pip install higher
import higher
#!pip install SALib
# Adjust the import paths as needed
llm_analysis_path = '/home/zhaijidong/qi/AIPerf-LLM/llm-analysis'

if llm_analysis_path not in sys.path:
    sys.path.append(llm_analysis_path)

from llm_analysis import analysis
from llm_analysis.config import get_model_config_by_name, get_gpu_config_by_name
import subprocess

# Import Sobol Sensitivity Analysis

from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd

# Define the PerformanceModel class
class PerformanceModel:
    def __init__(self, gpu_type, gpu_num, train_program, model_name, seq_len, global_batch, layers_to_run=None):
        self.performance_model = analysis
        self.global_batch = global_batch
        self.gpu_type = gpu_type
        self.gpu_num = gpu_num
        self.model_name = model_name
        self.seq_len = seq_len
        self.layers_to_run = layers_to_run  # Number of layers to run (partial execution)
        self.program = train_program
        self.model_configs = get_model_config_by_name(model_name)
        self.platform_configs = get_gpu_config_by_name(gpu_type)

    def predict(self, config):
        # Use the simulator to predict performance metrics
        dp = config['dp']
        gas = max(1, int(self.global_batch / (dp * config['mb'])))
        print("Using simulator to predict performance.")
        print("gradient_accumulation_steps: ", gas)
        result = analysis.train(
            model_name=self.model_name,
            total_num_tokens=51200,
            gpu_name=self.gpu_type,
            flops_efficiency=0.5,
            hbm_memory_efficiency=0.9,
            tp_size=config['tp'],
            sp_size=1,
            pp_size=config['pp'],
            activation_recomputation=config['ar'],
            batch_size_per_gpu=config['mb'],
            ds_zero=0,
            global_batch_size=self.global_batch,
            total_num_gpus=self.gpu_num,
            gradient_accumulation_steps=gas,
            seq_len=self.seq_len,
            output_dir='./output'
        )

        return (
            result.get('latency_per_iter', 0.0),
            result.get('max_mem_consum_ratio', 0.0),
            result.get('compute_ratio', 0.0),
            result.get('mem_ratio', 0.0),
            result.get('comm_ratio', 0.0),
        )

    def actual_run(self, config):
        # Run the actual training to get real performance metrics
        dp = config['dp']
        gas = max(1, int(self.global_batch / (dp * config['mb'])))

        command = [
            #'python', 'pretrain_gpt.py',
            'python', self.program,
            '--tensor-model-parallel-size', str(config['tp']),
            '--pipeline-model-parallel-size', str(config['pp']),
            '--num-layers', str(self.model_configs.num_layers),
            '--hidden-size', str(self.model_configs.hidden_dim),
            '--num-attention-heads', str(self.model_configs.num_key_value_heads),
            '--micro-batch-size', str(config['mb']),
            '--recompute-activations', str(config['ar']),
            '--global-batch-size', str(self.global_batch),
            '--seq-length', str(self.seq_len),
            '--max-position-embeddings', '4096',
            '--train-iters', '1',
            '--data-path', 'my_data',
            '--vocab-file', 'vocab.json',
            '--merge-file', 'merges.txt',
            '--save-interval', '10000',
            '--save', 'checkpoints',
            '--log-interval', '1'
        ]

        command = [arg for arg in command if arg != '']

        if dp > 1:
            command.extend(['--distributed-backend', 'nccl'])

        #if sequence_parallel:
        command.append('--sequence-parallel')

        env_vars = os.environ.copy()
        #env_vars['WORLD_SIZE'] = str(total_parallel_size)

        start_time = time.time()

        try:
            result = subprocess.run(command, env=env_vars, capture_output=True, text=True, check=True)
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  Command failed with return code {e.returncode}")
            training_time = 100 
            reward = -training_time
            return reward

        end_time = time.time()

        training_time = end_time - start_time

        reward = -training_time 

        #TODO: max_mem_consum_ratio, compute_ratio, mem_ratio, comm_ratio       

        return (
            latency_per_iter,
            max_mem_consum_ratio,
            compute_ratio,
            mem_ratio,
            comm_ratio
        )

# Define the TrainingEnvironment class
class TrainingEnvironment:
    def __init__(self, train_program, model_name, gpu_type, gpu_num, seq_len, global_batch, actual_run_frequency=0.2):
        #self.model_configs = get_model_config_by_name(model_name)
        #self.platform_configs = get_gpu_config_by_name(gpu_type)
        self.current_state = self.init_state()
        self.pm = PerformanceModel(gpu_type, gpu_num, train_program, model_name, seq_len, global_batch)
        self.actual_run_frequency = actual_run_frequency  # Frequency of actual runs
        self.history = {}
        self.low_level_reward = 0  # Previous reward of low-level agent
        self.performance_dataset = pd.DataFrame()
        self.sensitivity = None  # Sensitivity analysis results

    def reset(self):
        self.low_level_reward = 0
        self.current_state = self.init_state()
        return self.current_state

    def init_state(self):
        # Initialize the state variables
        latency = 0.0
        memory_utilization = 0.0
        return np.array([latency, memory_utilization])

    def step(self, high_action, low_action):
        config = self.action_to_config(high_action, low_action)
        new_state = []

        # Decide whether to use simulator or actual run based on frequency
        if random.random() < self.actual_run_frequency:
            # Perform actual run
            (latency_per_iter,
             max_mem_consum_ratio,
             compute_ratio,
             mem_ratio,
             comm_ratio) = (1, 1, 1, 1, 1)
            #self.pm.actual_run(config)
            print("Actual run performed.")
        else:
            # Use simulator prediction
            (latency_per_iter,
             max_mem_consum_ratio,
             compute_ratio,
             mem_ratio,
             comm_ratio) = self.pm.predict(config)
            print("Simulator prediction used.")

        # Compute reward
        if max_mem_consum_ratio < 1:
            reward = -latency_per_iter  # Negative latency as reward (minimize latency)
        else:
            reward = -100.0  # Penalty for exceeding memory capacity

        new_state.extend([
            latency_per_iter,
            max_mem_consum_ratio
        ])

        self.low_level_reward = reward
        self.current_state = np.array(new_state)

        # Collect performance data for sensitivity analysis
        performance_data = {
            'tp': config['tp'],
            'dp': config['dp'],
            'pp': config['pp'],
            'ar': config['ar'],
            'mb': config['mb'],
            'latency': latency_per_iter,
            'memory_util': max_mem_consum_ratio
        }
        
        self.performance_dataset = pd.concat([self.performance_dataset, pd.DataFrame([performance_data])], ignore_index=True)

        # Perform sensitivity analysis periodically
        if len(self.performance_dataset) % 50 == 0:
            #self.sensitivity = self.calculate_sensitivity()
            print("Sensitivity analysis updated.")

        return self.current_state, reward

    def action_to_config(self, high_action, low_action):
        # Convert actions to configuration parameters
        return {
            'tp': high_action['tp'],
            'dp': high_action['dp'],
            'pp': high_action['pp'],
            'ar': low_action['ar'],
            'mb': low_action['mb']
        }
"""
    def calculate_sensitivity(self):
        # Define the problem for Sobol analysis
        if len(self.performance_dataset) < 100:
            print("Insufficient data for sensitivity analysis.")
            return None

        problem = {
            'num_vars': 5,
            'names': ['tp', 'dp', 'pp', 'ar', 'mb'],
            'bounds': [[self.performance_dataset['tp'].min(), self.performance_dataset['tp'].max()],
                       [self.performance_dataset['dp'].min(), self.performance_dataset['dp'].max()],
                       [self.performance_dataset['pp'].min(), self.performance_dataset['pp'].max()],
                       [self.performance_dataset['ar'].min(), self.performance_dataset['ar'].max()],
                       [self.performance_dataset['mb'].min(), self.performance_dataset['mb'].max()]]
        }

        # Perform Sobol sensitivity analysis
        Y = self.performance_dataset['latency'].values
        X = self.performance_dataset[['tp', 'dp', 'pp', 'ar', 'mb']].values

        try:
            Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
            sensitivity = {name: Si['S1'][i] for i, name in enumerate(problem['names'])}
            return sensitivity
        except ZeroDivisionError:
            print("Sensitivity analysis failed due to insufficient variance.")
            return None
"""
# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)  # Policy network
        self.critic = nn.Linear(hidden_size, 1)  # Value network

    def forward(self, x):
        x = self.fc(x)
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

# Define the HighLevelAgent
class HighLevelAgent:
    def __init__(self, total_gpus, gpus_per_node, ar_range,
                 seq_len, global_batch, learning_rate=1e-3, gamma=0.99, actual_run_frequency=0.2):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.ar_range = ar_range
        self.global_batch = global_batch
        self.tp_options = [2 ** i for i in range(int(np.log2(gpus_per_node)) + 1)]
        self.dp_options = [2 ** i for i in range(int(np.log2(total_gpus)) + 1)]
        self.pp_options = [2 ** i for i in range(int(np.log2(total_gpus)) + 1)]
        self.high_level_actions = self.generate_high_level_actions()
        self.action_size = len(self.high_level_actions)
        self.state_size = 2  # memory_utilization and latency
        #self.env = TrainingEnvironment(model_name, gpu_type, total_gpus, seq_len, global_batch, actual_run_frequency)

        # Actor-Critic network and optimizer
        self.actor_critic = ActorCritic(self.state_size, self.action_size)
        self.meta_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.low_level_state_size = self.state_size + 3  # State size plus high-level action features
        self.low_level_agent = LowLevelAgent(self.low_level_state_size, ar_range, global_batch, learning_rate, gamma)

    def generate_high_level_actions(self):
        # Generate possible high-level actions based on parallelism options
        high_level_actions = []
        for tp, dp, pp in product(self.tp_options, self.dp_options, self.pp_options):
            total_parallel_size = tp * dp * pp
            if total_parallel_size == self.total_gpus:
                action = {'tp': tp, 'dp': dp, 'pp': pp}
                high_level_actions.append(action)
        return high_level_actions

    def select_action(self, state, fnet=None):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if fnet:
            # For meta-learning with higher library
            action_logits, state_value = fnet(state_tensor)
        else:
            action_logits, state_value = self.actor_critic(state_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Adjust action probabilities based on sensitivity
        #if self.env.sensitivity is not None:
        #    action_sensitivities = []
        #    for action in self.high_level_actions:
        #        sensitivity_score = 0
        #        for param in ['tp', 'dp', 'pp']:
        #            param_sensitivity = self.env.sensitivity.get(param, 0)
        #            sensitivity_score += param_sensitivity * action[param]
        #        action_sensitivities.append(sensitivity_score)
        #    action_sensitivities = torch.tensor(action_sensitivities, dtype=torch.float32)
        #    action_probs *= torch.exp(-action_sensitivities)
        #    action_probs /= action_probs.sum()

        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        value = state_value.squeeze()

        action = self.high_level_actions[action_idx.item()]
        return action, log_prob, value

    def train(self, episodes, episode_rewards, env):
        max_steps_per_episode = 32

        for episode in range(episodes):
            state = env.reset()

            with higher.innerloop_ctx(self.actor_critic, self.meta_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                episode_log_probs = []
                episode_values = []
                episode_rewards_local = []

                for step in range(max_steps_per_episode):
                    # High-level action selection
                    high_action, log_prob, value = self.select_action(state, fnet)
                    episode_log_probs.append(log_prob)
                    episode_values.append(value)

                    # Low-level agent generates actions based on high-level action
                    self.low_level_agent.generate_low_level_actions(self.ar_range, high_action['dp'])
                    #low_action = self.low_level_agent.select_action(state, high_action, self.env.sensitivity)
                    low_action = self.low_level_agent.select_action(state, high_action)

                    # Environment step
                    new_state, reward = env.step(high_action, low_action)

                    # Low-level agent updates
                    self.low_level_agent.update(reward)

                    # Collect rewards
                    episode_rewards_local.append(reward)

                    # Inner loop optimization for meta-learning
                    loss = -log_prob * (reward - value.item()) + nn.functional.mse_loss(value, torch.tensor([reward]))
                    diffopt.step(loss)

                    state = new_state

                # Meta-update
                #self.meta_optimizer.zero_grad()
                #cumulative_reward = sum(episode_rewards_local)
                #meta_loss = -cumulative_reward / len(episode_rewards_local)
                #meta_loss.backward()
                #self.meta_optimizer.step()

                #episode_rewards.append(cumulative_reward / len(episode_rewards_local))
                #print(f"Episode {episode + 1}: Average Reward = {episode_rewards[-1]:.4f}")

    def adapt_to_actual_system(self, env):
        # Few-shot adaptation to the actual system
        # Collect data from actual runs
        actual_run_configs = random.sample(self.high_level_actions, k=5)  # Collect data from 5 actual runs
        for config in actual_run_configs:
            low_action = {
                'ar': random.choice(range(self.ar_range)),
                'mb': random.choice([d for d in range(1, self.global_batch // config['dp'] + 1) if (self.global_batch // config['dp']) % d == 0])
            }
            env.step(config, low_action)

        # Fine-tune the meta-parameters using actual system data
        #self.train(episodes=1, episode_rewards, env)  # Fine-tune for one episode

# Define the LowLevelAgent
class LowLevelAgent:
    def __init__(self, state_size, ar_range, global_batch, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.global_batch = global_batch
        self.ar_range = ar_range

        # Max action size can be adjusted as needed
        self.max_action_size = 100
        self.actor_critic = ActorCritic(self.state_size, self.max_action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.action_mask = torch.ones(self.max_action_size, dtype=torch.float32)
        self.low_level_actions = [None] * self.max_action_size

    def generate_low_level_actions(self, ar_range, dp_size):
        activation_recompute_options = list(range(ar_range))
        assert self.global_batch % dp_size == 0, f"global_batch_size ({self.global_batch}) must be divisible by dp_size ({dp_size})"
        max_mb = self.global_batch // dp_size
        micro_batch_size = [d for d in range(1, max_mb + 1) if max_mb % d == 0]

        low_level_actions = []
        for ar, mb in product(activation_recompute_options, micro_batch_size):
            action = {'ar': ar, 'mb': mb}
            low_level_actions.append(action)

        num_valid_actions = len(low_level_actions)
        self.action_mask = torch.zeros(self.max_action_size, dtype=torch.float32)
        self.action_mask[:num_valid_actions] = 1

        if num_valid_actions < self.max_action_size:
            padding_actions = [None] * (self.max_action_size - num_valid_actions)
            self.low_level_actions = low_level_actions + padding_actions
        else:
            self.low_level_actions = low_level_actions[:self.max_action_size]
            self.action_mask = self.action_mask[:self.max_action_size]

        print(f"Low-level actions generated: {num_valid_actions}")

    def select_action(self, state, high_action, sensitivity=None):
        high_action_features = np.array(list(high_action.values()))
        state_with_high_action = np.concatenate((state, high_action_features))
        state_tensor = torch.tensor(state_with_high_action, dtype=torch.float32).unsqueeze(0)

        action_logits, state_value = self.actor_critic(state_tensor)
        masked_action_logits = action_logits.clone()
        masked_action_logits[0, self.action_mask == 0] = -float('inf')

        action_probs = torch.softmax(masked_action_logits, dim=-1)

        # Adjust action probabilities based on sensitivity
        if sensitivity is not None:
            action_sensitivities = []
            for action in self.low_level_actions:
                if action is not None:
                    sensitivity_score = sensitivity.get('ar', 0) * action['ar'] + sensitivity.get('mb', 0) * action['mb']
                else:
                    sensitivity_score = 0
                action_sensitivities.append(sensitivity_score)
            action_sensitivities = torch.tensor(action_sensitivities, dtype=torch.float32)
            action_probs *= torch.exp(-action_sensitivities)
            action_probs /= action_probs.sum()

        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        value = state_value.squeeze()

        action = self.low_level_actions[action_idx.item()]
        self.saved_log_prob = log_prob
        self.saved_value = value
        return action

    def update(self, reward):
        # Update the low-level agent using the received reward
        advantage = reward - self.saved_value.item()
        policy_loss = -self.saved_log_prob * advantage
        value_loss = nn.functional.mse_loss(self.saved_value, torch.tensor([reward], dtype=torch.float32))

        self.optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()

# Main execution
if __name__ == '__main__':
    total_gpus = 32
    gpus_per_node = 8
    episodes = 10  # Adjust as needed
    global_batch = 1024
    ar_range = 2
    model_name = "decapoda-research_llama-13b-hf"
    gpu_type = "a100-sxm-40gb"
    seq_len = 1024
    actual_run_frequency = 0.2  # 20% of the time perform actual runs
    train_program = '../pretrain_llama.py'

    high_level_agent = HighLevelAgent(
        total_gpus, gpus_per_node, ar_range,
        seq_len, global_batch, learning_rate=1e-3, gamma=0.99, actual_run_frequency=actual_run_frequency
    )
    
    env = TrainingEnvironment(train_program, model_name, gpu_type, total_gpus, seq_len, global_batch, actual_run_frequency)

    episode_rewards = []
    high_level_agent.train(episodes, episode_rewards, env)

    # Few-shot adaptation to actual system
    high_level_agent.adapt_to_actual_system(env)

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Reward over Episodes')
    plt.savefig('training_reward.png')

    # Save the model parameters
    torch.save({
        'high_level_agent_state_dict': high_level_agent.actor_critic.state_dict(),
        'low_level_agent_state_dict': high_level_agent.low_level_agent.actor_critic.state_dict(),
        'optimizer_state_dict': high_level_agent.meta_optimizer.state_dict(),
        'low_level_optimizer_state_dict': high_level_agent.low_level_agent.optimizer.state_dict(),
    }, 'agent_checkpoint.pth')