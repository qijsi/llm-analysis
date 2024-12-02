import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import time

llm_analysis_path = '/home/zhaijidong/qi/llm-analysis'
if llm_analysis_path not in sys.path:
    sys.path.append(llm_analysis_path)

from llm_analysis import analysis
from llm_analysis.config import get_model_config_by_name, get_gpu_config_by_name
import subprocess
import pandas as pd
from Environment import TrainingEnvironment

# Define High-Level (Parallelism) Agent
class HighLevelAgent:
    def __init__(self, total_gpus, gpus_per_node):
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.parallel_actions = self.generate_parallel_actions()
        self.policy_network = PolicyNetwork(len(self.parallel_actions))

    def generate_parallel_actions(self):
        tp_options = [2 ** i for i in range(int(np.log2(self.gpus_per_node)) + 1)]
        dp_options = [2 ** i for i in range(int(np.log2(self.total_gpus)) + 1)]
        pp_options = [2 ** i for i in range(int(np.log2(self.total_gpus)) + 1)]
        actions = []
        for tp, dp, pp in product(tp_options, dp_options, pp_options):
            total_parallel_size = tp * dp * pp
            if total_parallel_size == self.total_gpus:
                action = {'tp': tp, 'dp': dp, 'pp': pp}
                actions.append(action)
        return actions

    def select_parallel_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_logits = self.policy_network(state_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_idx = torch.distributions.Categorical(action_probs).sample().item()
        return self.parallel_actions[action_idx]

# Define Low-Level (Micro-batch and Activation Recompute) Agent
class LowLevelAgent:
    def __init__(self, global_batch, ar_range):
        self.global_batch = global_batch
        self.ar_range = ar_range
        self.micro_batch_sizes = []

    def generate_micro_batch_sizes(self, dp_size):
        max_mb = self.global_batch // dp_size
        self.micro_batch_sizes = [d for d in range(1, max_mb + 1) if max_mb % d == 0]

    def select_micro_batch_action(self):
        ar = random.choice(range(self.ar_range))
        mb = random.choice(self.micro_batch_sizes)
        return {'ar': ar, 'mb': mb}

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Define the two-level DPO Agent
class TwoLevelDPOAgent:
    def __init__(self, total_gpus, gpus_per_node, ar_range, global_batch, env, learning_rate=1e-3):
        self.high_level_agent = HighLevelAgent(total_gpus, gpus_per_node)
        self.low_level_agent = LowLevelAgent(global_batch, ar_range)
        self.env = env
        self.learning_rate = learning_rate

    def train(self, num_iterations=1000):
        for iteration in range(num_iterations):
            # Step 1: Select high-level action for parallelism
            parallel_action = self.high_level_agent.select_parallel_action([0, 0])  # Example state [0, 0]

            # Step 2: Based on high-level action, generate low-level actions
            dp_size = parallel_action['dp']
            self.low_level_agent.generate_micro_batch_sizes(dp_size)

            # Step 3: Select low-level action for micro-batch and activation recomputation
            micro_batch_action = self.low_level_agent.select_micro_batch_action()

            # Combine actions and evaluate
            full_action =ry_utilization, reward = self.env.predict(full_action)

            print(f"Iteration {iteration}, Reward: {reward}, Latency: {latency}, Memory Util: {memory_utilization}")

    def select_best_configuration(self):
        best_score = float('-inf')
        best_configuration = None

        for parallel_action in self.high_level_agent.parallel_actions:
            dp_size = parallel_action['dp']
            self.low_level_agent.generate_micro_batch_sizes(dp_size)
            for ar in range(self.low_level_agent.ar_range):
                for mb in self.low_level_agent.micro_batch_sizes:
                    config = {**parallel_action, 'ar': ar, 'mb': mb}
                    latency, memory_utilization, reward = self.env.predict(config)
                    if reward > best_score:
                        best_score = reward
                        best_configuration = config

        return best_configuration

# Main execution
if __name__ == '__main__':
    total_gpus = 32
    gpus_per_node = 8
    global_batch = 1024
    ar_range = 2
    model_name = "decapoda-research_llama-13b-hf"
    gpu_type = "a100-sxm-40gb"
    seq_len = 1024
    train_program = '../pretrain_llama.py'

    env = TrainingEnvironment(train_program, model_name, gpu_type, total_gpus, seq_len, global_batch, actual_run_frequency=0.2)
    agent = TwoLevelDPOAgent(total_gpus, gpus_per_node, ar_range, global_batch, env, learning_rate=1e-3)

    agent.train(num_iterations=100)
    best_configuration = agent.select_best_configuration()
    print("Best Configuration Found:")
    print(best_configuration)
