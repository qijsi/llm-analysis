import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from collections import deque

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
    
import matplotlib.pyplot as plt

llm_analysis_path = '/home/zhaijidong/qi/AIPerf-LLM/llm-analysis'

if llm_analysis_path not in sys.path:
	sys.path.append(llm_analysis_path)

from llm_analysis import analysis
from llm_analysis.config import get_model_configs, get_gpu_configs, get_dtype_configs
import subprocess

class PerformanceModel:
    def __init__(self, gpu_type, gpu_num, model_name, seq_len, global_batch):
        self.performance_model = analysis
        self.global_batch = global_batch
        self.gpu_type = gpu_type
        self.gpu_num = gpu_num
        self.model_name = model_name
        self.seq_len = seq_len
        self.max_memory_capacity = self.get_max_memory_capacity()
        self.max_memory_bandwidth = self.get_max_memory_bandwidth()

        self.config_histroy = {}
        self.prediction_errors = {}

        self.linkage_matrix = None
        self.num_clusters = 10
        self.cluster_labels = {}
        self.config_features = {}
        self.config_keys = []

        self.cluster_error = {}

    def get_max_memory_capacity(self):
        return 80 * 1024 #FIXME
    
    def get_max_memory_bandwidth(self):
        return 1.6 * 1024 #FIXME

    def predict(self, config):
        dp = config['dp']
        gas = max(1, int(self.global_batch / (dp*config['mb'])))
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
        
        predicted_latency = result.get('latency_per_iter')
        mem_consum_ratio = result.get('max_mem_consum_ratio')
        compute_ratio = result.get('compute_ratio')
        mem_ratio = result.get('mem_ratio')
        comm_ratio = result.get('comm_ratio')

        config_key = self.get_config_key(config)
        cluster_label = self.get_cluster_label(config_key)
        uncertainty = self.cluster_error.get(cluster_label, 1.0)

        #uncertaintity = self.estimate_uncertainty(config, predicted_latency)
        return predicted_latency, mem_consum_ratio, compute_ratio, mem_ratio, comm_ratio, uncertainty
    
    def estimate_uncertainty(self, config, predicted_latency):
        config_key = self.get_config_key(config)
        if config_key in self.config_history:
            return 0.0
        else:
            if not self.linkage_matrix:
                return 1.0
            else:
                if config_key not in self.cluster_labels:
                    label = self.assign_cluster(config)
                else:
                    label = self.cluster_labels[config_key]

                cluster_keys = [k for k, lbl in self.cluster_labels.items() if lbl == label]
                if not cluster_keys:
                    return 1.0
                errors = [self.prediction_errors[k] for k in cluster_keys if k in self.prediction_errors]
                if errors:
                    avg_error = np.mean(errors)
                    uncertainty = min(avg_error / predicted_latency, 1.0)
                    return uncertainty
                else:
                    return 1.0
        
    def run_actual(self, config):

        #TODO: run actual training
        actual_latency = 0
        actual_latency, actual_mem_consum_ratio, actual_compute_ratio, actual_mem_ratio, actual_comm_ratio = 0, 0, 0, 0, 0
        config_key = self.get_config_key(config)
        self.config_histroy[config_key] = actual_latency

        predicted_latency, mem_consum_ratio, compute_ratio, mem_ratio, comm_ratio, uncertainty = self.predict(config)

        latency_error = abs(predicted_latency - actual_latency) / actual_latency
        self.prediction_errors[config_key] = latency_error

        compute_ratio = abs(compute_ratio - actual_compute_ratio) / actual_compute_ratio
        mem_ratio = abs(mem_ratio - actual_mem_ratio) / actual_mem_ratio
        comm_ratio = abs(comm_ratio - actual_comm_ratio) / actual_comm_ratio
        mem_consum_ratio = abs(mem_consum_ratio - actual_mem_consum_ratio) / actual_mem_consum_ratio

        cluster_label = self.get_cluster_label(config_key)
        if cluster_label in self.cluster_error:
            self.cluster_error[cluster_label].append(latency_error)
        else:
            self.cluster_error[cluster_label] = [latency_error]

        self.cluster_error[cluster_label] = [np.mean(self.cluster_error[cluster_label])]

        return actual_latency, actual_mem_consum_ratio, actual_compute_ratio, actual_mem_ratio, actual_comm_ratio
    
    def get_config_key(self, config):
        return (config['tp'], config['dp'], config['pp'], config['ar'], config['mb'])
    
    def parse_config_key(self, key):
        return list(key)
    
    def hierarchical_clustering(self, config_keys, config_features, num_clusters):
        Z = linkage(config_features, method='ward')
        cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')
        self.cluster_labels = {key:label for key, label in zip(config_keys, cluster_labels)}
        
        for label in range(1, num_clusters + 1):
            errors = [self.prediction_errors[key] for key, lbl in self.cluster_labels.items() if lbl == label and key in self.prediction_errors]
            if errors:
                self.cluster_error[label] = [np.mean(errors)]
            else:
                self.cluster_error[label] = [1.0]

    def get_cluster_label(self, config_key):
        return self.cluster_labels.get(config_key, -1) # -1 indicates unknown cluster
    
    def update_model(self):
        #TODO: update model
        pass 

class TrainingEnvironment:
    def __init__(self, model_name, gpu_type, gpu_num, seq_len, global_batch):
        self.model_configs = get_model_configs
        self.platform_configs = get_gpu_configs
        self.dtype_configs = get_dtype_configs
        self.current_state = self.init_state()
        #self.previous_state = []
        self.pm = PerformanceModel(gpu_type, gpu_num, model_name, seq_len, global_batch)
        self.actual_run_frequency = 0.2
        self.history = {}
        self.low_level_reward = 0 # previous reward of low agent

    def reset(self):
        self.low_level_reward = 0
        return self.init_state()

    def init_state(self):
        compute_time_ratio = 0.0
        memory_time_ratio = 0.0
        communication_time_ratio = 0.0
        gpu_memory_utilization = 0.0

        #FIXME performance model cannot provide these values, should be set by real run
        #gpu_utilization = 0.0
        #intra_bandwidth_utilization = 0.0
        #inter_bandwidth_utilization = 0.0
        return np.array([compute_time_ratio, memory_time_ratio, communication_time_ratio, gpu_memory_utilization])

    def step(self, high_action, low_action):
        config = self.action_to_config(high_action, low_action)
        new_state = []
        predicted_latency_iter, predicted_mem_consum_ratio, predicted_compute_ratio, predicted_mem_ratio, predicted_comm_ratio = self.pm.train(config)

        perform_actual_run = False
        
        #print("predicted latency: ", predicted_latency_iter)
        if predicted_mem_consum_ratio < 1:
            reward = - predicted_latency_iter
        else:
            reward = -100.0 #FIXME
        new_state.extend([predicted_compute_ratio, predicted_mem_ratio, predicted_comm_ratio, predicted_mem_consum_ratio])
            
        self.low_level_reward = reward
        
        return new_state, reward

    def action_to_config(self, high_action, low_action):
        return {
            'tp': high_action['tp'],
            'dp': high_action['dp'],
            'pp': high_action['pp'],
            'ar': low_action['ar'],
            'mb': low_action['mb']
        }

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
        self.actor = nn.Linear(hidden_size, action_size) # 策略网络
        self.critic = nn.Linear(hidden_size, 1)          # 值网络

    def forward(self, x):
        x = self.fc(x)
        action_logits = self.actor(x)
        #action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic(x)
        return action_logits, state_value

class HighLevelAgent:
    def __init__(self, total_gpus, gpus_per_node, model_name, gpu_type, ar_range, seq_len, global_batch, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.ar_range = ar_range
        self.global_batch = global_batch
        self.tp_options = [2**i for i in range(int(np.log2(gpus_per_node)) + 1)]
        self.dp_options = [2**i for i in range(int(np.log2(total_gpus)) + 1)]
        self.pp_options = [2**i for i in range(int(np.log2(total_gpus)) + 1)]
        self.high_level_actions = self.generate_high_level_actions()
        self.action_size = len(self.high_level_actions)
        #self.action_size = 3
        self.state_size = 4
        self.env = TrainingEnvironment(model_name, gpu_type, total_gpus, seq_len, global_batch)

        # Actor-Critic 网络和优化器
        self.actor_critic = ActorCritic(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        self.replay_buffer = deque(maxlen=1000)
        
        self.low_level_state_size = self.state_size + 3
        print("len current env: {} high level action len: {}".format(self.state_size, self.action_size))
        self.low_level_agent = LowLevelAgent(self.low_level_state_size, ar_range, global_batch, learning_rate, gamma)

    def generate_high_level_actions(self):
        high_level_actions = []
        for tp, dp, pp in product(self.tp_options, self.dp_options, self.pp_options):
            total_parallel_size = tp * dp * pp
            if total_parallel_size == self.total_gpus:
                action = {'tp': tp, 'dp': dp, 'pp': pp}
                high_level_actions.append(action)
        return high_level_actions

    def select_action(self, state):
        print("state: ", state)
        #state_ = [float(value) for key, value in state]
        #print("state_: ", state_)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        action_logits, state_value = self.actor_critic(state_tensor.unsqueeze(0))
        action_probs = torch.softmax(action_logits, dim=-1)
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        self.saved_values.append(state_value)
        action = self.high_level_actions[action_idx.item()]
        return action

    def update(self):
        if len(self.rewards) == 0:
            return

        policy_losses = []
        value_losses = []

        for reward, log_prob, value in zip(self.rewards, self.saved_log_probs, self.saved_values):
            advantage = reward - value.item()
            policy_losses.append(-log_prob*advantage)
            target = torch.tensor([reward], dtype=torch.float32)
            value_losses.append(nn.functional.mse_loss(value.squeeze(), target))
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.saved_values[:]
        

    def train(self, episodes, episode_rewards):
        max_steps_per_episode = 32  # Or any appropriate number
        for episode in range(episodes):
            state = self.env.reset() #FIXME: why state needs to be reset

            self.rewards = []
            self.saved_log_probs = []
            self.saved_values = []
            
            done = False
            step = 0
            while not done and step < max_steps_per_episode:
                #low_level_reward = self.env.low_level_reward
                #state_with_history = np.append(state, low_level_reward)
                
                high_action = self.select_action(self.env.current_state)
                
                self.low_level_agent.low_level_actions = self.low_level_agent.generate_low_level_actions(ar_range, high_action['dp'])
                
                print(f"\nEpisode {episode + 1}")
                print(f"High Level Action: {high_action}")
                
                low_level_steps = 0
                max_low_level_steps = 10
                total_reward = 0 
                self.low_level_agent.rewards = []
                self.low_level_agent.saved_log_probs = []
                self.low_level_agent.saved_values = []

                while low_level_steps < max_low_level_steps:
                    print(f"state: {self.env.current_state}, high_action: {high_action}")
                    low_action= self.low_level_agent.select_action(self.env.current_state, high_action)
                    print(f"step: {low_level_steps}, Low Level Action: {low_action}")
                    new_state, reward = self.env.step(high_action, low_action)
                    print(f"next state: {new_state}, reward: {reward}")

                    # 更新低层代理
                    #print("update low level agent")
                    self.low_level_agent.rewards.append(reward)
                    total_reward += reward
                    state = new_state
                    self.env.current_state = new_state
                    low_level_steps += 1

                self.low_level_agent.update()
                average_reward = total_reward / max_low_level_steps
                self.rewards.append(average_reward)
                episode_rewards.append(average_reward)
                # 存储高层代理的奖励（可以将低层代理的奖励也纳入）
                #self.rewards.append(best_reward)
                self.update()               
                step += 1
            
class LowLevelAgent:
    def __init__(self, state_size, ar_range, global_batch, learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.learning_rate = learning_rate
        # Define the action space for the low-level agent based on the action received from the high-level agent
        self.state_size = state_size  # Similar state dimensions as the high-level agent for simplicity
        self.global_batch = global_batch
    
        self.max_action_size = ar_range * global_batch #FIXME
        self.ar_range = ar_range

        self.actor_critic = ActorCritic(self.state_size, self.max_action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        self.action_mask = torch.ones(self.max_action_size, dtype=torch.float32)
        self.low_level_actions = [None] * self.max_action_size
        # 经验数据库，存储已评估的配置及其实际运行结果
        self.evaluated_configs = {}  # key: config tuple, value: actual throughput

        # 初始实际运行频率
        self.actual_run_frequency = 0.5  # 可以动态调整
        
    def generate_low_level_actions(self, ar_range, dp_size):
        activation_recompute_options = list(range(ar_range))
        
        assert (
                self.global_batch % dp_size == 0
            ), f"global_batch_size ({self.global_batch}) must be divisible by dp_size ({dp_size})"

        max_mb = self.global_batch // dp_size 
        micro_batch_size = [d for d in range(1, max_mb+1) if max_mb % d == 0]
        gas = [int(self.global_batch/(x*dp_size)) for x in micro_batch_size ]    
                
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
            
        print("low level actions: ", len(low_level_actions))
        return self.low_level_actions
    
    def select_action(self, state, high_action):
        high_action_features = np.array(list(high_action.values()))
        state_with_high_action = np.concatenate((state, high_action_features))
        state_tensor = torch.tensor(state_with_high_action, dtype=torch.float32)
        
        action_logits, state_value = self.actor_critic(state_tensor.unsqueeze(0))
        
        masked_action_logits = action_logits.clone()
        masked_action_logits[0, self.action_mask == 0] = -float('inf')
        action_probs = torch.softmax(masked_action_logits, dim=-1)
        
        m = torch.distributions.Categorical(action_probs)
        action_idx = m.sample()
        
        print("action_idx: ", action_idx)
        self.saved_log_probs.append(m.log_prob(action_idx))
        self.saved_values.append(state_value)
        
        action = self.low_level_actions[action_idx.item()]
        return action
    
    def update(self):
        R = 0
        policy_losses = []
        value_losses = []
        for reward, log_prob, value in zip(self.rewards, self.saved_log_probs, self.saved_values):
            print("reward: {}, value: {}".format(reward, value.item()))
            R = reward
            advantage = R - value.item()
            policy_losses.append(-log_prob*advantage)
            value_losses.append(nn.functional.mse_loss(value.squeeze(), torch.tensor(R)))
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.saved_values[:]

if __name__ == '__main__':
    total_gpus = 32
    gpus_per_node = 8
    episodes = 16
    global_batch = 1024
    ar_range = 2
    model_name="decapoda-research_llama-13b-hf"
    gpu_type="a100-sxm-80gb"
    seq_len=1024

    high_level_agent = HighLevelAgent(total_gpus, gpus_per_node, model_name, gpu_type, ar_range, seq_len, global_batch)
    episode_rewards = []
    high_level_agent.train(episodes, episode_rewards)

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Reward over Episodes')
    #plt.show()
    plt.savefig('training_reward.png')

    # Save the model parameters
    torch.save({
        'high_level_agent_state_dict': high_level_agent.actor_critic.state_dict(),
        'low_level_agent_state_dict': high_level_agent.low_level_agent.actor_critic.state_dict(),
        'optimizer_state_dict': high_level_agent.optimizer.state_dict(),
        'low_level_optimizer_state_dict': high_level_agent.low_level_agent.optimizer.state_dict(),
    }, 'agent_checkpoint.pth')