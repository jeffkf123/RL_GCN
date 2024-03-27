
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from collections import namedtuple, deque
import random
import numpy as np

class GCN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)  # num_features: number of features per node
        self.conv2 = GCNConv(16, num_actions)   # num_actions: number of possible actions

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        # Second GCN layer (outputs Q-values)
        x = self.conv2(x, edge_index)

        return x

class DQN_GCN(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DQN_GCN, self).__init__()
        self.gcn = GCN(num_features, num_actions)  # GCN defined earlier

    def forward(self, data):
        return self.gcn(data)



class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN_Agent:
    def __init__(self, num_features, num_actions, device='cpu'):
        self.policy_net = DQN_GCN(num_features, num_actions).to(device)
        self.target_net = DQN_GCN(num_features, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.device = device
        self.num_actions = num_actions
        self.TARGET_UPDATE = 16

        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        
        self.epsilon_decay = 100
        self.step_count = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.step_count / self.epsilon_decay)
        self.step_count += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                # Flatten the Q-values and select the index with the highest value
                max_value, max_index = q_values.view(-1).max(0)
                # The action is then the server corresponding to this maximum Q-value
                action = max_index // q_values.size(1)  # Determine the server index from the flattened index
                return torch.tensor([[action]], device=self.device, dtype=torch.long)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)


    def optimize_model(self, batch_size, gamma=0.999):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Create masks for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

        # Process states and actions
        state_batch = [s for s in batch.state]
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = []
        for state in state_batch:
            state = state.to(self.device)
            q_values = self.policy_net(state)
            state_action_values.append(q_values)
        state_action_values = torch.cat(state_action_values).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_states = [s for s in batch.next_state if s is not None]

        # We need to handle graph data structure here. Processing each next state individually.
# Assuming next_states is a list of tensors for the next states that are not None
        for idx, next_state in enumerate(next_states):
            next_state = next_state.to(self.device)
            next_q_values = self.target_net(next_state)
            max_next_q_values = next_q_values.max(1)[0].detach()  # This should be a tensor of [num_nodes]
            
            # If you are getting a tensor of shape [num_nodes] but need to assign a scalar,
            # you might want to take the max again or use a specific index.
            # For example, taking the max across all nodes (assuming that's your intention):
            max_value = max_next_q_values.max().item()  # This ensures you get a scalar value
            if non_final_mask[idx]:
                next_state_values[idx] = max_value


        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)  # Update epsilon

    def update_target_net(self):
        if self.step_count % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())