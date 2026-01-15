import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import numpy as np
from collections import deque

from custom_graph import Graph
from stovec import Embed
from gymenv import CustomEnv
from simulator import simulate
from simulator import celf
import torch.nn.utils as nn_utils

# ——— hyperparameters ———
REPLAY_CAPACITY = 64    
GAMMA           = 0.99
LR              = 1e-4
EPSILON         = 0.20
TAU             = 0.005  
epoch_num       = 0


class QNet(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        # initialize to ones instead of random
        self.beta1 = nn.Parameter(torch.ones(embed_dim, 1))
        self.beta2 = nn.Parameter(torch.ones(1))
        self.beta3 = nn.Parameter(torch.ones(1))

        self.fc = nn.Linear(embed_dim * 2, 1)
        self.fc = nn_utils.weight_norm(self.fc)
        nn.init.xavier_normal_(self.fc.weight_v)

    def forward(self, node_embed, agg_embed):
        # L2 normalize each
        agg_n = F.normalize(agg_embed, p=2, dim=1)
        node_n = F.normalize(node_embed, p=2, dim=1)

        scaled_agg  = self.beta2 * agg_n
        # add tiny noise to avoid exact zero
        scaled_node = self.beta3 * node_n + torch.rand(1, device=node_n.device) * 1e-10

        combined = torch.cat((scaled_agg, scaled_node), dim=1)
        return self.fc(F.relu(combined))

class DQNAgent:
    def __init__(self, embed_dim=64):
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim     = embed_dim

        # shared alphas now start at 1
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.alpha4 = nn.Parameter(torch.ones(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        self.q_network = QNet(embed_dim)
        self.q_target  = QNet(embed_dim)
        self.q_target.load_state_dict(self.q_network.state_dict())
        for p in self.q_target.parameters(): p.requires_grad = False

        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + self.shared_alphas,
            lr=LR
        )
        

    def select_action(self, env, valid_nodes, epsilon):
        cur = env.embed.cur_embed
        # use mean instead of sum
        agg = cur.mean(dim=0, keepdim=True)

        #if random.random() < epsilon:
            #return random.choice(valid_nodes)

        v_emb = cur[valid_nodes]
        a_emb = agg.expand(v_emb.size(0), -1)
        qv    = self.q_network(v_emb, a_emb).squeeze(1)
        return valid_nodes[qv.argmax().item()]

    def train(self, batch_size, epoch_num,gamma=GAMMA):
        if len(self.replay_buffer) < batch_size:
            return 0,epoch_num

        batch = random.sample(self.replay_buffer, batch_size)
        losses = torch.zeros([1])

        for state, action, reward, next_state in batch:
            # current Q
            agg_s  = state.cur_embed.mean(0, keepdim=True)
            node_s = state.cur_embed[action].unsqueeze(0)
            q_s    = self.q_network(node_s, agg_s).squeeze(1)

            # next Q via target network
            agg_ns   = next_state.cur_embed.mean(0, keepdim=True)
            valid    = [v for v, lbl in enumerate(next_state.graph.labels) if lbl == 0]
            v_emb_ns = next_state.cur_embed[valid]
            a_emb_ns = agg_ns.expand(len(valid), -1)
            q_next_all = self.q_target(v_emb_ns, a_emb_ns).squeeze(1).detach()
            max_q_next = q_next_all.max() if valid else torch.tensor(0.0, device=q_s.device)

            target = reward + gamma * max_q_next
            target = target.view_as(q_s)
            losses = losses+(F.smooth_l1_loss(q_s, target))

        # loss = losses

        # # self.optimizer.zero_grad()
        # loss = loss.detach_().requires_grad_(True)
        # loss.backward(retain_graph=True)

        if epoch_num == 0:
            losses.backward()
        else:
            losses = losses.detach_().requires_grad_(True)
            losses.backward(retain_graph=True)
        loss = losses

        
        
        # print info of q_network
        print("\nparam_grad:")
        for name, param in self.q_network.named_parameters():
            if param.grad is not None:
                print(f"{name}:")
                print(f"  grad_mean: {param.grad.mean().item():.6f}")
                # print(f"  drad_std: {param.grad.std().item():.6f}")
                # print(f"  grad_max: {param.grad.max().item():.6f}")
                # print(f"  grad_min: {param.grad.min().item():.6f}")
        
        # print info of alpha
        print("\nalpha:")
        for i, alpha in enumerate(self.shared_alphas):
            if alpha.grad is not None:
                print(f"alpha{i+1}:",f"{alpha.item():.6f}")
                print(f"  grad: {alpha.grad.item():.6f}")
        


        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + self.shared_alphas,
            max_norm=10
        )
        self.optimizer.step()

        # — soft‐update target network —
        for src, tgt in zip(self.q_network.parameters(), self.q_target.parameters()):
            tgt.data.mul_(1 - TAU)
            tgt.data.add_(src.data * TAU)
        epoch_num += 1
        
        return loss.detach().numpy(), epoch_num

    def add_experience(self, state, action, reward, next_state):
        state_copy = state.copy_emb()
        next_state_copy = next_state.copy_emb()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))

    def evaluate(self, env, budget):
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.sum(dim=0) 
        #NODE NUMBER MUST START WITH ZERO!
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.q_network(node_embed.unsqueeze(0), agg_embed.unsqueeze(0)).item()
            q_list.append((q_value,v))
        q_list.sort(reverse=True)

        for (q,v) in q_list:
            print(q)
            if budget<=0:
                break
            env.embed.graph.labels[v] = 1
            budget-=1
        
        result = simulate(env.embed.graph,10000)

        end_time = time.time()

        print(f"DQN: {end_time - start_time:.2f} seconds")

        env.reset()

        
        return result
    
    def random_select(self, env, nodes, num_nodes):
        env.embed.update()
        random_numbers = random.sample(range(num_nodes), nodes)
        for i in random_numbers:
            env.embed.graph.labels[i] = 1
        result = simulate(env.embed.graph,1000)
        env.reset()
        return result
    
def train_agent(agent, env, episodes, batch_size,epoch_num):
    """
    Trains the DQN agent by interacting with the CustomEnv.

    Args:
        agent: Instance of DQNAgent.
        env: Instance of CustomEnv.
        episodes: Number of episodes to train.
        batch_size: Batch size for training.
    """

    for episode in range(episodes):
        # Reset the environment to get the initial embeddings
        env.reset()  
        done = False
        episode_reward = 0
        epsilon = 0.20

        loss_list=np.array([])
        # epoch_num = 0
        while not done:
            # Get the valid nodes (not yet selected)
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]

            # Select an action 
            orig_state = env.embed.copy_emb()
            action = agent.select_action(env, valid_nodes, epsilon)

            # Apply the action 
            state, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer
            agent.add_experience(orig_state, action, reward, state)

            # Train the agent
            loss,epoch_num = agent.train(batch_size,epoch_num)
            loss_list = np.append(loss_list,loss)
            episode_reward += reward
            
        epsilon *= 0.90
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {episode_reward}")
        print(f"Total influenced: {env.influence}")
        print(f"Critic loss: {np.mean(loss_list).item():.4f}")

    agent.replay_buffer.clear()
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'shared_alphas_state_dict': {f'alpha{i+1}': alpha for i, alpha in enumerate(agent.shared_alphas)}
    }, './New Graph Dataset/DQN_agent(p2p1_3c2).pth')

    
def DQN_main(num_nodes):
    
    input_file = './New Graph Dataset/subgraph1.txt'
    adj_list = {}

    for i in range(num_nodes+1): adj_list[i]= []
    max_node = 0
    with open(input_file, 'r') as file:
        for line in file:
            u, v, weight = line.strip().split()  # Split into u, v, and weight
            u, v = int(u), int(v)  # Convert node IDs to integers
            weight = float(weight)  # Convert weight to float
            
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append((v, weight))  # Add edge with weight
            max_node = max(max_node, max(u,v))

    
    if max_node<100 :
        return
    
    max_node = max_node
    # Create a Graph object with the adjacency list
    graph = Graph(max_node+1, adj_list)
    

    agent = DQNAgent()

    if os.path.exists('./New Graph Dataset/DQN_agent(p2p1_3c2).pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('./New Graph Dataset/DQN_agent(p2p1_3c2).pth')
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    
        # Restore shared alphas
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
    else:
        print("No pre-trained agent found. Creating a new agent...")
    env = CustomEnv(graph, agent.shared_alphas, 10)
    train_agent(agent, env, 10, 32,epoch_num)    



