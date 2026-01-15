"""
DDPG (Deep Deterministic Policy Gradient) 算法实现
用于解决图上的影响力最大化问题
"""

import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque
import heapq
from custom_graph import Graph
from stovec import Embed
from gymenv import CustomEnv
from simulator import simulate, celf
import torch.nn.utils as nn_utils
import numpy as np

# 超参数设置
REPLAY_CAPACITY = 64    # 经验回放缓冲区大小
GAMMA = 0.99           # 折扣因子
LR_ALPHAS = 0.00005     # 图嵌入更新的学习率
LR_CRITIC = 0.00010     # Critic网络的学习率
LR_ACTOR = 0.00010      # Actor网络的学习率
EPSILON = 0.20         # 探索率
GUMBEL_TAU = 0.75      # Gumbel Softmax的温度参数
TAU = 0.005            # 目标网络软更新系数

class QNet(nn.Module):
    """
    Q网络实现，同时作为Actor和Critic网络
    使用共享参数架构来评估节点价值和选择动作
    """
    def __init__(self, embed_dim=64):
        super(QNet, self).__init__()
        self.embed_dim = embed_dim

        # Critic网络的参数
        self.beta1 = nn.Parameter(torch.ones(embed_dim * 2, 1))
        self.beta2 = nn.Parameter(torch.ones(1))
        self.beta3 = nn.Parameter(torch.ones(1))

        # Actor网络的参数
        self.theta1 = nn.Parameter(torch.ones(embed_dim * 2, 1))
        self.theta2 = nn.Parameter(torch.ones(1))
        self.theta3 = nn.Parameter(torch.ones(1))

        # 最终的全连接层
        self.fc1 = nn.Linear(embed_dim * 2, 1)  # Critic输出层
        self.fc2 = nn.Linear(embed_dim * 2, 1)  # Actor输出层

        # 使用Xavier初始化和权重归一化
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1 = nn_utils.weight_norm(self.fc1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2 = nn_utils.weight_norm(self.fc2)

    def forward(self, node_embed, agg_embed, role='critic'):
        """
        前向传播函数
        Args:
            node_embed: 节点嵌入
            agg_embed: 聚合嵌入
            role: 'critic'或'actor'，决定使用哪组参数
        """
        if role == 'critic':
            # Critic网络的前向传播
            agg_embed=F.normalize(agg_embed, p=2, dim=1)  # L2归一化
            scaled_aggregate = self.beta2 * agg_embed
            node_embed=F.normalize(node_embed, p=2, dim=1)  # L2归一化
            scaled_node = self.beta3 * node_embed + torch.rand(1)*10**(-10)
            combined = torch.cat((scaled_aggregate, scaled_node), dim=1)
            return self.fc1(F.relu(combined))
        else:
            # Actor网络的前向传播
            # scaled_aggregate = self.theta2 * agg_embed
            # scaled_node = self.theta3 * node_embed
            agg_embed=F.normalize(agg_embed, p=2, dim=1)  # L2归一化
            scaled_aggregate =  agg_embed
            node_embed=F.normalize(node_embed, p=2, dim=1)  # L2归一化
            scaled_node = node_embed
            combined = torch.cat((scaled_aggregate, scaled_node), dim=1)
            return self.fc2(F.relu(combined))

class DDPGAgent:
    """
    DDPG智能体实现
    包含Actor-Critic架构和经验回放机制
    """
    def __init__(self, embed_dim=64):
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)  # 经验回放缓冲区
        self.embed_dim = embed_dim

        # 图嵌入更新的共享参数
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.alpha4 = nn.Parameter(torch.ones(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        # 创建Actor和Critic网络
        self.actor = QNet(embed_dim)
        self.critic = QNet(embed_dim)
        
        # 创建目标网络
        self.actor_target = QNet(embed_dim)
        self.critic_target = QNet(embed_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 冻结目标网络的参数
        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        # 优化器设置
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(
            list(self.critic.parameters()) + self.shared_alphas,
            lr=LR_CRITIC
        )

    def select_action(self, env, valid_nodes, epsilon):
        """
        选择动作的函数
        使用epsilon-greedy策略进行探索
        """
        cur = env.embed.cur_embed
        agg = cur.mean(dim=0, keepdim=True)
        #if random.random() < epsilon:
            #return random.choice(valid_nodes)  # 随机探索
        v_emb = cur[valid_nodes]
        a_emb = agg.expand(len(valid_nodes), -1)
        logits = self.actor(v_emb, a_emb, role='actor').squeeze(1)
        logits = logits - logits.max()
        one_hot = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=-1)
        idx = one_hot.argmax().item()
        return valid_nodes[idx]

    def add_experience(self, state, action, reward, next_state):
        """
        将经验添加到回放缓冲区
        """
        self.replay_buffer.append((
            state.copy_emb(),
            action,
            reward,
            next_state.copy_emb()
        ))

    def train(self, batch_size, gamma=GAMMA):
        """
        训练函数
        包含Critic和Actor的更新过程
        """
        if len(self.replay_buffer) < batch_size:
            return 0,0

        # Critic网络更新
        self.opt_critic.zero_grad()
        critic_losses = []
        batch = random.sample(self.replay_buffer, batch_size)
        for s, a, r, ns in batch:
            agg_s = s.cur_embed.mean(0, keepdim=True)
            agg_s = F.normalize(agg_s, p=2, dim=1)  # L2归一化
            
            q_s = self.critic(s.cur_embed[a].unsqueeze(0), agg_s, role='critic')

            agg_ns = ns.cur_embed.mean(0, keepdim=True)
            valid = [i for i, lbl in enumerate(ns.graph.labels) if lbl == 0]
            qn = self.critic_target(
                ns.cur_embed[valid],
                agg_ns.expand(len(valid), -1),
                role='critic'
            ).detach().max()
            target = r + gamma * qn
            critic_losses.append(F.smooth_l1_loss(q_s.squeeze(), target))

        loss_c = torch.stack(critic_losses).mean()
        # print(f"[DEBUG] Critic loss = {loss_c.item():.4f}")
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + self.shared_alphas,
            max_norm=10
        )
        self.opt_critic.step()

        # Actor网络更新
        self.opt_actor.zero_grad()
        actor_losses = []
        for s, a, r, ns in batch:
            agg_s = s.cur_embed.mean(0, keepdim=True)
            valid = [i for i, lbl in enumerate(s.graph.labels) if lbl == 0]
            v_emb = s.cur_embed[valid]
            a_emb = agg_s.expand(len(valid), -1)

            logits = self.actor(v_emb, a_emb, role='actor').squeeze(1)
            logits = logits - logits.max()
            probs = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=-1)
            logp = torch.log(probs + 1e-8)

            agg_ns = ns.cur_embed.mean(0, keepdim=True)
            valid_nxt = [i for i, lbl in enumerate(ns.graph.labels) if lbl == 0]
            qn = self.critic_target(
                ns.cur_embed[valid_nxt].detach(),
                agg_ns.expand(len(valid_nxt), -1),
                role='critic'
            ).detach().max()
            targ = r + gamma * qn

            advs = []
            '''
            for node in valid:
                qv = self.critic(
                    s.cur_embed[node].unsqueeze(0),
                    agg_s,
                    role='critic'
                ).detach()
                advs.append((targ - qv).squeeze())'''
            
            q_vs = self.critic(v_emb, a_emb, role='critic').detach().squeeze(1)                         
            advs = targ - q_vs       
            actor_losses.append(-(logp * advs).mean())

        loss_a = torch.stack(actor_losses).mean()
        # print(f"[DEBUG] Actor loss = {loss_a.item():.4f}")
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=10
        )
        self.opt_actor.step()

        # 目标网络软更新
        for src, tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt.data.mul_(1.0 - TAU)
            tgt.data.add_(src.data * TAU)
        for src, tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt.data.mul_(1.0 - TAU)
            tgt.data.add_(src.data * TAU)
            
        return loss_c.detach().numpy(), loss_a.detach().numpy()

    def evaluate(self, env, budget):
        """
        评估函数
        使用训练好的模型选择种子节点
        """
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.mean(dim=0)
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.critic(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor').item()
            q_list.append((q_value, v))
        q_list.sort(reverse=True)

        for (q, v) in q_list:
            print(q)
            if budget <= 0:
                break
            env.embed.graph.labels[v] = 1
            budget -= 1
        
        result = simulate(env.embed.graph, 10000)

        end_time = time.time()
        print(f"DDPG: {end_time - start_time:.2f} seconds")

        env.reset()
        return result

    def random_select(self, env, nodes, num_nodes):
        """
        随机选择种子节点的基准方法
        """
        env.embed.update()
        random_numbers = random.sample(range(num_nodes), nodes)
        for i in random_numbers:
            env.embed.graph.labels[i] = 1
        result = simulate(env.embed.graph, 1000)
        env.reset()
        return result

def train_agent(agent, env, episodes, batch_size, epsilon):
    """
    训练智能体的主函数
    
    参数:
        agent: DDPGAgent实例
        env: CustomEnv实例
        episodes: 训练轮数
        batch_size: 批次大小
        epsilon: 探索率
    """


    for episode in range(episodes):
        env.reset()
        done = False
        episode_reward = 0
        
        loss_c_list=[]
        loss_a_list=[]
        while not done:
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]
            orig_state = env.embed.copy_emb()
            action = agent.select_action(env, valid_nodes, epsilon)
            state, reward, done, _ = env.step(action)
            agent.add_experience(orig_state, action, reward, state)
            (loss_c, loss_a)=agent.train(batch_size)
            loss_c_list.append(loss_c)
            loss_a_list.append(loss_a)
            episode_reward += reward

        epsilon *= 0.90
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {episode_reward}")
        print(f"Total influenced: {env.influence}")
        print(f"Critic loss: {np.mean(loss_c_list).item():.4f}")
        print(f"Actor loss: {np.mean(loss_a_list).item():.4f}")

    # 保存模型参数
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'shared_alphas_state_dict': {f'alpha{i+1}': alpha for i, alpha in enumerate(agent.shared_alphas)},
    }, 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth')
        
        # Load Q-network (betas and thetas included)

def DDPG_main(num_nodes):
    """
    主函数
    用于初始化和运行DDPG算法
    """
    input_file = 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\subgraph1.txt'
    adj_list = {}

    # 初始化邻接表
    for i in range(num_nodes + 1):
        adj_list[i] = []
    max_node = 0

    # 读取图数据
    with open(input_file, 'r') as file:
        for line in file:
            u, v, weight = line.strip().split()
            u, v = int(u), int(v)
            weight = float(weight)
            adj_list[u].append((v, weight))
            max_node = max(max_node, max(u, v))

    if max_node < 100:
        return

    # 创建图对象和智能体
    graph = Graph(max_node + 1, adj_list)
    agent = DDPGAgent()
    epsilon = 0.30

    # 加载预训练模型（如果存在）
    if os.path.exists('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
    else:
        print("No pre-trained agent found. Creating a new agent...")

    # 创建环境并训练智能体
    budget=10
    env = CustomEnv(graph, agent.shared_alphas, budget)
    train_agent(agent, env, 10, 32, epsilon)