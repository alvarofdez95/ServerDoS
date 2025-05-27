# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:15:32 2025

@author: jafernandez
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vicomrl.agents.QMIX.qmix_mixing_network import MixingNetwork
from vicomrl.agents.QMIX.qmix_replay_buffer import ReplayBuffer
from vicomrl.agents.QMIX.qmix_drqn import DRQN

class QMIX:
    def __init__(self, obs_dim, state_dim, n_actions, n_agents, VDN, hidden_dim, mixing_embed_dim, buffer_capacity, lr, gamma, chkpt_dir):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.VDN = VDN
        self.hidden_dim = hidden_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
 
        # Redes
        self.agents = nn.ModuleList([DRQN(obs_dim, n_actions, hidden_dim, chkpt_dir).to(self.device) for _ in range(n_agents)])
        self.target_agents = nn.ModuleList([DRQN(obs_dim, n_actions, hidden_dim, chkpt_dir).to(self.device) for _ in range(n_agents)])
        if self.VDN == False: 
            self.mixing_network = MixingNetwork(state_dim, n_agents, mixing_embed_dim).to(self.device)
            self.target_mixing_network = MixingNetwork(state_dim, n_agents, mixing_embed_dim).to(self.device)

        # Optimizador
        self.optimizer = torch.optim.Adam(list(self.agents.parameters()) + list(self.mixing_network.parameters()), lr=lr, weight_decay=1e-4)

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Inicializar redes objetivo
        self.update_target_networks()

    def update_target_networks(self):
        """Actualiza las redes objetivo con los parámetros de las redes principales."""
        for target_agent, agent in zip(self.target_agents, self.agents):
            target_agent.load_state_dict(agent.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def select_actions(self, obs, action, hidden_states, epsilon):
        """Selecciona acciones usando ε-greedy."""
        actions_one_hot = np.eye(self.n_actions)[action]
        total_obs = np.concatenate((obs,actions_one_hot), axis=1)
        total_obs = torch.FloatTensor(total_obs).to(self.device)
        
        actions = []
        new_hidden_states = []
        for i, agent in enumerate(self.agents):
            q_values, new_h = agent(total_obs[i].unsqueeze(0), hidden_states[i].to(self.device))
            if np.random.rand() < epsilon:  # Exploración
                action = np.random.randint(self.n_actions)
            else:  # Explotación
                action = q_values.argmax(dim=-1).item()
            actions.append(action)
            new_hidden_states.append(new_h)
        #return np.array(actions), torch.stack(new_hidden_states, dim=1)
        return np.array(actions), new_hidden_states

    def train(self, batch_size):
        """Entrena el modelo usando un lote de experiencias."""
        if len(self.buffer) < batch_size:
            return

        # Muestrear un lote del buffer
        obs, actions, rewards, states, next_obs, next_states, terminated, truncated = self.buffer.sample(batch_size)
        
        actions_one_hot = np.eye(self.n_actions)[actions]
        total_obs = np.concatenate((obs,actions_one_hot), axis=2)
        total_obs_next = np.concatenate((next_obs,actions_one_hot), axis=2)
        # Convertir a tensores de PyTorch
        total_obs = torch.FloatTensor(total_obs).to(self.device)
        total_obs_next = torch.FloatTensor(total_obs_next).to(self.device)
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        states = torch.FloatTensor(states).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        terminated = torch.FloatTensor(terminated).to(self.device)
        truncated = torch.FloatTensor(truncated).to(self.device)

        # Calcular Q-values actuales
        q_values = []
        for i, agent in enumerate(self.agents):
            q, _ = agent(total_obs[:, i], self.agents[i].init_hidden(batch_size).to(self.device))
            q_values.append(q.gather(1, actions[:, i].unsqueeze(1)))
        q_values = torch.stack(q_values, dim=1)

        # Calcular Q_tot
        q_tot = self.mixing_network(q_values, states) #shape: torch.Size([batch_size, 1])
        
        

        # Calcular Q-values objetivo
        with torch.no_grad():
            next_q_values = []
            for i, target_agent in enumerate(self.target_agents):
                next_q, _ = target_agent(total_obs_next[:, i], target_agent.init_hidden(batch_size).to(self.device))
                next_q_values.append(next_q.max(dim=1, keepdim=True).values)
            next_q_values = torch.stack(next_q_values, dim=1)
            next_q_tot = self.target_mixing_network(next_q_values, next_states)
            
            
            target_q_tot = rewards[:, 0].unsqueeze(1) + self.gamma * next_q_tot * (1 - terminated[:,0].unsqueeze(1)) * (1 - truncated[:,0].unsqueeze(1))


            #if random.random() < 0.01: print("target_q_tot: ", torch.mean(target_q_tot))

        # Calcular la pérdida
        loss = F.mse_loss(q_tot, target_q_tot)

        # Actualizar parámetros
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_experience(self, obs, actions, reward, state, next_obs, next_state, terminated, truncated):
        """Guarda una experiencia en el replay buffer."""
        self.buffer.push(obs, actions, reward, state, next_obs, next_state, terminated, truncated)
    
