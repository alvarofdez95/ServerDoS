# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:00:42 2025

@author: jafernandez
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixingNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, mixing_embed_dim):
        """
        Inicializa la mixing network.
        
        Args:
            state_dim (int): Dimensión del estado global.
            n_agents (int): Número de agentes.
            mixing_embed_dim (int): Dimensión de la capa oculta de la mixing network.
        """
        super(MixingNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim

        # Hiperred para generar los pesos de la primera capa
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)

        # Hiperred para generar los pesos de la segunda capa
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)

        # Hiperred para generar los sesgos de la primera capa
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Hiperred para generar los sesgos de la segunda capa
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Combina los valores Q individuales en un valor Q global.
        
        Args:
            agent_qs (torch.Tensor): Valores Q de los agentes. Shape: (batch_size, n_agents).
            states (torch.Tensor): Estado global. Shape: (batch_size, state_dim).
        
        Returns:
            torch.Tensor: Valor Q global (Q_tot). Shape: (batch_size, 1).
        """
        batch_size = agent_qs.size(0)

        # Generar pesos y sesgos a partir del estado global
        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.mixing_embed_dim)
        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.mixing_embed_dim, 1)
        b1 = self.hyper_b1(states).view(-1, 1, self.mixing_embed_dim)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        # Primera capa de la mixing network
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # Shape: (batch_size, 1, n_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # Shape: (batch_size, 1, mixing_embed_dim)

        # Segunda capa de la mixing network
        y = torch.bmm(hidden, w2) + b2  # Shape: (batch_size, 1, 1)

        return y.view(-1, 1)  # Shape: (batch_size, 1)

