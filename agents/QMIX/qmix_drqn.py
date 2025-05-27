# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:51:52 2025

@author: jafernandez
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DRQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, chkpt_dir='tmp/dqn'):
        """
        Initializes the Deep Recurrent Q-Network (DRQN) for a single agent.

        Args:
            input_dim (int): Dimension of the input (observations).
            action_dim (int): Number of possible actions.
            hidden_dim (int): Number of hidden units in the GRU.
        
        IMPORTANTE: EL AGENTE RECIBE LA OBSERVACIÓN Y LA ACCIÓN EJECUTADA EN FORMATO ONE-HOT
        """
        super(DRQN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.checkpoint_dir = chkpt_dir

        # Capa fully connected para proyectar la observación
        self.fc = nn.Linear(input_dim + output_dim, hidden_dim)

        # Capa recurrente (GRU)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # Capa fully connected final para generar los valores Q
        self.q_network = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs, hidden_state):
        """
        Paso hacia adelante del DRQN.
        
        Args:
            obs (torch.Tensor): Observación actual. Shape: (batch_size, input_dim+output_dim).
            hidden_state (torch.Tensor): Estado oculto anterior. Shape: (batch_size, hidden_dim).
        
        Returns:
            torch.Tensor: Valores Q para cada acción. Shape: (batch_size, output_dim).
            torch.Tensor: Nuevo estado oculto. Shape: (batch_size, hidden_dim).
        """
        # Proyectar la observación en el espacio oculto
        x = F.relu(self.fc(obs))

        # Actualizar el estado oculto usando la RNN
        h = self.rnn(x, hidden_state)

        # Calcular los valores Q
        q_values = self.q_network(h)

        return q_values, h

    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden state of the GRU to zeros.

        Args:
            batch_size (int): Number of sequences in the batch.

        Returns:
            hidden_state (torch.Tensor): Zero-initialized hidden state, shape (1, batch_size, hidden_dim).
        """
        return torch.zeros(batch_size, self.hidden_dim)
    
    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        torch.save(self.state_dict(), checkpoint_file)
    
        
    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        state_dict = torch.load(os.path.join(self.checkpoint_dir, name))
        self.load_state_dict(state_dict)