# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:37:58 2023

@author: jafernandez
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQConvNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQConvNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        #conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        
        return actions
    
    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        T.save(self.state_dict(), checkpoint_file)
    
    def load_model(self, name):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.load_state_dict(T.load(checkpoint_file))


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, hidden_layer_dim, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_dims[0], hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    
    def forward(self, state):

        fc1 = F.relu(self.fc1(state))
        actions = self.fc2(fc1)
        
        return actions
    
    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        state_dict = T.load(os.path.join(self.checkpoint_dir, name))
        fc1_weights = state_dict['fc1.weight']
        fc1_layer_dim = fc1_weights.size()[0]
        inputs = fc1_weights.size()[1]
        fc2_weights = state_dict['fc2.weight']
        actions = fc2_weights.size()[0]
        self.fc1 = nn.Linear(inputs, fc1_layer_dim)
        self.fc2 = nn.Linear(fc1_layer_dim, actions)
        
        
        self.load_state_dict(state_dict)



class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear (512, n_actions)
        
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        #conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        V = self.V(flat1)
        A = self.A(flat1)
        
        return V, A
    
    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        T.save(self.state_dict(), checkpoint_file)
    
    def load_model(self, name):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.load_state_dict(T.load(checkpoint_file))
        
    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        state_dict = T.load(os.path.join(self.checkpoint_dir, name))
        hidden_weights = state_dict['fc1.weight']
        hidden_layer_dim = hidden_weights.size()[0]
        self.fc1 = nn.Linear(self.input_dims[0], hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, self.n_actions)
        
        
        self.load_state_dict(T.load(os.path.join(self.checkpoint_dir, name)))
        
class RNNNet(nn.Module):
    """
    Red neuronal recurrente
    """
    def __init__(self, input_shape, args):
        super(RNNNet, self).__init__()
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class RNNNetwork(nn.Module):
    def __init__(self, lr, n_actions, hidden_layer_dim, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_layer_dim = hidden_layer_dim
        self.fc1 = nn.Linear(input_dims[0], hidden_layer_dim)
        self.rnn = nn.GRUCell(hidden_layer_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, hidden_state):

        fc1 = F.relu(self.fc1(state))
        h_in = hidden_state.reshape(-1, self.hidden_layer_dim)
        h = self.rnn(fc1, h_in)
        q = self.fc2(h)

        return q, h
    
    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        state_dict = T.load(os.path.join(self.checkpoint_dir, name))
        fc1_weights = state_dict['fc1.weight']
        fc1_layer_dim = fc1_weights.size()[0]
        inputs = fc1_weights.size()[1]
        fc2_weights = state_dict['fc2.weight']
        actions = fc2_weights.size()[0]
        self.fc1 = nn.Linear(inputs, fc1_layer_dim)
        self.fc2 = nn.Linear(fc1_layer_dim, actions)
        
        
        self.load_state_dict(state_dict)

class MixingNetwork(nn.Module):
    def __init__(self, args):
        super(MixingNetwork, self).__init__()
        self.args = args
        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.mixing_embed_dim) # Pesos para la primera capa
        self.hyper_w2 = nn.Linear(args.state_shape, args.mixing_embed_dim) # Pesos para la segunda capa
        self.hyper_b1 = nn.Linear(args.state_shape, args.mixing_embed_dim)  # Bias para la primera capa
        self.hyper_b2 = nn.Linear(args.state_shape, 1)  # Bias para la salida final

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.args.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.args.n_agents)
        w1 = T.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.args.n_agents, self.args.mixing_embed_dim)
        w2 = T.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.args.mixing_embed_dim, 1)
        hidden = F.elu(T.bmm(agent_qs, w1))
        y = T.bmm(hidden, w2)
        q_total = y.view(bs, -1, 1)
        return q_total