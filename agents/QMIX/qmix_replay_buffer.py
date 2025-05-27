# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:40:29 2025

@author: jafernandez
"""

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity, prioritized=False, alpha=0.6, epsilon=1e-6, initial_priority=1.0):
        
        "capacity: máxima capacidad del buffer"
        self.capacity = capacity
        self.prioritized = prioritized
        if self.prioritized:
            self.alpha = alpha
            self.epsilon = epsilon
            self.initial_priority = initial_priority
            self.buffer = deque(maxlen=capacity)
            self.priorities = deque(maxlen=capacity)
            self.max_priority = initial_priority
        else:
            self.buffer = deque(maxlen=capacity)
        
        

    def push(self, obs, actions, reward, state, next_obs, next_state, terminated, truncated):
        
        "Guarda una experiencia en el buffer. Si el buffer está al máximo de su capacidad, se elimina el elemento más antiguo"
        
        if self.prioritized:
            if len(self.buffer) == 0:
                priority = self.initial_priority
            else:
                priority = self.max_priority
            self.priorities.append(priority)
            self.buffer.append((obs, actions, reward, state, next_obs, next_state, terminated, truncated))
            if priority > self.max_priority:
                self.max_priority = priority
        else:
            self.buffer.append((obs, actions, reward, state, next_obs, next_state, terminated, truncated))

    def sample(self, batch_size, beta=0.4):
        
        "Devuelve del buffer de experiencias un número de muestras (batch_size) en formato numpy"
        "Ejemplo: obs_batch --> np.array(batch_size, n_agents, obs_dim)"
        
        if self.prioritized:
            if len(self.buffer) == 0:
                raise ValueError("Buffer is empty")
            
            buffer_list = list(self.buffer)
            priorities = np.array(self.priorities, dtype=np.float32)
            
            # Calcula probabilidades de muestreo
            probabilities = (priorities + self.epsilon) ** self.alpha
            probabilities /= probabilities.sum()
            
            # Muestrea índices
            indices = np.random.choice(len(buffer_list), size=batch_size, p=probabilities)
            
            # Obtiene las experiencias muestreadas
            samples = [buffer_list[i] for i in indices]
            obs_batch, actions_batch, reward_batch, state_batch, next_obs_batch, next_state_batch, terminated_batch, truncated_batch = zip(*samples)
            
            # Calcula pesos de importancia (IS)
            weights = (1.0 / (len(self.buffer) * (priorities[indices] + self.epsilon))) ** beta
            weights /= weights.max()  # Normaliza los pesos
            
            return (
                np.array(obs_batch),
                np.array(actions_batch),
                np.array(reward_batch),
                np.array(state_batch),
                np.array(next_obs_batch),
                np.array(next_state_batch),
                np.array(terminated_batch),
                np.array(truncated_batch),
                indices,
                weights,
            )
        else:
            samples = random.sample(self.buffer, batch_size)
            obs_batch, actions_batch, reward_batch, state_batch, next_obs_batch, next_state_batch, terminated_batch, truncated_batch = zip(*samples)
            return (
                np.array(obs_batch),
                np.array(actions_batch),
                np.array(reward_batch),
                np.array(state_batch),
                np.array(next_obs_batch),
                np.array(next_state_batch),
                np.array(terminated_batch),
                np.array(truncated_batch),
            )

    def update_priorities(self, indices, new_priorities):
        if not self.prioritized:
            raise ValueError("Prioritized replay no está activado")
        
        new_max = self.max_priority
        for idx, priority in zip(indices, new_priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                if priority > new_max:
                    new_max = priority
        self.max_priority = new_max

    def __len__(self):
        return len(self.buffer)