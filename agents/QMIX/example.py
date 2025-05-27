# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:57:30 2025

@author: jafernandez
"""

import torch


from vicomrl.agents.QMIX.qmix_QMIX import QMIX
from vicomrl.agents.QMIX.qmix_mixing_network import MixingNetwork
from vicomrl.agents.QMIX.qmix_replay_buffer import ReplayBuffer
from vicomrl.agents.QMIX.qmix_drqn import DRQN
import numpy as np

# Example usage
#def test_qmix_total():
# Parámetros
obs_dim = 10
state_dim = 20
n_actions = 4
n_agents = 3
hidden_dim = 64
mixing_embed_dim = 32
buffer_capacity = 10000
lr = 0.001
gamma = 0.99
batch_size = 32

# Crear el modelo QMIX
qmix = QMIX(obs_dim, state_dim, n_actions, n_agents, hidden_dim, mixing_embed_dim, buffer_capacity, lr, gamma)

# Ejemplo de interacción con el entorno
for episode in range(1000):
    obs = np.random.randn(n_agents, obs_dim)  # Observaciones iniciales
    state = np.random.randn(state_dim)  # Estado global inicial
    hidden_states = [agent.init_hidden() for agent in qmix.agents]  # Estados ocultos iniciales
    episode_reward = 0

    for step in range(100):  # Máximo de 100 pasos por episodio
        # Seleccionar acciones
        actions, hidden_states = qmix.select_actions(
            torch.FloatTensor(obs).unsqueeze(0),
            hidden_states,
            epsilon=0.1
        )

        # Ejecutar acciones en el entorno (simulado)
        next_obs = np.random.randn(n_agents, obs_dim)  # Nuevas observaciones
        next_state = np.random.randn(state_dim)  # Nuevo estado global
        reward = np.random.rand()  # Recompensa global
        terminated = False  # El episodio no ha terminado
        truncated = False  # El episodio no ha sido truncado

        # Guardar experiencia en el buffer
        qmix.save_experience(obs, actions, reward, state, next_obs, next_state, terminated, truncated)

        # Entrenar el modelo
        qmix.train(batch_size)

        # Actualizar observaciones y estado
        obs = next_obs
        state = next_state
        episode_reward += reward

    print(f"Episodio {episode}, Recompensa: {episode_reward}")

def test_mixing_network():
    # Parámetros
    state_dim = 10
    n_agents = 3
    mixing_embed_dim = 32
    batch_size = 64
    
    # Crear la mixing network
    mixing_network = MixingNetwork(state_dim, n_agents, mixing_embed_dim)
    
    # Valores Q de los agentes (batch_size, n_agents)
    agent_qs = torch.randn(batch_size, n_agents)
    
    # Estado global (batch_size, state_dim)
    states = torch.randn(batch_size, state_dim)
    
    # Calcular Q_tot
    q_tot = mixing_network(agent_qs, states)
    print(q_tot.shape)  # Debería ser (batch_size, 1)
    return q_tot

def test_replay_buffer():
    # Crear el replay buffer
    buffer = ReplayBuffer(capacity=10000)

    # Ejemplo de una experiencia
    obs = np.array([[0.1, 0.2], [0.3, 0.4]])  # Observaciones de 2 agentes (n_agents, obs_dim)
    actions = np.array([0, 1])  # Acciones de 2 agentes (n_agents,)
    reward = 1.0  # Recompensa global (escalar)
    state = np.array([0.5, 0.6])  # Estado global (state_dim,)
    next_obs = np.array([[0.2, 0.3], [0.4, 0.5]])  # Nuevas observaciones (n_agents, obs_dim)
    next_state = np.array([0.6, 0.7])  # Nuevo estado global (state_dim,)
    terminated = False  # El episodio no ha terminado naturalmente
    truncated = False  # El episodio no ha sido truncado

    # Guardar la experiencia en el buffer
    batch_size = 2
    for i in range(batch_size):
        buffer.push(obs, actions, reward, state, next_obs, next_state, terminated, truncated)

    #batch_size = 2

    # Muestrear un lote de experiencias
    #obs_batch, actions_batch, reward_batch, state_batch, next_obs_batch, next_state_batch, terminated_batch, truncated_batch  = buffer.sample(batch_size=batch_size)

    return buffer    

def test_drqn():
    # Parámetros
    input_dim = 10  # Dimensión de la observación
    hidden_dim = 64  # Dimensión del estado oculto
    output_dim = 4  # Número de acciones posibles
    batch_size = 32  # Tamaño del batch
    
    # Crear el DRQN
    drqn = DRQN(input_dim, output_dim, hidden_dim)
    
    # Observación actual (batch_size, input_dim)
    obs = torch.randn(batch_size, input_dim)
    
    # Inicializar el estado oculto (batch_size, hidden_dim)
    hidden_state = drqn.init_hidden(batch_size)
    
    # Paso hacia adelante
    q_values, new_hidden_state = drqn(obs, hidden_state)
    
    print("Q-values:", q_values.shape)  # Debería ser (batch_size, output_dim)
    print("Nuevo estado oculto:", new_hidden_state.shape)  # Debería ser (batch_size, hidden_dim)
    
    return q_values, new_hidden_state

if __name__ == "__main__":
    #q_tot, updated_hidden_states = test_qmix_total()
    #q_tot = test_mixing_network()
    
    #buffer = test_replay_buffer()
    # Muestrear un lote de experiencias
    batch_size = 2
    #obs_batch, actions_batch, reward_batch, state_batch, next_obs_batch, next_state_batch, terminated_batch, truncated_batch  = buffer.sample(batch_size=batch_size)
    
    #q_values, new_hidden_state = test_drqn()
    
    #test_qmix_total()