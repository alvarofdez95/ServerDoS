# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:11:21 2025

@author: jafernandez
"""

import torch
import numpy as np
from datetime import datetime

class QMIXTrainer:
    def __init__(self, qmix, env, target_update_freq, epsilon_start, epsilon_end, batch_size, max_steps):
        """
        Inicializa el entrenador de QMIX.
        
        Args:
            qmix (QMIX): Instancia de la clase QMIX.
            env: Entorno multiagente.
            target_update_freq (int): Frecuencia de actualización de las redes objetivo.
            epsilon_start (float): Valor inicial de ε.
            epsilon_end (float): Valor final de ε.
            epsilon_decay_steps (int): Número de pasos para decaer ε de inicio a fin.
            batch_size (int): Tamaño del lote para entrenar.
            max_steps (int): Número máximo de pasos por episodio.
        """
        self.qmix = qmix
        self.env = env
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        #self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.steps = 0
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    def get_epsilon(self, progress):
        """Calcula el valor actual de ε usando decaimiento lineal."""
        # if self.steps >= self.epsilon_decay_steps:
        #     return self.epsilon_end
        # return self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.steps / self.epsilon_decay_steps)
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def train(self, num_episodes):
        """Entrena el modelo QMIX durante un número de episodios."""
        
        episode_rewards = []
        steps_per_episode = []
        mean_reward = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()  # Reiniciar el entorno
            obs = np.array(list(obs.values()))
            state = self.env.state()  # Obtener el estado global inicial
            hidden_states = [agent.init_hidden() for agent in self.qmix.agents]  # Estados ocultos iniciales
            episode_reward = 0
            actions_gen = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            actions = np.array(list(actions_gen.values()))
            # Ejecutar acciones en el entorno
            next_obs, reward, terminated, truncated, info = self.env.step(actions_gen)
            #reward = np.array(list(reward.values())) 
            #episode_reward += reward
            next_state = self.env.state()

            for step in range(self.max_steps):
                self.steps += 1

                # Seleccionar acciones usando ε-greedy
                epsilon = self.get_epsilon(episode/num_episodes)
                actions, hidden_states = self.qmix.select_actions(
                    obs,
                    actions,
                    hidden_states,
                    epsilon
                )

                # Ejecutar acciones en el entorno
                next_obs, reward, terminated, truncated, _ = self.env.step(dict(zip(self.env.agents, actions)))
                
                next_obs = np.array(list(next_obs.values()))
                reward = np.array(list(reward.values()))                 
                terminated = np.array(list(terminated.values()))
                truncated = np.array(list(truncated.values()))
                
                next_state = self.env.state()

                # Guardar experiencia en el replay buffer
                self.qmix.save_experience(obs, actions, reward, state, next_obs, next_state, terminated, truncated)

                # Entrenar el modelo con un lote de experiencias
                self.qmix.train(self.batch_size)

                # Actualizar observaciones y estado
                obs = next_obs
                state = next_state
                episode_reward += reward

                # Actualizar las redes objetivo cada target_update_freq pasos
                if self.steps % self.target_update_freq == 0:
                    self.qmix.update_target_networks()

                # Verificar si el episodio ha terminado
                if any(terminated) or any(truncated):
                    break
            
            steps_per_episode.append(step+1)
            episode_rewards.append(episode_reward)
            mean_reward.append(episode_reward[0]/(step+1))
            
            

            # Guardar el modelo periódicamente
            if (episode + 1) % (num_episodes/10) == 0:
                # Imprimir el progreso del entrenamiento
                print(f"Episodio {episode + 1}, Recompensa del episodio: {episode_reward}, ε: {epsilon:.4f}, Pasos: {self.steps}, Recompensa media acumulada: {np.mean(mean_reward)}")
                for i, agent in enumerate(self.qmix.agents):
                    name = "defender_" + str(i) + "_" + self.timestamp + "_episode_" + str(episode+1)
                    agent.save_checkpoint(name)
        
        return episode_rewards, steps_per_episode, mean_reward