# test.py
import numpy as np
import matplotlib.pyplot as plt
from cyber_env import CyberSecurityEnv

env = CyberSecurityEnv()
num_episodes = 3

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step = 0
    # Listas para almacenar métricas de la simulación
    traffic_history = []
    mal_history = []
    reward_def_history = []
    reward_att_history = []
    filtered_history = []
    honeypot_history = []
    ids_history = []

    while not done:
        # Para prueba, usamos políticas sencillas (por ejemplo, aleatorias)
        actions = {}
        for i in range(env.num_subnets):
            actions[f"def_{i}"] = env.defender_action_space.sample()
        actions["attacker"] = env.attacker_action_space.sample()

        obs, rewards, dones, info = env.step(actions)
        done = dones["__all__"]

        # Registrar métricas del paso
        metrics = info["metrics"]
        total_traffic = metrics["total_benign"] + metrics["total_malicious"]
        traffic_history.append(total_traffic)
        mal_history.append(metrics["total_malicious"])
        # Asumimos recompensa defensora promedio y recompensa atacante
        rew_def = np.mean([rewards[f"def_{i}"] for i in range(env.num_subnets)])
        rew_att = rewards["attacker"]
        reward_def_history.append(rew_def)
        reward_att_history.append(rew_att)
        filtered_history.append(metrics["filtered"])
        honeypot_history.append(metrics["honeypot_captured"])
        ids_history.append(metrics["ids_dropped"])

        step += 1

    # Tras cada episodio, graficar resultados
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(traffic_history, label='Tráfico total (paqs)')
    plt.plot(mal_history, label='Tráfico malicioso')
    plt.xlabel('Paso')
    plt.ylabel('Paquetes')
    plt.legend()
    plt.title(f'Episodio {ep+1} - Tráfico en el servidor')
    
    plt.subplot(2,1,2)
    plt.plot(reward_def_history, label='Recompensa defensores')
    plt.plot(reward_att_history, label='Recompensa atacante')
    plt.xlabel('Paso')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.title('Recompensas por paso')
    
    plt.tight_layout()
    plt.savefig(f'episode_{ep+1}_metrics.png')  # Guardar figura
    plt.show()

    # Mostrar registros resumen
    print(f"Episodio {ep+1} finalizado en {step} pasos.")
    print(f"Recompensa total defensores (media): {sum(reward_def_history):.2f}")
    print(f"Recompensa total atacante: {sum(reward_att_history):.2f}")
    print(f"Máximo tráfico malicioso: {max(mal_history):.2f} paqs")
    print(f"Eventos filtrados totales: {sum(filtered_history):.2f}, honeypots: {sum(honeypot_history):.2f}, IDS: {sum(ids_history):.2f}\n")
