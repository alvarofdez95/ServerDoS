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
    # New history trackers
    compromised_hosts_history = []
    active_honeynets_history = []
    intel_items_history = []
    defender_costs_history = []

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
        # Collect new metrics
        compromised_hosts_history.append(metrics["total_compromised_hosts"])
        active_honeynets_history.append(metrics["honeynets_active"])
        intel_items_history.append(metrics["shared_intel_items"])
        defender_costs_history.append(metrics["total_defender_action_costs"])

        step += 1

    # Tras cada episodio, graficar resultados
    plt.figure(figsize=(12, 10)) # Adjusted figsize for 3 subplots
    plt.subplot(3,1,1) # Changed to 3,1,1
    plt.plot(traffic_history, label='Tráfico total (paqs)')
    plt.plot(mal_history, label='Tráfico malicioso')
    plt.xlabel('Paso')
    plt.ylabel('Paquetes')
    plt.legend()
    plt.title(f'Episodio {ep+1} - Tráfico en el servidor')
    
    plt.subplot(3,1,2) # Changed to 3,1,2
    plt.plot(reward_def_history, label='Recompensa defensores (media)')
    plt.plot(reward_att_history, label='Recompensa atacante')
    plt.xlabel('Paso')
    plt.ylabel('Recompensa')
    plt.legend()
    plt.title('Recompensas por paso')

    plt.subplot(3,1,3) # New subplot for additional metrics
    plt.plot(compromised_hosts_history, label='Hosts Comprometidos')
    plt.plot(active_honeynets_history, label='Honeynets Activos')
    plt.xlabel('Paso')
    plt.ylabel('Cantidad')
    plt.legend()
    plt.title(f'Episodio {ep+1} - Métricas Adicionales')
    
    plt.tight_layout()
    plt.savefig(f'episode_{ep+1}_metrics.png')  # Guardar figura
    plt.show()

    # Mostrar registros resumen
    print(f"Episodio {ep+1} finalizado en {step} pasos.")
    print(f"Recompensa total defensores (media): {sum(reward_def_history):.2f}")
    print(f"Recompensa total atacante: {sum(reward_att_history):.2f}")
    print(f"Coste total acciones defensores: {sum(defender_costs_history):.2f}")
    print(f"Máximo tráfico malicioso: {max(mal_history) if mal_history else 0:.2f} paqs")
    print(f"Eventos filtrados totales: {sum(filtered_history):.2f}, honeypots capturado: {sum(honeypot_history):.2f}, IDS drops: {sum(ids_history):.2f}")
    print(f"Total hosts comprometidos (final): {compromised_hosts_history[-1] if compromised_hosts_history else 0}")
    print(f"Media honeynets activos por paso: {np.mean(active_honeynets_history) if active_honeynets_history else 0:.2f}")
    print(f"Total items de inteligencia compartidos (final): {intel_items_history[-1] if intel_items_history else 0}\n")
