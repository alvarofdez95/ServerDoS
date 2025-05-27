# cyber_env.py
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CyberSecurityEnv(gym.Env):
    """
    Entorno de ciberseguridad multiagente:
    - Un servidor central con capacidad limitada.
    - Varias subredes (num_subnets), cada una con hosts.
    - Hosts pueden ser benignos o comprometidos.
    - Agentes defensores (uno por subred) y agente atacante global.
    """

    def __init__(self, config=None):
        super().__init__()
        # Parámetros de configuración (opcionalmente se pueden pasar por config)
        self.min_subnets = 2
        self.max_subnets = 4
        self.min_hosts = 3
        self.max_hosts = 8
        self.server_capacity = 100  # capacidad máxima de tráfico del servidor

        # Inicialmente sin agentes; se crearán en reset()
        self.agents = []
        self.possible_agents = ["attacker"]  # agente atacante siempre posible
        # (los defensores se agregan al reset)
        
        # Espacios de observación y acción vacíos; se configuran tras reset()
        self.defender_observation_space = None
        self.attacker_observation_space = None
        self.defender_action_space = None
        self.attacker_action_space = None

    def reset(self, seed=None, options=None):
        # Inicialización aleatoria del entorno
        self.step_count = 0
        self.num_subnets = random.randint(self.min_subnets, self.max_subnets)
        # Generar número de hosts por subred aleatoriamente
        self.hosts_per_subnet = [random.randint(self.min_hosts, self.max_hosts) 
                                 for _ in range(self.num_subnets)]
        # Marcar hosts comprometidos aleatoriamente
        # E.g., 20% hosts comprometidos en promedio
        self.compromised = {}
        for i in range(self.num_subnets):
            num_hosts = self.hosts_per_subnet[i]
            compromised_hosts = random.sample(range(num_hosts), 
                                              k=random.randint(0, num_hosts//5 + 1))
            self.compromised[i] = set(compromised_hosts)

        # Crear agentes defensores para cada subred
        self.agents = ["attacker"]
        for i in range(self.num_subnets):
            self.agents.append(f"def_{i}")
        # Posibles agentes incluyen el atacante y todos los defensores
        self.possible_agents = list(self.agents)

        # Definir espacios de acción/observación basados en la configuración
        max_hosts = max(self.hosts_per_subnet)
        # Espacio de acción de defensor: (filter_host, bandwidth_limit, IDS, DPI, honeypot_host)
        self.defender_action_space = spaces.Tuple((
            spaces.Discrete(max_hosts+1),   # 0=no filtrar, i>0 filtrar host i-1
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # límite de ancho banda [0,1]
            spaces.Discrete(2),            # IDS: 0=no, 1=sí
            spaces.Discrete(2),            # DPI: 0=no, 1=sí
            spaces.Discrete(max_hosts+1)   # 0=no honeypot, i>0 honeypot en host i-1
        ))
        # Espacio de acción del atacante: (subnet, host, protocolo, intensidad, frecuencia, spoof, evasion)
        self.attacker_action_space = spaces.Tuple((
            spaces.Discrete(self.num_subnets),
            spaces.Discrete(max_hosts),
            spaces.Discrete(3),  # 0=TCP,1=UDP,2=ICMP
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # intensidad [0,1]
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # frecuencia [0,1]
            spaces.Discrete(2),  # spoofing: 0=no, 1=sí
            spaces.Discrete(2)   # evasión: 0=no, 1=sí
        ))
        # Espacio de observación del defensor: vector con [benign_TCP, benign_UDP, benign_ICMP, mal_TCP, mal_UDP, mal_ICMP]
        # Se normaliza en [0, inf) según máximos esperados
        self.defender_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        # Espacio de observación del atacante: [carga_servidor, benign_total, mal_total, (otros indicadores)]
        self.attacker_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Estado inicial: sin tráfico
        self.server_load = 0
        # No hay término inmediato
        obs = self._get_observations()
        return obs, {}

    def step(self, action_dict):
        """
        action_dict debe contener: 
          - "attacker": acción del agente atacante (tuple).
          - "def_i": acción del defensor i (tuple), para i=0..num_subnets-1.
        """
        self.step_count += 1
        # 1) Generar tráfico benigno de todos los hosts
        #    A cada host benigno le asignamos un pequeño tráfico aleatorio por tipo
        benign_traffic = {}
        for i in range(self.num_subnets):
            benign_traffic[i] = {'TCP':0, 'UDP':0, 'ICMP':0}
            for h in range(self.hosts_per_subnet[i]):
                # Generación aleatoria de tráfico benigno por host
                benign_traffic[i]['TCP']  += random.randint(0,5)
                benign_traffic[i]['UDP']  += random.randint(0,5)
                benign_traffic[i]['ICMP'] += random.randint(0,3)
        # 2) Procesar acción del atacante y agregar tráfico malicioso
        att_action = action_dict.get("attacker", None)
        mal_traffic = {i: {'TCP':0, 'UDP':0, 'ICMP':0} for i in range(self.num_subnets)}
        if att_action is not None:
            subnet_id, host_id, proto, intensity, freq, spoof, evasion = att_action
            intensity = float(intensity[0])  # extraer valor real de Box
            freq = float(freq[0])
            # Decidir si ocurre ataque esta vez según frecuencia
            if random.random() < freq:
                # Cantidad de paquetes maliciosos = intensidad * constante
                MAX_MAL = 20
                mal_pkts = int(intensity * MAX_MAL)
                protocol = ['TCP','UDP','ICMP'][proto]
                # Si hay spoofing, simular que proviene de host aleatorio
                if spoof == 1:
                    # elegir un host aleatorio en esa subred (o misma si ninguno)
                    if self.hosts_per_subnet[subnet_id] > 0:
                        spoof_host = random.randrange(self.hosts_per_subnet[subnet_id])
                    else:
                        spoof_host = 0
                    target_host = spoof_host
                else:
                    target_host = host_id if host_id < self.hosts_per_subnet[subnet_id] else 0
                # Agregar paquetes maliciosos al flujo de la subred
                mal_traffic[subnet_id][protocol] += mal_pkts
        # 3) Inicializar contadores para métricas
        total_filtered = 0
        total_ids_dropped = 0
        total_honeypot = 0
        total_rate_limited = 0
        # 4) Aplicar acciones de cada defensor a su subred
        delivered_benign_sub = {}
        delivered_malicious_sub = {}
        for i in range(self.num_subnets):
            action = action_dict.get(f"def_{i}", None)
            # Extraer estados de la subred i
            benign_TCP = benign_traffic[i]['TCP']
            benign_UDP = benign_traffic[i]['UDP']
            benign_ICMP = benign_traffic[i]['ICMP']
            mal_TCP = mal_traffic[i]['TCP']
            mal_UDP = mal_traffic[i]['UDP']
            mal_ICMP = mal_traffic[i]['ICMP']
            # Inicialmente no hay filtrados
            # Filtrado por host específico (acción de filtrado)
            if action is not None:
                filter_host = action[0]  # valor entero
                bw_limit = float(action[1][0])
                use_ids = bool(action[2])
                use_dpi = bool(action[3])
                honeypot_host = action[4]
                # Si filtrar host k (>0), eliminar todo tráfico de ese host
                if filter_host > 0:
                    host_idx = filter_host - 1
                    # Asumimos que todo el tráfico de ese host está incluido en los totales beninos/malosos
                    # Modelamos simplemente como si elimináramos una fracción
                    # Por simplificación, asumimos que filtrar elimina algo proporcional:
                    # por ejemplo, suponiendo que cada host contribuye igual, borramos 1/n partes
                    parts = self.hosts_per_subnet[i]
                    if parts > 0:
                        drop_TCP = benign_TCP/parts + mal_TCP/parts
                        drop_UDP = benign_UDP/parts + mal_UDP/parts
                        drop_ICMP = benign_ICMP/parts + mal_ICMP/parts
                    else:
                        drop_TCP = drop_UDP = drop_ICMP = 0
                    # Quitar del total y contar en filtrados
                    benign_TCP -= drop_TCP
                    mal_TCP   -= drop_TCP
                    benign_UDP -= drop_UDP
                    mal_UDP   -= drop_UDP
                    benign_ICMP -= drop_ICMP
                    mal_ICMP   -= drop_ICMP
                    total_filtered += (drop_TCP + drop_UDP + drop_ICMP)
                # Honeypot en host j (>0): desviamos todo tráfico de ese host
                if honeypot_host > 0:
                    host_idx = honeypot_host - 1
                    parts = self.hosts_per_subnet[i]
                    if parts > 0:
                        # Todo el tráfico de ese host va al honeypot (se captura malicioso)
                        cap_TCP = mal_TCP/parts
                        cap_UDP = mal_UDP/parts
                        cap_ICMP = mal_ICMP/parts
                        # Se pierde también el tráfico benigno de ese host
                        benign_loss = (benign_TCP + benign_UDP + benign_ICMP) / parts
                    else:
                        cap_TCP = cap_UDP = cap_ICMP = benign_loss = 0
                    # Restar esos paquetes del servidor
                    mal_TCP   -= cap_TCP
                    mal_UDP   -= cap_UDP
                    mal_ICMP  -= cap_ICMP
                    benign_TCP -= benign_loss
                    benign_UDP -= benign_loss
                    benign_ICMP -= benign_loss
                    # Contabilizar
                    total_honeypot += (cap_TCP + cap_UDP + cap_ICMP)
                # Aplicar limitación de ancho de banda en la subred
                total_sub_benign = benign_TCP + benign_UDP + benign_ICMP
                total_sub_mal = mal_TCP + mal_UDP + mal_ICMP
                # Se reduce ambos tipos de tráfico
                lim = bw_limit
                drop_ben = total_sub_benign * lim
                drop_mal = total_sub_mal * lim
                benign_TCP *= (1-lim)
                benign_UDP *= (1-lim)
                benign_ICMP *= (1-lim)
                mal_TCP *= (1-lim)
                mal_UDP *= (1-lim)
                mal_ICMP *= (1-lim)
                total_rate_limited += (drop_ben + drop_mal)
                # IDS y DPI: detectan y eliminan parte del tráfico malicioso restante
                # Factor de detección: sin evasión=0.5, con evasión=0.2 (más difícil detectar)
                if use_ids or use_dpi:
                    drop_factor = 0.0
                    if use_ids:
                        drop_factor += 0.5 if (action[6] == 0) else 0.2  # acción[6] sería evasión, pero la tenemos arriba
                    if use_dpi:
                        drop_factor += 0.5 if (action[7] == 0) else 0.2
                    # No exceder 1.0
                    drop_factor = min(drop_factor, 1.0)
                    # Calcular paquetes maliciosos detectados y eliminados
                    detected = (mal_TCP + mal_UDP + mal_ICMP) * drop_factor
                    total_ids_dropped += detected
                    mal_TCP = (mal_TCP + mal_UDP + mal_ICMP) * (1 - drop_factor)
                    # Como simplificación, distribuimos el resto proporcionalmente
                    # (en realidad podríamos afinar por protocolo)
                    mal_UDP = 0
                    mal_ICMP = 0
            # Datos finales entregados al servidor desde esta subred
            delivered_benign = benign_TCP + benign_UDP + benign_ICMP
            delivered_mal = mal_TCP + mal_UDP + mal_ICMP
            delivered_benign_sub[i] = delivered_benign
            delivered_malicious_sub[i] = delivered_mal
        # 5) Agregar contribuciones al servidor
        total_benign = sum(delivered_benign_sub.values())
        total_malicious = sum(delivered_malicious_sub.values())
        total_traffic = total_benign + total_malicious
        # Si excede capacidad del servidor, se recorta (priorizando eliminar malicioso primero)
        if total_traffic > self.server_capacity:
            if total_malicious >= self.server_capacity:
                # Saturación completa con tráfico malicioso
                total_malicious = self.server_capacity
                total_benign = 0
            else:
                # Parte benigno limitada
                total_benign = min(total_benign, self.server_capacity - total_malicious)
        self.server_load = total_benign + total_malicious

        # 6) Construir observaciones y recompensas por agente
        obs = {}
        rewards = {}
        dones = {}
        # Observación para cada defensor: tráfico (TCP,UDP,ICMP) benigno vs malicioso en su subred
        for i in range(self.num_subnets):
            obs[f"def_{i}"] = np.array([
                delivered_benign_sub[i] * 0.5,  # escala ejemplo
                delivered_benign_sub[i] * 0.3,
                delivered_benign_sub[i] * 0.2,
                delivered_malicious_sub[i] * 0.5,
                delivered_malicious_sub[i] * 0.3,
                delivered_malicious_sub[i] * 0.2
            ], dtype=np.float32)
            # Recompensa defensora: benigno entregado - malicioso entregado
            rewards[f"def_{i}"] = delivered_benign_sub[i] - delivered_malicious_sub[i]
            dones[f"def_{i}"] = False
        # Observación para el atacante: carga total del servidor, total benigno y malicioso,
        # más eventualmente otros indicadores (simplificamos 4 dimensiones)
        obs["attacker"] = np.array([
            self.server_load,
            total_benign,
            total_malicious,
            0.0  # placeholder
        ], dtype=np.float32)
        # Recompensa del atacante: equivalente a tráfico malicioso entregado
        rewards["attacker"] = total_malicious
        dones["attacker"] = False
        # Señalar fin de episodio cuando se cumplan condiciones (ej. máximo de pasos)
        done_flag = False
        max_steps = 100
        if self.step_count >= max_steps:
            done_flag = True
        dones["__all__"] = done_flag

        # 7) Información adicional (logs, métricas)
        info = {
            "metrics": {
                "total_benign": total_benign,
                "total_malicious": total_malicious,
                "filtered": total_filtered,
                "ids_dropped": total_ids_dropped,
                "honeypot_captured": total_honeypot,
                "rate_limited": total_rate_limited,
                "server_load": self.server_load
            }
        }
        return obs, rewards, dones, info
