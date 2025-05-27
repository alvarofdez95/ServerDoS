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

        # New constants for action spaces
        self.N_PREDEFINED_RULES = 5
        self.N_VULNERABILITY_TYPES = 3
        self.N_DEFENSE_TYPES_TO_TARGET = 3

        # New constants for observation spaces
        self.N_ALERT_SEVERITIES = 3
        self.N_TRAFFIC_ANALYSIS_FEATURES = 3
        self.N_HONEYNET_REPORT_FEATURES = 2
        self.N_SHARED_INTEL_FEATURES = 2
        self.N_DEFENSE_POSTURE_FEATURES = 3
        self.N_ATTACK_FEEDBACK_FEATURES = 2
        # self.max_hosts is used for max_vulnerability_features_per_defender
        # self.max_vulnerability_features_per_defender = self.max_hosts * self.N_VULNERABILITY_TYPES
        self.MAX_MAL_AMP = 100 # For DDoS amplification attacks
        self.PATCH_COST = 0.1 # Cost for patching a vulnerability
        self.HONEYNET_DEPLOY_COST = 0.5 # Cost for deploying a honeynet
        self.bandwidth_levels = [0.0, 0.25, 0.5, 0.75, 1.0] # For discretized bandwidth limiting

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

        # Initialize new state variables
        self.host_vulnerabilities = {} # Key: subnet_id, Value: {host_id: [vuln_status_list]}
        for i in range(self.num_subnets):
            self.host_vulnerabilities[i] = {}
            for h_idx in range(self.hosts_per_subnet[i]):
                # Each host has a list of binary flags for N_VULNERABILITY_TYPES
                self.host_vulnerabilities[i][h_idx] = [
                    1 if random.random() < 0.25 else 0 for _ in range(self.N_VULNERABILITY_TYPES)
                ]

        self.honeynet_status = [0] * self.num_subnets # 0 for off, 1 for on, per subnet
        
        self.isolated_hosts = {} # Key: subnet_id, Value: set of isolated host_idx
        for i in range(self.num_subnets):
            self.isolated_hosts[i] = set()
            
        self.shared_intelligence_pool = [] # Stores shared intel items (e.g., dicts representing intel)
        
        # For attacker_recon_results, initializing structures that directly feed observations
        # These will be updated by attacker's reconnaissance actions if they are implemented
        self.attacker_recon_aggregate_vuln_score = np.zeros(self.num_subnets, dtype=np.float32)
        self.attacker_recon_defense_posture = np.zeros((self.num_subnets, self.N_DEFENSE_POSTURE_FEATURES), dtype=np.float32)
        
        self.last_attack_feedback = {'status': 0, 'damage': 0.0} # status: 0 neutral, 1 success, -1 fail

        # Crear agentes defensores para cada subred
        self.agents = ["attacker"]
        for i in range(self.num_subnets):
            self.agents.append(f"def_{i}")
        # Posibles agentes incluyen el atacante y todos los defensores
        self.possible_agents = list(self.agents)

        # Definir espacios de acción/observación basados en la configuración
        max_hosts_val = 0
        if self.hosts_per_subnet: # Ensure hosts_per_subnet is not empty
            max_hosts_val = max(self.hosts_per_subnet) if self.hosts_per_subnet else 0
        
        # Espacio de acción de defensor:
        # Original actions: (filter_host, bandwidth_limit, IDS, DPI, honeypot_host)
        # New actions: DeployAdvancedFirewallRule, DeployHoneynet, IsolateCompromisedHost, ShareThreatIntelligence, PatchVulnerability
        defender_action_components = list(self.defender_action_space.spaces) if hasattr(self, 'defender_action_space') and self.defender_action_space else [None]*10
        defender_action_components[0] = spaces.Discrete(max_hosts_val + 1)   # 0=no filtrar, i>0 filtrar host i-1
        defender_action_components[1] = spaces.Discrete(len(self.bandwidth_levels)) # DISCRETIZED: BW Limit index
        defender_action_components[2] = spaces.Discrete(2)            # IDS: 0=no, 1=sí
        defender_action_components[3] = spaces.Discrete(2)            # DPI: 0=no, 1=sí
        defender_action_components[4] = spaces.Discrete(max_hosts_val + 1)   # 0=no honeypot, i>0 honeypot en host i-1
        # New Defender Actions
        defender_action_components[5] = spaces.Discrete(self.N_PREDEFINED_RULES) # DeployAdvancedFirewallRule
        defender_action_components[6] = spaces.Discrete(2)                       # DeployHoneynet (0: no change/off, 1: deploy/on)
        defender_action_components[7] = spaces.Discrete(max_hosts_val + 1)       # IsolateCompromisedHost (0: no action, i > 0: isolate host i-1)
        defender_action_components[8] = spaces.Discrete(2)                       # ShareThreatIntelligence (0: no, 1: yes)
        defender_action_components[9] = spaces.Tuple((spaces.Discrete(max_hosts_val + 1), spaces.Discrete(self.N_VULNERABILITY_TYPES))) # PatchVulnerability (host_idx, vuln_type)
        
        self.defender_action_space = spaces.Tuple(tuple(defender_action_components))
        
        # Ensure the original definition is replaced if it existed, or created if it didn't.
        # This is a bit verbose to handle the case where the Tuple might not have been initialized with 10 elements.
        # A cleaner way if always starting from scratch:
        # self.defender_action_space = spaces.Tuple((
        #     spaces.Discrete(max_hosts_val + 1),
        #     spaces.Discrete(len(self.bandwidth_levels)), # DISCRETIZED BW Limit
        #     spaces.Discrete(2),
        #     spaces.Discrete(2),
        #     spaces.Discrete(max_hosts_val + 1),
        #     spaces.Discrete(self.N_PREDEFINED_RULES),
        #     spaces.Discrete(2),
        #     spaces.Discrete(max_hosts_val + 1),
        #     spaces.Discrete(2),
        #     spaces.Tuple((spaces.Discrete(max_hosts_val + 1), spaces.Discrete(self.N_VULNERABILITY_TYPES)))
        # ))

        # Espacio de acción del atacante:
        # Original actions: (subnet, host, protocolo, intensidad, frecuencia, spoof, evasion)
            # New Defender Actions
            spaces.Discrete(self.N_PREDEFINED_RULES), # DeployAdvancedFirewallRule
            spaces.Discrete(2),                       # DeployHoneynet (0: no change/off, 1: deploy/on)
            spaces.Discrete(max_hosts_val + 1),       # IsolateCompromisedHost (0: no action, i > 0: isolate host i-1)
            spaces.Discrete(2),                       # ShareThreatIntelligence (0: no, 1: yes)
            spaces.Tuple((spaces.Discrete(max_hosts_val + 1), spaces.Discrete(self.N_VULNERABILITY_TYPES))) # PatchVulnerability (host_idx, vuln_type)
        ))
        
        # Espacio de acción del atacante:
        # Original actions: (subnet, host, protocolo, intensidad, frecuencia, spoof, evasion)
        # New actions: NetworkReconnaissance, ExploitSpecificVulnerability, LaunchDDoSAmplification, AttemptToDisableDefenses
        self.attacker_action_space = spaces.Tuple((
            spaces.Discrete(self.num_subnets), # Target subnet for original attack actions
            spaces.Discrete(max_hosts_val if max_hosts_val > 0 else 1), # Target host for original attack (ensure Discrete > 0)
            spaces.Discrete(3),  # 0=TCP,1=UDP,2=ICMP for original attack
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # intensidad [0,1] for original attack
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # frecuencia [0,1] for original attack
            spaces.Discrete(2),  # spoofing: 0=no, 1=sí for original attack
            spaces.Discrete(2),  # evasión: 0=no, 1=sí for original attack
            # New Attacker Actions
            spaces.Discrete(self.num_subnets + 1), # NetworkReconnaissance (0: no recon, i > 0: recon subnet i-1)
            spaces.Tuple((spaces.Discrete(self.num_subnets), spaces.Discrete(max_hosts_val if max_hosts_val > 0 else 1), spaces.Discrete(self.N_VULNERABILITY_TYPES))), # ExploitSpecificVulnerability (subnet, host_in_subnet, vuln_type)
            spaces.Tuple((spaces.Discrete(self.num_subnets), spaces.Box(low=0.0, high=1.0, shape=(1,)))), # LaunchDDoSAmplification (target_subnet, intensity_multiplier)
            spaces.Tuple((spaces.Discrete(self.num_subnets), spaces.Discrete(self.N_DEFENSE_TYPES_TO_TARGET))) # AttemptToDisableDefenses (target_subnet, defense_type)
        ))
        
        
        # Espacio de observación del defensor
        # Original: 6 features
        # New:
        # 1. SubnetVulnerabilityAssessment: max_hosts_val * N_VULNERABILITY_TYPES
        # 2. DetailedSubnetTrafficAnalysis: N_TRAFFIC_ANALYSIS_FEATURES
        # 3. SpecificIDSAlertFeed: N_ALERT_SEVERITIES
        # 4. SharedThreatIntelligenceFeed: N_SHARED_INTEL_FEATURES
        # 5. HoneynetActivityReport: N_HONEYNET_REPORT_FEATURES
        defender_obs_shape = 6 + \
                             (max_hosts_val * self.N_VULNERABILITY_TYPES) + \
                             self.N_TRAFFIC_ANALYSIS_FEATURES + \
                             self.N_ALERT_SEVERITIES + \
                             self.N_SHARED_INTEL_FEATURES + \
                             self.N_HONEYNET_REPORT_FEATURES
        self.defender_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(defender_obs_shape,), dtype=np.float32
        )

        # Espacio de observación del atacante
        # Original: 4 features
        # New:
        # 1. ReconnaissanceResults (simplified): self.num_subnets (aggregate vulnerability score per subnet)
        # 2. EstimatedDefensePostureOfSubnets: self.num_subnets * N_DEFENSE_POSTURE_FEATURES
        # 3. AttackSuccessFailureFeedback: N_ATTACK_FEEDBACK_FEATURES
        # 4. AvgSubnetSaturation: 1
        attacker_obs_shape = 4 + \
                             self.num_subnets + \
                             (self.num_subnets * self.N_DEFENSE_POSTURE_FEATURES) + \
                             self.N_ATTACK_FEEDBACK_FEATURES + \
                             1
        self.attacker_observation_space = spaces.Box(
            low=0, high=np.inf, shape=(attacker_obs_shape,), dtype=np.float32
        )

        # Estado inicial: sin tráfico
        self.server_load = 0
        # For _get_observations in reset:
        self.current_max_hosts_val = 0
        if self.hosts_per_subnet:
            self.current_max_hosts_val = max(self.hosts_per_subnet) if self.hosts_per_subnet else 0
        # Initialize placeholders for traffic values for the first call to _get_observations from reset
        self.delivered_benign_sub_at_step_end = {i: 0 for i in range(self.num_subnets)}
        self.delivered_malicious_sub_at_step_end = {i: 0 for i in range(self.num_subnets)}
        self.total_benign_traffic_at_step_end = 0
        self.total_malicious_traffic_at_step_end = 0
        
        obs = self._get_observations()
        return obs, {}

    def _get_observations(self):
        obs = {}
        # Defender observations
        for i in range(self.num_subnets):
            # Original 6 features (scaled traffic)
            delivered_benign = self.delivered_benign_sub_at_step_end.get(i, 0.0)
            delivered_malicious = self.delivered_malicious_sub_at_step_end.get(i, 0.0)
            original_obs_def_part = np.array([
                delivered_benign * 0.5, delivered_benign * 0.3, delivered_benign * 0.2,
                delivered_malicious * 0.5, delivered_malicious * 0.3, delivered_malicious * 0.2
            ], dtype=np.float32)

            # SubnetVulnerabilityAssessment
            vuln_assessment_flat = []
            num_hosts_in_subnet = self.hosts_per_subnet[i]
            for h_idx in range(num_hosts_in_subnet):
                vuln_assessment_flat.extend(self.host_vulnerabilities[i].get(h_idx, [0]*self.N_VULNERABILITY_TYPES))
            padding_len = (self.current_max_hosts_val - num_hosts_in_subnet) * self.N_VULNERABILITY_TYPES
            vuln_assessment_flat.extend([0.0] * padding_len)
            vuln_assessment_obs = np.array(vuln_assessment_flat, dtype=np.float32)

            # DetailedSubnetTrafficAnalysis (Placeholder)
            traffic_analysis_obs = np.random.rand(self.N_TRAFFIC_ANALYSIS_FEATURES).astype(np.float32)
            
            # SpecificIDSAlertFeed (Placeholder)
            ids_alert_feed_obs = np.random.randint(0, 5, size=self.N_ALERT_SEVERITIES).astype(np.float32)
            
            # SharedThreatIntelligenceFeed
            num_intel_items = len(self.shared_intelligence_pool)
            avg_severity = 0.0
            if num_intel_items > 0:
                # Assuming intel_item might not be a dict or have 'severity', provide defaults
                severities = [intel.get('severity', 0) for intel in self.shared_intelligence_pool if isinstance(intel, dict)]
                if severities:
                    avg_severity = np.mean(severities)
            shared_intel_feed_obs = np.array([float(num_intel_items), avg_severity], dtype=np.float32)
            
            # HoneynetActivityReport
            honeynet_report_obs = np.zeros(self.N_HONEYNET_REPORT_FEATURES, dtype=np.float32)
            if self.honeynet_status[i] == 1: # If honeynet is active for this subnet
                honeynet_report_obs = np.random.rand(self.N_HONEYNET_REPORT_FEATURES).astype(np.float32)

            obs[f"def_{i}"] = np.concatenate([
                original_obs_def_part,
                vuln_assessment_obs,
                traffic_analysis_obs,
                ids_alert_feed_obs,
                shared_intel_feed_obs,
                honeynet_report_obs
            ]).astype(np.float32)

        # Attacker observations
        original_obs_att_part = np.array([
            self.server_load, 
            self.total_benign_traffic_at_step_end, 
            self.total_malicious_traffic_at_step_end, 
            0.0  # Placeholder from original
        ], dtype=np.float32)
        
        recon_agg_vuln_score_obs = self.attacker_recon_aggregate_vuln_score.astype(np.float32)
        est_defense_posture_obs = self.attacker_recon_defense_posture.flatten().astype(np.float32)
        
        attack_feedback_obs = np.array([
            float(self.last_attack_feedback['status']), 
            float(self.last_attack_feedback['damage'])
        ], dtype=np.float32)
        
        # AvgSubnetSaturation
        total_subnet_traffic_sum = sum(self.delivered_benign_sub_at_step_end.get(j, 0.0) + self.delivered_malicious_sub_at_step_end.get(j, 0.0) for j in range(self.num_subnets))
        avg_subnet_saturation_val = 0.0
        if self.num_subnets > 0 and self.server_capacity > 0 : # Avoid division by zero
            # Proxy for max capacity per subnet; can be refined
            max_cap_per_subnet_proxy = self.server_capacity / self.num_subnets 
            if max_cap_per_subnet_proxy > 0:
                 avg_subnet_saturation_val = total_subnet_traffic_sum / (self.num_subnets * max_cap_per_subnet_proxy)
            elif total_subnet_traffic_sum > 0: # if max_cap_per_subnet_proxy is 0 but there's traffic
                 avg_subnet_saturation_val = 1.0 # Consider it saturated
        elif total_subnet_traffic_sum > 0: # If num_subnets or server_capacity is 0, but traffic exists (edge case)
            avg_subnet_saturation_val = 1.0 # Saturated

        avg_subnet_sat_obs = np.array([avg_subnet_saturation_val], dtype=np.float32)

        obs["attacker"] = np.concatenate([
            original_obs_att_part,
            recon_agg_vuln_score_obs,
            est_defense_posture_obs,
            attack_feedback_obs,
            avg_subnet_sat_obs
        ]).astype(np.float32)
        
        return obs

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
        
        # Initialize last_attack_feedback for this step (can be overwritten by attacker actions)
        self.last_attack_feedback = {'status': 0, 'damage': 0.0}

        if att_action is not None:
            # Unpack all attacker actions (original 7 + 4 new)
            orig_att_subnet_idx, orig_att_host_idx, orig_att_proto_idx, \
            orig_att_intensity_box, orig_att_freq_box, orig_att_spoof_toggle, orig_att_evasion_toggle, \
            recon_subnet_target_choice, \
            exploit_action_tuple, \
            ddos_amp_action_tuple, \
            disable_defense_action_tuple = att_action

            # Original attack logic (first 7 actions)
            orig_att_intensity = float(orig_att_intensity_box[0])
            orig_att_freq = float(orig_att_freq_box[0])
            if random.random() < orig_att_freq:
                MAX_MAL = 20 # Original max malicious packets
                mal_pkts = int(orig_att_intensity * MAX_MAL)
                protocol = ['TCP','UDP','ICMP'][orig_att_proto_idx]
                
                target_host_for_original_attack = 0
                if self.hosts_per_subnet[orig_att_subnet_idx] > 0: # Check if subnet has hosts
                    if orig_att_spoof_toggle == 1:
                        target_host_for_original_attack = random.randrange(self.hosts_per_subnet[orig_att_subnet_idx])
                    else:
                        target_host_for_original_attack = orig_att_host_idx if orig_att_host_idx < self.hosts_per_subnet[orig_att_subnet_idx] else 0
                
                mal_traffic[orig_att_subnet_idx][protocol] += mal_pkts
                # Basic feedback for original attack type, can be refined
                self.last_attack_feedback = {'status': 1, 'damage': mal_pkts * 0.05}


            # New Attacker Actions Processing
            # NetworkReconnaissance (att_action[7])
            recon_target_subnet_idx = recon_subnet_target_choice - 1 # 0 is no-op, 1 to N_subnets maps to 0 to N_subnets-1
            if recon_target_subnet_idx >= 0 and recon_target_subnet_idx < self.num_subnets:
                current_vuln_score = 0
                for host_id_in_recon_subnet in range(self.hosts_per_subnet[recon_target_subnet_idx]):
                    if host_id_in_recon_subnet in self.host_vulnerabilities[recon_target_subnet_idx]:
                        current_vuln_score += sum(self.host_vulnerabilities[recon_target_subnet_idx][host_id_in_recon_subnet])
                self.attacker_recon_aggregate_vuln_score[recon_target_subnet_idx] = float(current_vuln_score)
                # Placeholder for defense posture update
                self.attacker_recon_defense_posture[recon_target_subnet_idx] = [float(random.randint(0,1)), float(random.randint(0,1)), random.random()]
                self.last_attack_feedback = {'status': 0, 'damage': 0} # Recon is not direct damage

            # ExploitSpecificVulnerability (att_action[8])
            exploit_subnet_idx, exploit_host_in_subnet_idx, exploit_vuln_type_idx = exploit_action_tuple
            if exploit_subnet_idx < self.num_subnets and \
               exploit_host_in_subnet_idx < self.hosts_per_subnet[exploit_subnet_idx]:
                if self.host_vulnerabilities[exploit_subnet_idx][exploit_host_in_subnet_idx][exploit_vuln_type_idx] == 1:
                    self.compromised[exploit_subnet_idx].add(exploit_host_in_subnet_idx)
                    self.last_attack_feedback = {'status': 1, 'damage': 10.0}
                    mal_traffic[exploit_subnet_idx]['TCP'] += 10 # Example malicious traffic
                else:
                    self.last_attack_feedback = {'status': -1, 'damage': 0.0}
            
            # LaunchDDoSAmplification (att_action[9])
            ddos_amp_subnet_idx, ddos_amp_intensity_box = ddos_amp_action_tuple
            ddos_amp_intensity = float(ddos_amp_intensity_box[0])
            if ddos_amp_subnet_idx < self.num_subnets:
                mal_pkts_amp = int(ddos_amp_intensity * self.MAX_MAL_AMP)
                mal_traffic[ddos_amp_subnet_idx]['UDP'] += mal_pkts_amp
                self.last_attack_feedback = {'status': 1, 'damage': mal_pkts_amp * 0.1}

            # AttemptToDisableDefenses (att_action[10])
            disable_defense_subnet_idx, disable_defense_type_idx = disable_defense_action_tuple
            if disable_defense_subnet_idx < self.num_subnets:
                # Placeholder: actual disabling logic would be complex (e.g., timers)
                self.last_attack_feedback = {'status': 0, 'damage': 1.0} # Minor effect for now

        # 3) Inicializar contadores para métricas
        total_filtered = 0
        total_ids_dropped = 0
        total_honeypot = 0
        total_rate_limited = 0
        total_defender_action_costs = 0.0

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
                # Unpack all defender actions (original 5 + 5 new)
                filter_host_choice, bw_limit_action_index, use_ids_toggle, use_dpi_toggle, honeypot_host_choice, \
                adv_firewall_rule_idx, deploy_honeynet_toggle, isolate_host_choice, \
                share_intel_toggle, patch_action_tuple = action
                
                # Original defender actions
                filter_host = filter_host_choice
                actual_bw_limit = self.bandwidth_levels[bw_limit_action_index] # Convert index to float value
                use_ids = bool(use_ids_toggle)
                use_dpi = bool(use_dpi_toggle)
                honeypot_host = honeypot_host_choice

                # New Defender Actions Processing
                # DeployAdvancedFirewallRule (action[5])
                # Rule k reduces all incoming malicious traffic by k * 0.05
                firewall_effect_multiplier = 1.0 - (adv_firewall_rule_idx * 0.05)
                mal_TCP *= firewall_effect_multiplier
                mal_UDP *= firewall_effect_multiplier
                mal_ICMP *= firewall_effect_multiplier
                
                # DeployHoneynet (action[6])
                if deploy_honeynet_toggle == 1:
                    if self.honeynet_status[i] == 0: # Cost only if deploying new
                        total_defender_action_costs += self.HONEYNET_DEPLOY_COST
                    self.honeynet_status[i] = 1
                
                # IsolateCompromisedHost (action[7])
                target_host_to_isolate = isolate_host_choice - 1 # 0 is no-op
                if target_host_to_isolate >= 0 and target_host_to_isolate < self.hosts_per_subnet[i]:
                    if target_host_to_isolate not in self.isolated_hosts[i]: # Avoid re-isolating
                        self.isolated_hosts[i].add(target_host_to_isolate)
                        # Proportional reduction in traffic if host is isolated
                        if self.hosts_per_subnet[i] > 0:
                            reduction_factor = 1.0 / self.hosts_per_subnet[i]
                            isolated_benign_traffic = (benign_TCP + benign_UDP + benign_ICMP) * reduction_factor
                            isolated_mal_traffic = (mal_TCP + mal_UDP + mal_ICMP) * reduction_factor
                            
                            benign_TCP -= benign_TCP * reduction_factor
                            benign_UDP -= benign_UDP * reduction_factor
                            benign_ICMP -= benign_ICMP * reduction_factor
                            mal_TCP -= mal_TCP * reduction_factor
                            mal_UDP -= mal_UDP * reduction_factor
                            mal_ICMP -= mal_ICMP * reduction_factor
                            total_filtered += (isolated_benign_traffic + isolated_mal_traffic)

                # ShareThreatIntelligence (action[8])
                if share_intel_toggle == 1:
                    self.shared_intelligence_pool.append({
                        'source_subnet': i, 
                        'type': 'generic_intel', 
                        'step': self.step_count
                    })

                # PatchVulnerability (action[9])
                patch_host_choice, patch_vuln_type_idx = patch_action_tuple
                target_host_to_patch = patch_host_choice - 1 # 0 is no-op
                if target_host_to_patch >= 0 and target_host_to_patch < self.hosts_per_subnet[i] and \
                   0 <= patch_vuln_type_idx < self.N_VULNERABILITY_TYPES: # Check vuln_type_idx bounds
                    if target_host_to_patch in self.host_vulnerabilities[i]: 
                         if self.host_vulnerabilities[i][target_host_to_patch][patch_vuln_type_idx] == 1: # Cost only if actually patching
                            total_defender_action_costs += self.PATCH_COST
                         self.host_vulnerabilities[i][target_host_to_patch][patch_vuln_type_idx] = 0 # Patched
                
                # --- Original Defender Logic (applied after new actions might have modified traffic) ---
                # Si filtrar host k (>0), eliminar todo tráfico de ese host
                if filter_host > 0:
                    host_idx = filter_host - 1 # This is the original filter_host logic
                    # Ensure host_idx is valid and not already isolated (though isolation already reduced traffic)
                    if host_idx < self.hosts_per_subnet[i] and host_idx not in self.isolated_hosts[i]:
                        # Asumimos que todo el tráfico de ese host está incluido en los totales beninos/malosos
                        # This is a simplification; true per-host traffic is not tracked.
                        # The impact of this filter might be on already reduced traffic if isolation happened.
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
                        parts = self.hosts_per_subnet[i]
                        if parts > 0: # Avoid division by zero if somehow subnet has no hosts
                            drop_TCP = benign_TCP/parts + mal_TCP/parts
                            drop_UDP = benign_UDP/parts + mal_UDP/parts
                            drop_ICMP = benign_ICMP/parts + mal_ICMP/parts
                        else:
                            drop_TCP = drop_UDP = drop_ICMP = 0
                        
                        benign_TCP = max(0, benign_TCP - drop_TCP) # Ensure not negative
                        mal_TCP = max(0, mal_TCP - drop_TCP)
                        benign_UDP = max(0, benign_UDP - drop_UDP)
                        mal_UDP = max(0, mal_UDP - drop_UDP)
                        benign_ICMP = max(0, benign_ICMP - drop_ICMP)
                        mal_ICMP = max(0, mal_ICMP - drop_ICMP)
                        total_filtered += (drop_TCP + drop_UDP + drop_ICMP)

                # Honeypot en host j (>0): desviamos todo tráfico de ese host
                if honeypot_host > 0: # This is the original honeypot_host logic
                    host_idx = honeypot_host - 1
                    # Ensure host_idx is valid and not isolated
                    if host_idx < self.hosts_per_subnet[i] and host_idx not in self.isolated_hosts[i]:
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
                        
                        mal_TCP = max(0, mal_TCP - cap_TCP)
                        mal_UDP = max(0, mal_UDP - cap_UDP)
                        mal_ICMP = max(0, mal_ICMP - cap_ICMP)
                        benign_TCP = max(0, benign_TCP - benign_loss)
                        benign_UDP = max(0, benign_UDP - benign_loss)
                        benign_ICMP = max(0, benign_ICMP - benign_loss)
                        # Contabilizar
                        total_honeypot += (cap_TCP + cap_UDP + cap_ICMP)
                
                # Aplicar limitación de ancho de banda en la subred
                total_sub_benign = benign_TCP + benign_UDP + benign_ICMP
                total_sub_mal = mal_TCP + mal_UDP + mal_ICMP
                # Se reduce ambos tipos de tráfico
                # lim = bw_limit # Old logic with bw_limit as float from Box
                drop_ben = total_sub_benign * actual_bw_limit
                drop_mal = total_sub_mal * actual_bw_limit
                benign_TCP *= (1-actual_bw_limit)
                benign_UDP *= (1-actual_bw_limit)
                benign_ICMP *= (1-actual_bw_limit)
                mal_TCP *= (1-actual_bw_limit)
                mal_UDP *= (1-actual_bw_limit)
                mal_ICMP *= (1-actual_bw_limit)
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

        # 5) Agregar contribuciones al servidor
        # Store delivered traffic per subnet for observation construction
        self.delivered_benign_sub_at_step_end = delivered_benign_sub
        self.delivered_malicious_sub_at_step_end = malicious_sub # Corrected variable name

        total_benign = sum(delivered_benign_sub.values())
        total_malicious = sum(malicious_sub.values()) # Corrected variable name
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
        
        # Store final total traffic for observation construction
        self.total_benign_traffic_at_step_end = total_benign
        self.total_malicious_traffic_at_step_end = total_malicious

        # 6) Construir observaciones y recompensas por agente
        # Calculate current_max_hosts_val for this step's _get_observations
        self.current_max_hosts_val = 0
        if self.hosts_per_subnet: # Should always be true in step
            self.current_max_hosts_val = max(self.hosts_per_subnet) if self.hosts_per_subnet else 0
        obs = self._get_observations()
        
        rewards = {}
        dones = {}

        # Defender Team Reward
        # total_benign_traffic_at_step_end and total_malicious_traffic_at_step_end are after server capacity
        global_defender_reward = self.total_benign_traffic_at_step_end \
                                 - (1.5 * self.total_malicious_traffic_at_step_end) \
                                 - total_defender_action_costs
        for i in range(self.num_subnets):
            rewards[f"def_{i}"] = global_defender_reward
            dones[f"def_{i}"] = False
        
        # Attacker Reward
        rewards["attacker"] = self.total_malicious_traffic_at_step_end # Malicious traffic that reached server
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
                "total_benign": self.total_benign_traffic_at_step_end, # After server capacity
                "total_malicious": self.total_malicious_traffic_at_step_end, # After server capacity
                "filtered": total_filtered,
                "ids_dropped": total_ids_dropped,
                "honeypot_captured": total_honeypot,
                "rate_limited": total_rate_limited,
                "server_load": self.server_load,
                "total_compromised_hosts": sum(len(self.compromised[s]) for s in range(self.num_subnets)),
                "honeynets_active": sum(self.honeynet_status),
                "shared_intel_items": len(self.shared_intelligence_pool),
                "total_defender_action_costs": total_defender_action_costs
            }
        }
        return obs, rewards, dones, info
