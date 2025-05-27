import argparse
import os
import time # For default run_name
from abc import ABC, abstractmethod
import gymnasium as gym # For action_space
from torch.utils.tensorboard import SummaryWriter # For TensorBoard logging
import csv # For CSV logging
import collections # For action counting
import numpy as np # For np.mean in live plotter and MA calculation
import matplotlib.pyplot as plt # For LivePlotter
from cyber_env import CyberSecurityEnv # Assuming cyber_env.py is in the same directory or PYTHONPATH

# Agent Interface
class BaseAgent(ABC):
    def __init__(self, action_space, agent_id=None):
        self.action_space = action_space
        self.agent_id = agent_id

    @abstractmethod
    def get_action(self, observation, env_instance=None): # Renamed env to env_instance
        pass

    def learn(self, experience):
        print(f"Agent {self.agent_id or type(self).__name__}: Learn method called (placeholder). Experience: {experience}")

    def save_model(self, path):
        print(f"Agent {self.agent_id or type(self).__name__}: Save model to {path} (placeholder).")

    def load_model(self, path):
        print(f"Agent {self.agent_id or type(self).__name__}: Load model from {path} (placeholder).")

# Concrete Agent Implementations
class RandomAgent(BaseAgent):
    def __init__(self, action_space, agent_id=None):
        super().__init__(action_space, agent_id)

    def get_action(self, observation, env_instance=None):
        return self.action_space.sample()

class VicomRL_DQNAgent(BaseAgent): # Renamed from CustomRLAgent
    def __init__(self, observation_space, action_space, agent_id_for_library=None, rl_config_args=None, chkpt_dir_base='./models'):
        super().__init__(action_space, agent_id_for_library)
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.n # Assuming Discrete action space for individual RL agents
        self.rl_config_args = rl_config_args
        self.chkpt_dir = os.path.join(chkpt_dir_base, agent_id_for_library or 'dqn_agent')
        os.makedirs(self.chkpt_dir, exist_ok=True)
        
        dqn_args = {
            'lr': rl_config_args.learning_rate, 
            'gamma': rl_config_args.qmix_gamma, # Using qmix_gamma for DQN too, can be separated
            'batch_size': rl_config_args.batch_size
            # Add other relevant args from rl_config_args or defaults
        }
        print(f"VicomRL_DQNAgent ({self.agent_id}): Initializing with obs_dim={self.obs_dim}, act_dim={self.act_dim}, chkpt_dir={self.chkpt_dir}, args_def={dqn_args} (Placeholder: from agents.DQN.AgentsDQN import DQNAgent; self.policy = DQNAgent(dqn_args, self.act_dim, self.obs_dim, self.chkpt_dir))")
        
        model_path_to_load = None
        if self.agent_id == "attacker" and rl_config_args.attacker_model_path:
            model_path_to_load = rl_config_args.attacker_model_path
        elif "def_" in (self.agent_id or "") and rl_config_args.defender_model_path: # For collaborative defenders
            model_path_to_load = rl_config_args.defender_model_path # May need adjustment if paths are agent-specific

        if model_path_to_load:
            self.load_model(model_path_to_load)

    def get_action(self, observation, env_instance=None):
        print(f"VicomRL_DQNAgent ({self.agent_id}): Get action (Placeholder: action = self.policy.choose_action(observation) if hasattr(self,'policy') else self.action_space.sample())")
        return self.action_space.sample() # Placeholder

    def learn(self, experience):
        print(f"VicomRL_DQNAgent ({self.agent_id}): Learn (Placeholder: self.policy.store_transition(...) and self.policy.learn() if hasattr(self,'policy'))")

    def save_model(self, path=None):
        eff_path = path or self.chkpt_dir
        print(f"VicomRL_DQNAgent ({self.agent_id}): Save model to {eff_path} (Placeholder: self.policy.save_models() if hasattr(self,'policy'))")

    def load_model(self, path=None):
        eff_path = path or self.chkpt_dir
        print(f"VicomRL_DQNAgent ({self.agent_id}): Load model from {eff_path} (Placeholder: self.policy.load_models() if hasattr(self,'policy'))")


class VicomRL_QMIXDefenders(BaseAgent): # Renamed from CentralizedDefenderRLAgent
    def __init__(self, env, model_path=None, rl_config_args=None, chkpt_dir_base='./models'):
        self.defender_ids = sorted([agent_id for agent_id in env.possible_agents if "def_" in agent_id])
        self.n_agents = len(self.defender_ids)
        self.obs_dim = env.defender_observation_space.shape[0] # Individual defender obs_dim
        
        # QMIX typically deals with a set of discrete actions per agent.
        # The environment's defender_action_space is a Tuple.
        # For now, using a placeholder value for self.act_dim for the QMIX internal agent model.
        # The mapping from this simplified QMIX action output to the env's Tuple action needs to be handled.
        self.act_dim = 10 # Placeholder: e.g., 10 abstract actions QMIX chooses from per defender
        
        # Placeholder for global state dimension
        self.state_dim = self.n_agents * self.obs_dim + 1 (server_load) + 1 (total_compromised)
        
        self.chkpt_dir = os.path.join(chkpt_dir_base, 'qmix_defenders')
        os.makedirs(self.chkpt_dir, exist_ok=True)
        
        qmix_params = {
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'state_dim': self.state_dim,
            'action_dim': self.act_dim, # Simplified action dim for QMIX internal model
            'qmix_vdn': rl_config_args.qmix_vdn,
            'qmix_hidden_dim': rl_config_args.qmix_hidden_dim,
            'qmix_mixing_embed_dim': rl_config_args.qmix_mixing_embed_dim,
            'qmix_buffer_capacity': rl_config_args.qmix_buffer_capacity,
            'gamma': rl_config_args.qmix_gamma,
            'lr': rl_config_args.learning_rate, # QMIX might use a specific LR from its args
            'batch_size': rl_config_args.batch_size,
            'target_update_freq': rl_config_args.qmix_target_update_freq,
            'epsilon_start': rl_config_args.qmix_epsilon_start,
            'epsilon_end': rl_config_args.qmix_epsilon_end,
            'epsilon_decay_period': rl_config_args.qmix_epsilon_decay_period,
            'chkpt_dir': self.chkpt_dir
        }
        print(f"VicomRL_QMIXDefenders: Initializing QMIX with n_agents={self.n_agents}, obs_dim={self.obs_dim}, state_dim={self.state_dim}, act_dim={self.act_dim}, params={qmix_params} (Placeholder: from agents.QMIX... import QMIX, QMIXTrainer; self.qmix_controller = QMIX(...); self.qmix_trainer = QMIXTrainer(...))")

        if model_path: # This comes from rl_config_args.defender_model_path
            self.load_model(model_path)
        
        # The action space for the BaseAgent is what this agent *outputs* to the environment
        super().__init__(gym.spaces.Dict({d_id: env.defender_action_space for d_id in self.defender_ids}), "qmix_defenders_group")

    def _construct_global_state(self, obs_dict, env):
        # Example global state: concatenate all defender obs, server load, and total compromised hosts
        # Ensure consistent order for obs_dict values if keys are not sorted by default elsewhere
        flat_obs_list = [obs_dict[d_id] for d_id in self.defender_ids if d_id in obs_dict]
        state_parts = [np.concatenate(flat_obs_list)] if flat_obs_list else [np.array([])]
        state_parts.append(np.array([env.server_load], dtype=np.float32))
        state_parts.append(np.array([sum(len(env.compromised[s]) for s in range(env.num_subnets))], dtype=np.float32))
        return np.concatenate(state_parts).astype(np.float32)

    def get_action(self, observation_dict, env_instance=None): # observation_dict for QMIX contains individual defender obs
        obs_list = [observation_dict[d_id] for d_id in self.defender_ids if d_id in observation_dict]
        state_vector = self._construct_global_state(observation_dict, env_instance)
        print(f"VicomRL_QMIXDefenders: Get actions (Placeholder: actions_indices = self.qmix_controller.choose_actions(obs_list, state_vector); map indices to actual Tuple actions for each agent)")
        # Placeholder: Return random Tuple action for each defender
        # This part needs a mapping from simplified QMIX action (e.g., discrete 0-9) to the complex Tuple action.
        return {def_id: env_instance.defender_action_space.sample() for def_id in self.defender_ids if def_id in observation_dict}

    def learn(self, experience_batch):
        print(f"VicomRL_QMIXDefenders ({self.agent_id}): Learn (Placeholder: self.qmix_trainer.train(experience_batch))")

    def save_model(self, path=None):
        eff_path = path or self.chkpt_dir
        print(f"VicomRL_QMIXDefenders ({self.agent_id}): Save model to {eff_path} (Placeholder: self.qmix_controller.save_models())")

    def load_model(self, path=None):
        eff_path = path or self.chkpt_dir
        print(f"VicomRL_QMIXDefenders ({self.agent_id}): Load model from {eff_path} (Placeholder: self.qmix_controller.load_models())")


# Agent Factory
def create_agents(cli_args, env): # cli_args are the parsed arguments
    attacker_agent = None
    defender_payload = {} # Will be a dict for collaborative, or a single agent for centralized/random_group

    # Create Attacker
    if cli_args.attacker_type == 'random':
        attacker_agent = RandomAgent(env.attacker_action_space, agent_id="attacker")
    elif cli_args.attacker_type == 'rl':
        attacker_agent = VicomRL_DQNAgent( # Changed from CustomRLAgent
            env.attacker_observation_space,
            env.attacker_action_space,
            agent_id_for_library="attacker",
            rl_config_args=cli_args,
            chkpt_dir_base=os.path.join(cli_args.log_dir, 'models') # Save models under the run's log_dir
        )
    else:
        raise ValueError(f"Unsupported attacker_type: {cli_args.attacker_type}")

    # Create Defenders
    active_defender_ids = [agent_id for agent_id in env.agents if "def_" in agent_id]
    if not active_defender_ids and cli_args.defender_type not in ['centralized_rl']: 
        print("Warning: No active defender agents found in env.agents. Fallback to max_subnets for setup if possible.")
        active_defender_ids = [f"def_{i}" for i in range(env.max_subnets)]


    if cli_args.defender_type == 'random':
        for agent_id in active_defender_ids:
            defender_payload[agent_id] = RandomAgent(env.defender_action_space, agent_id=agent_id)
    elif cli_args.defender_type == 'collaborative_rl': # Individual DQN agents for defenders
        for agent_id in active_defender_ids:
            defender_payload[agent_id] = VicomRL_DQNAgent( # Changed from CustomRLAgent
                env.defender_observation_space, 
                env.defender_action_space, # This is Tuple, DQN needs Discrete. This is a MISMATCH to address.
                                           # For now, VicomRL_DQNAgent assumes Discrete action_space.n
                                           # This will likely CRASH if env.defender_action_space is Tuple.
                                           # Temporary fix: pass a simplified discrete space for placeholder
                agent_id_for_library=agent_id,
                rl_config_args=cli_args,
                chkpt_dir_base=os.path.join(cli_args.log_dir, 'models')
            )
    elif cli_args.defender_type == 'centralized_rl': # QMIX for defenders
        defender_payload = VicomRL_QMIXDefenders( # Changed from CentralizedDefenderRLAgent
            env,
            model_path=cli_args.defender_model_path, # This is the model path for the whole QMIX model
            rl_config_args=cli_args,
            chkpt_dir_base=os.path.join(cli_args.log_dir, 'models')
        )
    else:
        raise ValueError(f"Unsupported defender_type: {cli_args.defender_type}")

    return attacker_agent, defender_payload

# Action Counting Function
def get_action_type_counts(action_dict, defender_ids, num_subnets_for_attacker_check):
    counts = collections.defaultdict(int)
    
    # Attacker Action Counting
    attacker_action = action_dict.get("attacker")
    if attacker_action is not None:
        # Original DoS-like attack: (subnet, host, proto, intensity, freq, spoof, evasion)
        if attacker_action[3][0] > 0 or attacker_action[4][0] > 0: # intensity or freq > 0
            counts['attacker_std_attack'] += 1
        
        # Reconnaissance: (subnet_target_choice) - index 7
        # 0 is no-op, >0 means a subnet is targeted
        if attacker_action[7] > 0:
            counts['attacker_recon'] += 1
            
        # Exploit: (subnet_idx, host_idx, vuln_type_idx) - index 8
        # Count if a valid subnet is targeted (subnet_idx < num_subnets)
        exploit_target_subnet = attacker_action[8][0]
        if exploit_target_subnet < num_subnets_for_attacker_check : # Assumes num_subnets is passed correctly
             counts['attacker_exploit'] +=1

        # DDoS Amplification: (target_subnet_idx, intensity_multiplier_box) - index 9
        if attacker_action[9][1][0] > 0: # intensity > 0
            counts['attacker_ddos_amp'] += 1
            
        # Disable Defenses: (target_subnet_idx, defense_type_idx) - index 10
        disable_target_subnet = attacker_action[10][0]
        if disable_target_subnet < num_subnets_for_attacker_check:
            counts['attacker_disable_defense'] += 1

    # Defender Action Counting
    for def_id in defender_ids:
        def_action = action_dict.get(def_id)
        if def_action is not None:
            # Original actions
            if def_action[0] > 0: counts['defender_filter_host'] += 1
            if def_action[1] > 0: counts['defender_bw_limit'] += 1 # Index 0 is 0.0 limit
            if def_action[2] == 1: counts['defender_ids_on'] += 1
            if def_action[3] == 1: counts['defender_dpi_on'] += 1
            if def_action[4] > 0: counts['defender_honeypot_host'] += 1
            # New actions
            if def_action[5] > 0: counts['defender_adv_firewall'] += 1 # Assuming 0 is "no specific rule"
            if def_action[6] == 1: counts['defender_deploy_honeynet'] += 1
            if def_action[7] > 0: counts['defender_isolate_host'] += 1
            if def_action[8] == 1: counts['defender_share_intel'] += 1
            if def_action[9][0] > 0: counts['defender_patch_vuln'] += 1 # Patch host_idx > 0
            
    return counts

def main(args):
    if args.run_name is None:
        args.run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Update log_dir to include the specific run_name
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(args.log_dir, exist_ok=True) # Ensure the directory is created
    
    print(f"Training run: {args.run_name}")
    print(f"Log directory: {args.log_dir}")
    print("Configuration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    # Setup Loggers
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    csv_file_path = os.path.join(args.log_dir, 'episode_metrics.csv')
    csv_headers = [
        'episode',
        'total_reward_attacker',
        'total_reward_defenders_team', # Team reward is now a single value
        'avg_reward_defenders_team_last_100',
        'benign_traffic_total_server', # Traffic that reached server
        'malicious_traffic_total_server', # Traffic that reached server
        'server_load_avg_episode',
        'compromised_hosts_total_final',
        'defender_action_costs_total_episode',
        # Attacker action counts
        'attacker_std_attack_count', 'attacker_recon_count', 'attacker_exploit_count',
        'attacker_ddos_amp_count', 'attacker_disable_defense_count',
        # Defender action counts (summed over all defenders)
        'defender_filter_host_count', 'defender_bw_limit_count', 'defender_ids_on_count',
        'defender_dpi_on_count', 'defender_honeypot_host_count', 'defender_adv_firewall_count',
        'defender_deploy_honeynet_count', 'defender_isolate_host_count',
        'defender_share_intel_count', 'defender_patch_vuln_count'
    ]
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    csv_writer.writeheader()
    print(f"TensorBoard logs and CSV metrics will be saved to: {args.log_dir}")
    
    env = CyberSecurityEnv()
    # Important: Call env.reset() to initialize num_subnets, hosts_per_subnet, possible_agents, etc.
    # which are needed by create_agents for some agent types and by the env itself.
    initial_obs, initial_info = env.reset() 

    attacker, defenders = create_agents(args, env)

    print(f"\nAttacker agent: {type(attacker).__name__} (ID: {attacker.agent_id})")
    if isinstance(defenders, dict):
        defender_types = {k: type(v).__name__ for k,v in defenders.items()}
        defender_ids = {k: v.agent_id for k,v in defenders.items()}
        print(f"Defender agents: {defender_types} (IDs: {defender_ids})")
    else: # For centralized agent
        print(f"Defender agent: {type(defenders).__name__} (ID: {defenders.agent_id}, Controls: {defenders.defender_ids})")

    # Further implementation will follow in subsequent steps (training loop)
    print("\nAgent setup complete. Starting training loop...")

    live_plotter = LivePlotter(args.num_episodes, args.disable_live_plots, args.log_dir)

    # Data storage for moving averages
    defender_team_rewards_history_for_ma = []
    MA_WINDOW = 100 # Moving average window

    try:
        for episode in range(args.num_episodes):
            obs_dict, info = env.reset()
            
            # Separate attacker and defender observations
            attacker_obs = obs_dict.pop("attacker") 
            # obs_dict now primarily contains defender observations if not empty,
            # or is the full dict for the centralized controller if it needs all obs.
            # For VicomRL_QMIXDefenders, it expects the dict of individual defender obs.
            
            episode_reward_attacker = 0.0
            episode_reward_defenders_team = 0.0 # Store the team reward once per step
            episode_benign_traffic_sum = 0.0
            episode_malicious_traffic_sum = 0.0
            episode_server_load_sum = 0.0
            episode_defender_action_costs = 0.0
            episode_action_counts = collections.defaultdict(int)

            max_steps_per_episode = getattr(env, 'max_steps', 200) 

            for step_num in range(max_steps_per_episode):
                # 1. Get actions from agents
                action_dict = {}
                action_dict["attacker"] = attacker.get_action(attacker_obs, env_instance=env)

                if isinstance(defenders, dict): # Collaborative or Random defenders
                    for def_id, def_agent in defenders.items():
                        if def_id in obs_dict: 
                             action_dict[def_id] = def_agent.get_action(obs_dict[def_id], env_instance=env)
                else: # Centralized defender (e.g., QMIX)
                    defender_actions = defenders.get_action(obs_dict, env_instance=env) 
                    action_dict.update(defender_actions)
                
                final_action_dict = {k: v for k, v in action_dict.items() if k in env.agents}

                # 2. Step the environment
                next_obs_dict, rewards, dones, truncated_dict, info = env.step(final_action_dict)
                done = dones["__all__"] or truncated_dict["__all__"] # Use "truncated" from gym API

                # 3. Store experience and trigger learning (simplified placeholders)
                attacker.learn({'obs': attacker_obs, 'action': final_action_dict.get("attacker"), 'reward': rewards.get("attacker"), 'next_obs': next_obs_dict.get("attacker"), 'done': done})
                
                if isinstance(defenders, dict):
                    for def_id, def_agent in defenders.items():
                        if def_id in obs_dict: # If agent was active
                            def_agent.learn({'obs': obs_dict.get(def_id), 'action': final_action_dict.get(def_id), 'reward': rewards.get(def_id), 'next_obs': next_obs_dict.get(def_id), 'done': done })
                else: # Centralized
                     # QMIX learn typically takes a batch of (state, actions, rewards, next_state, dones_list, global_state, next_global_state)
                     # For placeholder, just passing rewards and dones.
                    defenders.learn({'rewards': rewards, 'dones': dones})


                # 4. Accumulate episode metrics
                episode_reward_attacker += rewards.get("attacker", 0.0)
                
                first_defender_key = next((k for k in rewards if "def_" in k), None)
                if first_defender_key:
                    current_step_defender_team_reward = rewards[first_defender_key]
                    episode_reward_defenders_team += current_step_defender_team_reward
                
                # Accumulate sums from per-step metrics
                episode_benign_traffic_sum += info["metrics"].get("total_benign", 0.0)
                episode_malicious_traffic_sum += info["metrics"].get("total_malicious", 0.0)
                episode_server_load_sum += info["metrics"].get("server_load", 0.0)
                episode_defender_action_costs += info["metrics"].get("total_defender_action_costs", 0.0)

                # Aggregate action counts for the episode
                active_defender_ids_this_step = [agent_id for agent_id in env.agents if "def_" in agent_id] # Get current defenders
                step_action_counts = get_action_type_counts(final_action_dict, active_defender_ids_this_step, env.num_subnets)
                for key, value in step_action_counts.items():
                    episode_action_counts[key] += value
                
                # 5. Prepare for next step
                attacker_obs = next_obs_dict.pop("attacker")
                obs_dict = next_obs_dict 

                if done:
                    break
            
            # End of episode
            # 6. Log to TensorBoard & CSV
            avg_server_load = episode_server_load_sum / (step_num + 1)
            
            defender_team_rewards_history_for_ma.append(episode_reward_defenders_team)
            if len(defender_team_rewards_history_for_ma) > MA_WINDOW:
                defender_team_rewards_history_for_ma.pop(0)
            avg_reward_def_ma = np.mean(defender_team_rewards_history_for_ma) if defender_team_rewards_history_for_ma else 0.0

            # Use final values from info["metrics"] for traffic as they reflect server capacity effects
            final_benign_traffic = info["metrics"].get("total_benign", 0.0)
            final_malicious_traffic = info["metrics"].get("total_malicious", 0.0)
            final_compromised_hosts = info["metrics"].get("total_compromised_hosts", 0)

            tb_writer.add_scalar('Reward/Attacker', episode_reward_attacker, episode)
            tb_writer.add_scalar('Reward/DefenderTeam', episode_reward_defenders_team, episode)
            tb_writer.add_scalar('Reward/DefenderTeam_MA100', avg_reward_def_ma, episode)
            tb_writer.add_scalar('Traffic/Benign_Total_Server_EpisodeEnd', final_benign_traffic, episode)
            tb_writer.add_scalar('Traffic/Malicious_Total_Server_EpisodeEnd', final_malicious_traffic, episode)
            tb_writer.add_scalar('Performance/Avg_Server_Load_Episode', avg_server_load, episode)
            tb_writer.add_scalar('Performance/Total_Compromised_Hosts_EpisodeEnd', final_compromised_hosts, episode)
            tb_writer.add_scalar('Costs/Defender_Action_Costs_Episode', episode_defender_action_costs, episode)
            
            # Log action counts to TensorBoard
            attacker_counts_for_tb = {k.replace('attacker_', ''): v for k, v in episode_action_counts.items() if 'attacker_' in k}
            defender_counts_for_tb = {k.replace('defender_', ''): v for k, v in episode_action_counts.items() if 'defender_' in k}
            if attacker_counts_for_tb:
                tb_writer.add_scalars('ActionCounts/Attacker', attacker_counts_for_tb, episode)
            if defender_counts_for_tb:
                tb_writer.add_scalars('ActionCounts/Defender', defender_counts_for_tb, episode)

            csv_row = {
                'episode': episode + 1,
                'total_reward_attacker': episode_reward_attacker,
                'total_reward_defenders_team': episode_reward_defenders_team,
                'avg_reward_defenders_team_last_100': avg_reward_def_ma,
                'benign_traffic_total_server': final_benign_traffic,
                'malicious_traffic_total_server': final_malicious_traffic,
                'server_load_avg_episode': avg_server_load,
                'compromised_hosts_total_final': final_compromised_hosts,
                'defender_action_costs_total_episode': episode_defender_action_costs,
                # Add action counts to CSV row
                'attacker_std_attack_count': episode_action_counts.get('attacker_std_attack', 0),
                'attacker_recon_count': episode_action_counts.get('attacker_recon', 0),
                'attacker_exploit_count': episode_action_counts.get('attacker_exploit', 0),
                'attacker_ddos_amp_count': episode_action_counts.get('attacker_ddos_amp', 0),
                'attacker_disable_defense_count': episode_action_counts.get('attacker_disable_defense', 0),
                'defender_filter_host_count': episode_action_counts.get('defender_filter_host', 0),
                'defender_bw_limit_count': episode_action_counts.get('defender_bw_limit', 0),
                'defender_ids_on_count': episode_action_counts.get('defender_ids_on', 0),
                'defender_dpi_on_count': episode_action_counts.get('defender_dpi_on', 0),
                'defender_honeypot_host_count': episode_action_counts.get('defender_honeypot_host', 0),
                'defender_adv_firewall_count': episode_action_counts.get('defender_adv_firewall', 0),
                'defender_deploy_honeynet_count': episode_action_counts.get('defender_deploy_honeynet', 0),
                'defender_isolate_host_count': episode_action_counts.get('defender_isolate_host', 0),
                'defender_share_intel_count': episode_action_counts.get('defender_share_intel', 0),
                'defender_patch_vuln_count': episode_action_counts.get('defender_patch_vuln', 0)
            }
            csv_writer.writerow(csv_row)

            # 7. Update Live Plot
            if not args.disable_live_plots:
                live_plotter.update(
                    episode + 1, 
                    episode_reward_attacker, 
                    episode_reward_defenders_team, 
                    avg_reward_def_ma, 
                    final_benign_traffic, 
                    final_malicious_traffic
                )
            
            print(f"Episode {episode + 1}/{args.num_episodes} completed. Steps: {step_num+1}. Attacker Reward: {episode_reward_attacker:.2f}, Defender Team Reward: {episode_reward_defenders_team:.2f}, MA Defender: {avg_reward_def_ma:.2f}")

            # 8. Save models at interval
            if (episode + 1) % args.save_model_interval == 0:
                attacker.save_model() 
                if isinstance(defenders, dict):
                    for def_agent in defenders.values():
                        def_agent.save_model()
                else: 
                    defenders.save_model()
                print(f"Models saved at episode {episode + 1}")
        
        # Post-training tasks
        print("Training loop finished. Saving final models...")
        attacker.save_model() 
        if isinstance(defenders, dict):
            for def_agent in defenders.values():
                def_agent.save_model()
        else: # Centralized agent
            defenders.save_model()
        print("Final models saved.")

    finally: # Ensure loggers and plotters are closed
        csv_file.close()
        tb_writer.close()
        if 'live_plotter' in locals() : # live_plotter is always defined now
            live_plotter.close()
        print("Training finished. Resources closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Launch Script for ServerDoS Environment")

    # Attacker arguments
    parser.add_argument('--attacker_type', type=str, default='random', choices=['random', 'rl'], 
                        help="Type of attacker agent.")
    parser.add_argument('--attacker_model_path', type=str, default=None, 
                        help="Path to pre-trained attacker model (if attacker_type is 'rl').")

    # Defender arguments
    parser.add_argument('--defender_type', type=str, default='random', 
                        choices=['random', 'centralized_rl', 'collaborative_rl'], 
                        help="Type of defender agent/s.")
    parser.add_argument('--defender_model_path', type=str, default=None, 
                        help="Path to pre-trained defender model(s).")

    # Training loop arguments
    parser.add_argument('--num_episodes', type=int, default=1000, 
                        help="Number of episodes to train.")
    parser.add_argument('--log_dir', type=str, default='./training_logs', 
                        help="Base directory to save TensorBoard logs and CSV files.")
    parser.add_argument('--run_name', type=str, default=None, 
                        help="Optional run name for logging, defaults to a timestamped name. Logs will be in log_dir/run_name.")
    parser.add_argument('--disable_live_plots', action='store_true', 
                        help="Disable live plotting during training.") # Default is False

    # RL agent specific arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help="Learning rate for RL agents.")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size for RL agent training.")
    parser.add_argument('--save_model_interval', type=int, default=100, 
                        help="Interval (in episodes) for saving RL models.")
                        
    # QMIX specific arguments
    parser.add_argument('--qmix_vdn', action='store_true', 
                        help="Use VDN architecture for QMIX defenders.")
    parser.add_argument('--qmix_hidden_dim', type=int, default=64, 
                        help="Hidden dimension for QMIX networks.")
    parser.add_argument('--qmix_mixing_embed_dim', type=int, default=32, 
                        help="Mixing embed dimension for QMIX.")
    parser.add_argument('--qmix_buffer_capacity', type=int, default=5000, 
                        help="Replay buffer capacity for QMIX.")
    parser.add_argument('--qmix_gamma', type=float, default=0.99, 
                        help="Discount factor (gamma) for QMIX.")
    parser.add_argument('--qmix_target_update_freq', type=int, default=200, 
                        help="Target network update frequency for QMIX (e.g., in steps or episodes).")
    parser.add_argument('--qmix_epsilon_start', type=float, default=1.0, 
                        help="Initial epsilon for QMIX exploration.")
    parser.add_argument('--qmix_epsilon_end', type=float, default=0.05, 
                        help="Final epsilon for QMIX exploration.")
    parser.add_argument('--qmix_epsilon_decay_period', type=int, default=50000, 
                        help="Period over which epsilon decays for QMIX (e.g., in steps).")

    args = parser.parse_args()
    main(args)


# Live Plotter Class (Optional, for visualization during training)
class LivePlotter:
    def __init__(self, num_episodes, disable_live_plots=False, log_dir='.'):
        self.disable = disable_live_plots
        self.log_dir = log_dir # Store log_dir for saving the plot
        if self.disable:
            return

        plt.ion() # Turn on interactive mode
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.tight_layout(pad=3.0) # Add padding between subplots

        # Initialize data storage for plots
        self.episode_numbers = []
        self.attacker_rewards = []
        self.defender_team_rewards = []
        self.defender_rewards_ma = [] # Moving average for defender team
        self.benign_traffic = []
        self.malicious_traffic = []

        # Setup Attacker vs Defender Rewards Plot
        self.axes[0].set_title('Agent Rewards')
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Total Reward per Episode')
        self.line_attacker_rew, = self.axes[0].plot([], [], label='Attacker Reward', color='red')
        self.line_defender_team_rew, = self.axes[0].plot([], [], label='Defender Team Reward', color='blue')
        self.line_defender_ma, = self.axes[0].plot([], [], label='Defender MA (100ep)', color='cyan', linestyle='--')
        self.axes[0].legend(loc='upper left')

        # Setup Traffic Plot
        self.axes[1].set_title('Traffic Reaching Server')
        self.axes[1].set_xlabel('Episode')
        self.axes[1].set_ylabel('Total Packets per Episode')
        self.line_benign_traffic, = self.axes[1].plot([], [], label='Benign Traffic', color='green')
        self.line_malicious_traffic, = self.axes[1].plot([], [], label='Malicious Traffic', color='magenta')
        self.axes[1].legend(loc='upper left')
        
        # Setup Additional Metrics Plot (Example: Compromised Hosts, can be customized)
        # This plot was in test.py, adapting for training context
        self.axes[2].set_title('Training Performance Metrics')
        self.axes[2].set_xlabel('Episode')
        self.axes[2].set_ylabel('Count / Value')
        # Example lines, can be changed based on what's most useful to plot live from training
        self.line_compromised_hosts, = self.axes[2].plot([], [], label='Compromised Hosts (Final)', color='orange')
        self.line_defender_costs, = self.axes[2].plot([], [], label='Defender Action Costs', color='purple')
        self.axes[2].legend(loc='upper left')


    def update(self, episode_num, attacker_reward, defender_team_reward, defender_ma, 
                 benign_traffic_server, malicious_traffic_server,
                 # Add new metrics as needed for the third plot
                 final_compromised_hosts=None, total_defender_action_costs=None 
                 ):
        if self.disable:
            return

        self.episode_numbers.append(episode_num)
        self.attacker_rewards.append(attacker_reward)
        self.defender_team_rewards.append(defender_team_reward)
        self.defender_rewards_ma.append(defender_ma)
        self.benign_traffic.append(benign_traffic_server)
        self.malicious_traffic.append(malicious_traffic_server)

        # Update rewards plot
        self.line_attacker_rew.set_data(self.episode_numbers, self.attacker_rewards)
        self.line_defender_team_rew.set_data(self.episode_numbers, self.defender_team_rewards)
        self.line_defender_ma.set_data(self.episode_numbers, self.defender_rewards_ma)
        self.axes[0].relim()
        self.axes[0].autoscale_view(True,True,True)

        # Update traffic plot
        self.line_benign_traffic.set_data(self.episode_numbers, self.benign_traffic)
        self.line_malicious_traffic.set_data(self.episode_numbers, self.malicious_traffic)
        self.axes[1].relim()
        self.axes[1].autoscale_view(True,True,True)
        
        # Update additional metrics plot (example)
        if final_compromised_hosts is not None and total_defender_action_costs is not None:
             # Assuming these are single values per episode for plotting
            current_compromised = getattr(self, 'compromised_hosts_data', [])
            current_costs = getattr(self, 'defender_costs_data', [])
            current_compromised.append(final_compromised_hosts)
            current_costs.append(total_defender_action_costs)
            self.compromised_hosts_data = current_compromised
            self.defender_costs_data = current_costs
            self.line_compromised_hosts.set_data(self.episode_numbers, self.compromised_hosts_data)
            self.line_defender_costs.set_data(self.episode_numbers, self.defender_costs_data)
            self.axes[2].relim()
            self.axes[2].autoscale_view(True,True,True)


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.disable:
            return
        
        if hasattr(self, 'fig'): # Check if figure was created
            save_path = os.path.join(self.log_dir, 'final_training_plots.png')
            self.fig.savefig(save_path)
            print(f"Final plots saved to {save_path}")

            plt.ioff() # Turn off interactive mode
            plt.show() # Keep plots open until manually closed
