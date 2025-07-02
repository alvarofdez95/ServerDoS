# ServerDoS Cybersecurity Simulation Environment

## Overview

ServerDoS is a multi-agent reinforcement learning (MARL) environment built using the `gymnasium` library. It simulates a dynamic network environment where defender agents protect their assigned subnets and a central server, while a global attacker agent attempts to compromise systems and disrupt services. The general goal is to provide a platform for studying and developing strategies for cyber attack and defense scenarios.

## Environment Details

*   **Agents & Objectives**:
    *   **Defenders**: One per subnet. Their primary objective is to protect their local subnet from malicious activities, minimize damage to hosts, and ensure the successful delivery of benign traffic to the central server. They operate under a collaborative team-based reward system.
    *   **Attacker**: A single global agent. Its objectives include compromising hosts across various subnets, disrupting the central server (e.g., by overwhelming its capacity), and evading detection by defender mechanisms.
*   **Key Components**:
    *   **Central Server**: Has a limited processing capacity. If traffic (benign + malicious) exceeds this, packets are dropped.
    *   **Subnets**: Multiple subnets, each containing a variable number of hosts.
    *   **Hosts**: Can be benign or compromised. Hosts also have specific vulnerabilities that can be exploited by the attacker and patched by defenders.
*   **Dynamic Elements**:
    *   The network topology (number of subnets, hosts per subnet) is randomized at the start of each episode.
    *   The initial set of compromised hosts and their vulnerabilities are also randomized per episode.

## Agent Capabilities (Key Actions & Observations)

### Defenders

*   **Actions (High-Level)**:
    *   Traffic Filtering: Implement basic host-based filtering or deploy more advanced, predefined firewall rules.
    *   Bandwidth Limiting: Apply rate limits to traffic in their subnet (discretized levels).
    *   Detection Systems: Activate/deactivate Intrusion Detection Systems (IDS) and Deep Packet Inspection (DPI).
    *   Deception: Deploy single honeypots or more extensive honeynets.
    *   Incident Response: Isolate compromised hosts.
    *   Proactive Defense: Patch specific vulnerabilities on hosts.
    *   Collaboration: Share threat intelligence with other defenders.
*   **Observations (High-Level)**:
    *   Local traffic patterns (benign vs. malicious).
    *   Vulnerability status of hosts within their subnet.
    *   Alerts from IDS/IPS systems (placeholder).
    *   Intelligence shared by other defenders.
    *   Activity reports from deployed honeynets (placeholder).
*   **Reward**:
    *   Defenders receive a shared team reward. This global reward is calculated based on the total benign traffic successfully delivered, penalized by malicious traffic reaching the server and the costs associated with defensive actions taken (e.g., patching, deploying honeynets).

### Attacker

*   **Actions (High-Level)**:
    *   Standard DoS Attacks: Generate malicious traffic towards specific targets.
    *   Network Reconnaissance: Scan subnets to gather intelligence on vulnerabilities and defenses.
    *   Vulnerability Exploitation: Target specific vulnerabilities on hosts.
    *   DDoS Amplification: Launch high-volume attacks.
    *   Evasion & Disruption: Attempt to disable or evade defensive measures (placeholder effect).
*   **Observations (High-Level)**:
    *   Results from reconnaissance actions (e.g., aggregated vulnerability scores per subnet, estimated defense postures).
    *   Feedback on the success/failure and damage of recent attacks.
    *   Global server load and an estimate of average subnet saturation.

## File Structure

*   `cyber_env.py`: Contains the core `CyberSecurityEnv` class definition, detailing the environment's logic, action/observation spaces, state transitions, and reward calculations.
*   `test.py`: A script to run simulation episodes. It uses random policies for all agents to demonstrate environment interaction and visualizes key metrics (e.g., traffic types, rewards, number of compromised hosts, active honeynets) using `matplotlib`.
*   `train.py`: A template script that outlines the setup for training multi-agent reinforcement learning agents (e.g., using algorithms like QMIX or VDN). It currently employs random action selection and serves as a starting point for implementing actual learning algorithms.
*   `README.md`: This file.
*   `launch_train.py`: A configurable script for running training sessions, managing agent types, logging, and (placeholder) RL model training.

## Setup and Dependencies

*   Python 3.8+
*   Main Python package dependencies:
    *   `gymnasium`
    *   `numpy`
    *   `matplotlib`
    *   `torch` (for TensorBoard `SummaryWriter`)
*   It is recommended to set up a virtual environment and install dependencies, potentially from a `requirements.txt` file (not currently provided, but can be created from imports).

## How to Run

*   **Test Simulation**:
    To run a basic simulation with random agents and visualize the results, execute:
    ```bash
    python test.py
    ```
    This will generate plots for each episode and print summary statistics to the console.

*   **Training Agents**:
    Inspect `train.py` to understand the placeholder setup for training MARL agents. For more advanced training configurations, logging, and agent selection, use `launch_train.py` (see details below).

## Training with `launch_train.py`

The `launch_train.py` script provides a more comprehensive way to run training sessions for agents in the `CyberSecurityEnv`. It allows configuration of different agent types (random, placeholder RL), logging to TensorBoard and CSV, and managing model saving.

### Purpose
*   Execute training episodes for the `CyberSecurityEnv`.
*   Configure attacker and defender agent types (random play or placeholder RL agents).
*   Manage logging of training metrics (rewards, environment state, action counts) to TensorBoard and CSV files.
*   Handle saving of (placeholder) agent models.
*   Provide live plotting of key metrics during training.

### Command-Line Arguments

The script uses `argparse` for configuration. Key arguments include:

*   **Agent Selection**:
    *   `--attacker_type`: Type of attacker agent (`random`, `rl`). Default: `random`.
    *   `--defender_type`: Type of defender setup (`random`, `collaborative_rl`, `centralized_rl`). Default: `random`.
*   **Model Paths (for RL agents)**:
    *   `--attacker_model_path`: Path to a pre-trained attacker model.
    *   `--defender_model_path`: Path to a pre-trained defender model.
*   **Training Duration**:
    *   `--num_episodes`: Number of training episodes. Default: `1000`.
*   **Logging & Output**:
    *   `--log_dir`: Base directory for all logs. Default: `./training_logs`.
    *   `--run_name`: Specific name for the current run; logs will be in `<log_dir>/<run_name>`. Defaults to a timestamp.
    *   `--disable_live_plots`: Flag to disable `matplotlib` live plots.
*   **Common RL Parameters**:
    *   `--learning_rate`: Learning rate for RL agents. Default: `1e-4`.
    *   `--batch_size`: Batch size for RL training. Default: `32`.
*   **QMIX Specific Parameters**:
    *   A group of arguments like `--qmix_hidden_dim`, `--qmix_mixing_embed_dim`, `--qmix_gamma`, etc., are available for configuring the QMIX agent (if `defender_type` is `centralized_rl`).

For a full list of all available arguments and their descriptions, run:
```bash
python launch_train.py --help
```

### Agent Configurations Supported

The script supports the following agent configurations via command-line arguments:

*   **Attacker (`--attacker_type`)**:
    *   `random`: Attacker takes random actions based on its action space.
    *   `rl`: Uses the `VicomRL_DQNAgent` placeholder, intended for a DQN-based agent from the hypothetical `vicomrl` library.
*   **Defenders (`--defender_type`)**:
    *   `random`: All defender agents take random actions based on their individual action spaces.
    *   `collaborative_rl`: Each defender agent is an independent `VicomRL_DQNAgent` placeholder. This setup is for decentralized execution where defenders might learn individually or share parameters/experiences through the `vicomrl` library.
    *   `centralized_rl`: All defender agents are controlled by a single `VicomRL_QMIXDefenders` agent placeholder, intended for a QMIX-based algorithm from the `vicomrl` library.

### Output and Logging

*   **Console Output**: The script prints the full configuration at startup, followed by per-episode summaries including rewards, steps taken, and moving averages. Placeholder messages from agent methods (init, learn, save, load) are also printed.
*   **Log Directory**: All output files (TensorBoard logs, CSV metrics, saved models, plots) for a specific run are saved under a unique directory: `<log_dir>/<run_name>/`.
*   **TensorBoard**: Detailed metrics are logged for TensorBoard. This includes:
    *   Rewards (attacker, defender team, defender team moving average).
    *   Traffic metrics (benign/malicious reaching server).
    *   Performance metrics (average server load, total compromised hosts).
    *   Costs (total defender action costs).
    *   Action counts (categorized for attacker and summed for defenders).
    Launch TensorBoard with: `tensorboard --logdir <log_dir>` (or `tensorboard --logdir ./training_logs` if using the default).
*   **CSV File**: A file named `episode_metrics.csv` is created in the run's log directory. It contains a detailed row for each episode, including all metrics logged to TensorBoard.
*   **Plots**:
    *   If live plotting is not disabled (`--disable_live_plots` is not used), a `matplotlib` window shows live updates of rewards and traffic.
    *   At the end of the training run (even if live plotting was disabled, as long as the plotter object was created), the final state of these plots is saved as `final_training_plots.png` in the run's log directory.
*   **Models**: Placeholder agent models are "saved" (indicated by print statements) into a `models` subdirectory within the run's log directory (e.g., `<log_dir>/<run_name>/models/attacker/` or `<log_dir>/<run_name>/models/qmix_defenders/`). Saving occurs at intervals specified by `--save_model_interval` and at the very end of the training run.

### Using Custom RL Library (`vicomrl`)

The `rl` options for attacker and defender types (`VicomRL_DQNAgent`, `VicomRL_QMIXDefenders`) are currently placeholders. They are designed to demonstrate where a custom or third-party MARL library (referred to as `vicomrl` in comments) would be integrated.

To use these with an actual RL library:
1.  Ensure the library (e.g., `vicomrl`) is installed and accessible in your `PYTHONPATH`.
2.  Modify the `VicomRL_DQNAgent` and `VicomRL_QMIXDefenders` classes in `launch_train.py`:
    *   Replace placeholder print statements with actual instantiation and usage of your library's agent/policy classes.
    *   Adjust the `get_action`, `learn`, `save_model`, and `load_model` methods to call the corresponding methods of your library's agents.
    *   Ensure the observation and action spaces provided by `CyberSecurityEnv` are compatible with what your RL agents expect. For instance, the `VicomRL_DQNAgent` currently assumes a discrete action space (`action_space.n`) which might need adaptation for the environment's `Tuple` action spaces, especially for defenders in a collaborative setup. The `VicomRL_QMIXDefenders` has a placeholder for mapping its internal discrete actions per agent to the environment's complex `Tuple` actions.

## Purpose & Potential Uses

*   **Research Platform**: Serves as a testbed for advancing research in multi-agent reinforcement learning, particularly in the context of cybersecurity and autonomous defense.
*   **Strategy Simulation**: Allows for the simulation and analysis of various cyber attack and defense strategies and their interactions in a controlled environment.
*   **Educational Tool**: Can be used for educational purposes to help understand the complex dynamics of network attacks, defensive mechanisms, and the application of AI in cybersecurity.
