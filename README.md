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
*   `README.md`: This file, providing an overview and documentation for the project.

## Setup and Dependencies

*   Python 3.8+
*   Main Python package dependencies:
    *   `gymnasium`
    *   `numpy`
    *   `matplotlib`
*   It is recommended to set up a virtual environment and install dependencies, potentially from a `requirements.txt` file (not currently provided, but can be created from imports).

## How to Run

*   **Test Simulation**:
    To run a basic simulation with random agents and visualize the results, execute:
    ```bash
    python test.py
    ```
    This will generate plots for each episode and print summary statistics to the console.

*   **Training Agents**:
    Inspect `train.py` to understand the placeholder setup for training MARL agents. To perform actual training, you would need to integrate a MARL library (e.g., RLlib, PyMARL) and replace the random action sampling with policy-based action selection and learning updates.

## Purpose & Potential Uses

*   **Research Platform**: Serves as a testbed for advancing research in multi-agent reinforcement learning, particularly in the context of cybersecurity and autonomous defense.
*   **Strategy Simulation**: Allows for the simulation and analysis of various cyber attack and defense strategies and their interactions in a controlled environment.
*   **Educational Tool**: Can be used for educational purposes to help understand the complex dynamics of network attacks, defensive mechanisms, and the application of AI in cybersecurity.
