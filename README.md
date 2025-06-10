
# Surprise Modulated Exploration: Dopamine-Gated RND (DG-RND)

This repository contains the PyTorch implementation of **Dopamine-Gated Random Network Distillation (DG-RND)**, a novel intrinsic motivation mechanism designed to improve exploration in sparse-reward reinforcement learning environments. The agent is trained using Proximal Policy Optimization (PPO).

The primary evaluation environment is the notoriously difficult Atari game **Montezuma's Revenge**, where efficient exploration is the key to performance.

## Core Concept: Dopamine-Gated RND

Standard Random Network Distillation (RND) encourages exploration by rewarding the agent for visiting states where its predictor network has a high prediction error. However, this can lead to the agent becoming desensitized in areas of sustained novelty or prematurely fixated as the predictor overfits.

**DG-RND** addresses this by introducing a dynamic, adaptive gating mechanism inspired by dopaminergic prediction error signaling in the brain.

The mechanism works as follows:
1.  **Novelty Expectation:** The agent maintains an Exponential Moving Average (EMA) of the recent raw novelty (RND prediction error). This EMA serves as an adaptive baseline for the agent's "expectation" of novelty.
2.  **Surprise Signal:** The intrinsic reward is modulated by a dynamic gate that measures the *discrepancy* between the instantaneous novelty of a new state and the agent's current novelty expectation.
3.  **Gated Reward:** A sigmoid function transforms this "surprise" into a gating factor.
    *   If a state is **surprisingly novel** (much higher novelty than the EMA), the gate opens (approaches 1), and the full intrinsic reward is passed through.
    *   If a state's novelty is **as expected** (close to the EMA), the gate is halfway closed (around 0.5), scaling down the reward.
    *   If a state is **less novel than expected** (e.g., a familiar state in a new region), the gate closes further (approaches 0), dampening the reward.

This allows the agent to focus its exploratory drive on genuinely surprising states, fostering more robust and efficient exploration.

## Key Features

-   **PPO + RND Foundation**: A solid and well-structured implementation of PPO with RND.
-   **Dopamine-Gated RND Module**: The novel intrinsic reward mechanism is cleanly integrated into the `RNDIntrinsicRewardModule` in `ppo_rnd.py`.
-   **Montezuma's Revenge Room Counter**: A custom Gymnasium wrapper (`MontezumaRoomsWrapper` in `env_utils.py`) that inspects the game's RAM to accurately track the number of unique rooms visited per episode—a critical metric for exploration.
-   **Vectorized Environments**: Uses `SyncVectorEnv` for fast parallel environment rollouts.
-   **Comprehensive Logging & Plotting**:
    -   Logs all metrics to TensorBoard for real-time monitoring.
    -   Saves a `final_info.json` with summary statistics upon completion.
    -   Includes a `plot.py` script to automatically generate learning curves and final performance bar charts from experiment runs.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://your-repo-url.git
    cd your_repo_repository
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install the required dependencies. You will need `gymnasium` with Atari support, `torch`, `tensorboard`, and a few other libraries.
    ```bash
    pip install gymnasium[atari] ale-py torch numpy tqdm pandas seaborn matplotlib opencv-python
    ```
    *Note: Ensure your `torch` installation is compatible with your CUDA version if you plan to use a GPU.*

## How to Run Experiments

The training scripts use command-line arguments for configuration. The results for each run will be saved in the directory specified by `--out_dir`.

### 1. Run the Baseline (PPO + Standard RND)

To run the baseline agent, use the `run_baseline.py` script.

```bash
python run_baseline.py --out_dir run_0 --env-id "ALE/MontezumaRevenge-v5" --total-timesteps 100000000
```
This will create a `run_0` directory containing the `log.txt`, `tb_logs`, and a `final_info.json` file.

### 2. Run the DG-RND Agent

To run the agent with the Dopamine-Gated RND mechanism, use the `run_dgrnd.py` script. You can configure the gating mechanism with the `--rnd_ema_alpha` and `--rnd_gating_kappa` arguments.

```bash
python run_dgrnd.py \
    --out_dir run_1 \
    --env-id "ALE/MontezumaRevenge-v5" \
    --total-timesteps 100000000 \
    --rnd_ema_alpha 0.01 \
    --rnd_gating_kappa 1.0
```
This will create a `run_1` directory with the results for this configuration.

## How to Plot Results

The `plot.py` script can generate plots for multiple runs, comparing them directly.

1.  Make sure your experiment directories (`run_0`, `run_1`, etc.) are present.
2.  Update the `labels` dictionary in `plot.py` to match your run directories and desired legend names.
3.  Run the script:

    ```bash
    python plot.py --output_dir plots
    ```
    This will create a `plots/` directory and save all the generated `.png` figures there. You can also specify which runs to plot:
    ```bash
    python plot.py --output_dir plots --runs run_0 run_1
    ```

## Codebase Overview

-   **`run_baseline.py`**: Main script to train the standard PPO+RND agent.
-   **`run_dgrnd.py`**: Main script to train the PPO+DG-RND agent.
-   **`ppo_rnd.py`**: Contains the core algorithmic components:
    -   `PPOAgent`: The main agent class that handles the training loop.
    -   `ActorCritic`: The policy and value network.
    -   `RNDIntrinsicRewardModule`: The module that calculates intrinsic rewards. **This is where the DG-RND logic is implemented.**
    -   `RunningMeanStdTorch`: A utility for normalizing rewards and observations.
-   **`env_utils.py`**: Contains utilities for creating and wrapping the Gymnasium environments, including the crucial `MontezumaRoomsWrapper`.
-   **`plot.py`**: Utility script for visualizing and comparing results from different experiment runs.
