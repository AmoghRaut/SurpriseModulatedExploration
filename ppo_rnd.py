import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import os
import gym
import time
import warnings

# --- RND Components (Paste the classes defined previously here) ---
class RunningMeanStdTorch:
    # Simple Welford's online algorithm without MPI
    # Shape: Shape of the data stream (e.g., observation shape, () for rewards)
    def __init__(self, shape=(), epsilon=1e-4, device='cpu'): # Added device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon # Small value to avoid division by zero
        self._shape = shape
        self._epsilon = epsilon
        self.device = device # Store device

    def update(self, x):
        # x should be a torch tensor on the correct device
        # Assumes x is shape (batch_size, *self.shape)
        if x.device != self.device:
             warnings.warn(f"Input tensor device ({x.device}) differs from RMS device ({self.device}). Moving input to {self.device}.")
             x = x.to(self.device)

        x = x.to(torch.float32)

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False) # Use population variance for stability
        batch_count = x.shape[0]
        if batch_count > 0: # Avoid update if batch is empty
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        # Combine variances using the parallel algorithm formula (simplified for serial batches)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return torch.sqrt(self.var)

    def normalize(self, x, clip_range=None, epsilon=1e-8):
        # Normalize input tensor x (assumed to be on self.device or moved)
        if x.device != self.device:
             warnings.warn(f"Input tensor device ({x.device}) differs from RMS device ({self.device}). Moving input to {self.device}.")
             x = x.to(self.device)

        # Ensure calculations are done in float32 for model compatibility
        mean = self.mean.float()
        std = torch.sqrt(self.var).float()
        normalized_x = (x.float() - mean) / (std + epsilon)
        if clip_range:
            normalized_x = torch.clamp(normalized_x, -clip_range, clip_range)
        return normalized_x

    def denormalize(self, normalized_x, epsilon=1e-8):
         # Denormalize input tensor normalized_x
        if normalized_x.device != self.device:
             warnings.warn(f"Input tensor device ({normalized_x.device}) differs from RMS device ({self.device}). Moving input to {self.device}.")
             normalized_x = normalized_x.to(self.device)

        mean = self.mean.float()
        std = torch.sqrt(self.var).float()
        x = normalized_x * (std + epsilon) + mean
        return x
    def get_state(self):
        """Returns the internal state for checkpointing."""
        # Ensure tensors are moved to CPU before converting to numpy
        return {
            'mean': self.mean.cpu().numpy(),
            'var': self.var.cpu().numpy(),
            'count': self.count
            # novelty_ema is part of RNDIntrinsicRewardModule state
        }

    def set_state(self, state, device):
        """Restores the internal state from a checkpoint."""
        # Convert numpy arrays back to tensors on the target device
        self.mean = torch.tensor(state['mean'], dtype=torch.float32).to(device)
        self.var = torch.tensor(state['var'], dtype=torch.float32).to(device)
        self.count = state['count']
        self.device = device # Ensure device attribute is updated


class RNDNetwork(nn.Module):
    """Defines the convolutional base and final FC layer for RND."""
    def __init__(self, input_channels=1, rep_size=512):
        super().__init__()
        # Adjusted based on original code c1,c2,c3 sizes
        nf = [32, 64, 64]
        rf = [8, 4, 3]
        stride = [4, 2, 1]

        self.conv1 = nn.Conv2d(input_channels, nf[0], kernel_size=rf[0], stride=stride[0])
        self.conv2 = nn.Conv2d(nf[0], nf[1], kernel_size=rf[1], stride=stride[1])
        self.conv3 = nn.Conv2d(nf[1], nf[2], kernel_size=rf[2], stride=stride[2])

        # NOTE: Calculate flattened size based on assumed 84x84 input AFTER channel selection
        # Adjust if your input size differs!
        h, w = 84, 84
        def conv_out_size(in_size, kernel, stride, padding=0): # VALID padding = 0
            return (in_size + 2 * padding - (kernel - 1) - 1) // stride + 1
        h = conv_out_size(h, rf[0], stride[0])
        w = conv_out_size(w, rf[0], stride[0])
        h = conv_out_size(h, rf[1], stride[1])
        w = conv_out_size(w, rf[1], stride[1])
        h = conv_out_size(h, rf[2], stride[2])
        w = conv_out_size(w, rf[2], stride[2])
        self.flattened_size = h * w * nf[2]

        self.fc = nn.Linear(self.flattened_size, rep_size)
        # Optional: Add weight initialization if needed

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

class RNDPredictorNetwork(nn.Module):
    """Defines the RND predictor network."""
    def __init__(self, input_channels=1, rep_size=512, enlargement=2):
        super().__init__()
        nf = [32, 64, 64] # Match target structure
        rf = [8, 4, 3]
        stride = [4, 2, 1]

        self.conv1 = nn.Conv2d(input_channels, nf[0], kernel_size=rf[0], stride=stride[0])
        self.conv2 = nn.Conv2d(nf[0], nf[1], kernel_size=rf[1], stride=stride[1])
        self.conv3 = nn.Conv2d(nf[1], nf[2], kernel_size=rf[2], stride=stride[2])

        # Use same flattened size calculation as RNDNetwork
        h, w = 84, 84
        def conv_out_size(in_size, kernel, stride, padding=0):
             return (in_size + 2 * padding - (kernel - 1) - 1) // stride + 1
        h = conv_out_size(h, rf[0], stride[0])
        w = conv_out_size(w, rf[0], stride[0])
        h = conv_out_size(h, rf[1], stride[1])
        w = conv_out_size(w, rf[1], stride[1])
        h = conv_out_size(h, rf[2], stride[2])
        w = conv_out_size(w, rf[2], stride[2])
        self.flattened_size = h * w * nf[2]

        # Predictor has extra FC layers
        fc_hid_size = 256 * enlargement
        self.fc1 = nn.Linear(self.flattened_size, fc_hid_size)
        self.fc2 = nn.Linear(fc_hid_size, fc_hid_size)
        self.fc3 = nn.Linear(fc_hid_size, rep_size)
        # Optional: Add weight initialization if needed

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNDIntrinsicRewardModule:
    def __init__(self, observation_shape, action_space, device, learning_rate=1e-4, update_proportion=0.25, rep_size=512, ema_alpha=0.01, gating_kappa=1.0):
        self.device = device
        self.ema_alpha = ema_alpha
        self.gating_kappa = gating_kappa
        self.novelty_ema = torch.tensor(0.0, device=self.device) # Initialize as tensor on device

        # Assuming single channel input based on TF code analysis [:,:,:,-1:]
        input_channels = 1
        # Shape for RMS should be (C, H, W) or just () for rewards
        # Use only H, W from observation_shape assuming C is handled
        single_frame_obs_shape = (input_channels, observation_shape[1], observation_shape[2]) # (1, H, W)
        self.ob_shape = single_frame_obs_shape # Store single frame shape for RMS
        print(f"RND using observation RMS shape: {self.ob_shape}")

        self.ob_rms = RunningMeanStdTorch(shape=self.ob_shape, epsilon=1e-4, device=device)
        self.rew_rms = RunningMeanStdTorch(shape=(), epsilon=1e-4, device=device)

        self.target = RNDNetwork(input_channels=input_channels, rep_size=rep_size).to(device)
        self.predictor = RNDPredictorNetwork(input_channels=input_channels, rep_size=rep_size).to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.update_proportion = update_proportion
        # self.ema_alpha and self.novelty_ema are initialized above

    def _preprocess_obs(self, obs_batch):
        # obs_batch: (B, Stack, H, W, C=1) or (B*T, Stack, H, W, C=1) from FrameStackObservation, on self.device
        # Select the latest frame: shape (B or B*T, H, W, C=1)
        latest_frame = obs_batch[:, -1, :, :, :]

        # Permute to (B or B*T, C=1, H, W) for Conv2d compatibility if needed by RMS normalize
        # The RMS normalize expects (C, H, W) stats, so input should match
        # Assuming GrayScale wrapper keeps channel dim, C=1
        if latest_frame.dim() == 4 and latest_frame.shape[-1] == 1: # B, H, W, C
            obs_for_rnd = latest_frame.permute(0, 3, 1, 2) # -> (B, C=1, H, W)
        else:
            # Handle unexpected shape or already correct shape
            # This might need adjustment based on exact output of GrayScaleObservation
            print(f"Warning: Unexpected observation shape in RND preprocess: {latest_frame.shape}")
            obs_for_rnd = latest_frame # Pass through, might error later

        # Normalize using ob_rms - expects shape matching self.ob_shape (1, H, W)
        normalized_obs = self.ob_rms.normalize(obs_for_rnd, clip_range=5.0)
        return normalized_obs # Shape (B or B*T, 1, H, W)

    def compute_reward(self, next_obs_batch):
        """Computes the raw (unnormalized) intrinsic reward."""
        # next_obs_batch shape: (B, Stack, H, W, C) when called from rollout
        normalized_next_obs = self._preprocess_obs(next_obs_batch) # Output: (B, 1, H, W)

        with torch.no_grad():
            target_features = self.target(normalized_next_obs)     # Output: (B, rep_size)
            predictor_features = self.predictor(normalized_next_obs) # Output: (B, rep_size)

        # Calculate MSE reward per sample
        raw_reward = F.mse_loss(target_features, predictor_features, reduction='none').mean(dim=-1) # Output: (B,)

        # ---- START Dopamine Gating ----
        # Update EMA of novelty (raw intrinsic reward)
        # Ensure raw_reward.mean() is a scalar tensor on the same device as self.novelty_ema
        current_raw_reward_mean = raw_reward.mean() # This is a tensor
        self.novelty_ema = (1.0 - self.ema_alpha) * self.novelty_ema + self.ema_alpha * current_raw_reward_mean

        # Calculate gating factor D_t
        # self.novelty_ema is already a tensor on self.device
        gating_factor = torch.sigmoid(self.gating_kappa * (raw_reward - self.novelty_ema.detach())) # Detach EMA to prevent gradients flowing through it to itself

        # Modulate reward
        gated_reward = raw_reward * gating_factor
        reward_to_return = gated_reward
        # ---- END Dopamine Gating ----

        # --- REMOVE Reshaping Logic ---
        # if len(next_obs_batch.shape) == 5: # B, T, C, H, W # Condition will be false here
        #      reward = raw_reward.view(next_obs_batch.shape[0], next_obs_batch.shape[1]) # Reshape to (B, T)
        # else:
        #      reward = raw_reward # Shape (B,)
        # --- Return the reward directly ---
        return reward_to_return # Shape (B,) - Correct shape for storage

    def update_predictor(self, next_obs_batch):
        """Computes predictor loss and performs optimizer step."""
        normalized_next_obs = self._preprocess_obs(next_obs_batch)

        with torch.no_grad(): # Target features don't need gradients
             target_features = self.target(normalized_next_obs)
        predictor_features = self.predictor(normalized_next_obs)

        loss_per_sample = F.mse_loss(target_features, predictor_features, reduction='none').mean(dim=-1)

        mask = torch.rand(loss_per_sample.shape, device=self.device) < self.update_proportion
        loss = (loss_per_sample * mask.float()).sum() / torch.clamp(mask.float().sum(), min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping for predictor optimizer
        # nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item()

    def normalize_reward(self, reward_batch):
        """Normalizes a batch of intrinsic rewards using rew_rms."""
        if not isinstance(reward_batch, torch.Tensor):
             reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)

        flat_rewards = reward_batch.view(-1).to(self.device) # Ensure on correct device
        self.rew_rms.update(flat_rewards)

        std_dev = torch.sqrt(self.rew_rms.var).float() # Already on device
        # Avoid division by zero or instability with very small std dev
        normalized_reward = reward_batch / (std_dev + 1e-8)
        return normalized_reward # Keep on device

    def update_obs_rms(self, obs_batch):
         """Updates observation normalization statistics."""
         # Assumes obs_batch is (B, Stack, H, W, C) or (B*T, Stack, H, W, C), potentially on GPU
         original_shape = obs_batch.shape
         is_batched_time_series = len(original_shape) == 5 # B, T, C, H, W - Check this dim order carefully

         # Reshape and select latest frame logic needs refining based on actual input shape
         # Assuming input from collector is (B*T, Stack, H, W, C) where B*T = batch_size
         # Select latest frame -> (B*T, H, W, C)
         latest_frame = obs_batch[:, -1, :, :, :]

         # Permute to (B*T, C, H, W) and move to RMS device (e.g., CPU)
         if latest_frame.dim() == 4 and latest_frame.shape[-1] == 1: # B*T, H, W, C=1
            obs_for_update = latest_frame.permute(0, 3, 1, 2).to(self.ob_rms.device) # -> (B*T, C=1, H, W)
         else:
             # Handle unexpected shape
             print(f"Warning: Unexpected observation shape in RND update_obs_rms: {latest_frame.shape}")
             # Attempt to move device anyway, might need further shape adjustments
             obs_for_update = latest_frame.to(self.ob_rms.device)

         # --- Cast to float64 before passing to update ---
         self.ob_rms.update(obs_for_update.to(torch.float32))
         # -------------------------------------------------


# --- PPO Components ---

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, frame_stack_k, num_actions):
        super().__init__()
        input_channels_ac = frame_stack_k # Input channels are the stacked frames
        # Use a network structure similar to the RND networks for feature extraction
        nf = [32, 64, 64]
        rf = [8, 4, 3]
        stride = [4, 2, 1]

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(input_channels_ac, nf[0], kernel_size=rf[0], stride=stride[0])),
            nn.ReLU(),
            layer_init(nn.Conv2d(nf[0], nf[1], kernel_size=rf[1], stride=stride[1])),
            nn.ReLU(),
            layer_init(nn.Conv2d(nf[1], nf[2], kernel_size=rf[2], stride=stride[2])),
            nn.ReLU(),
            nn.Flatten(),
            # Calculate flattened size based on input H,W=84x84 AFTER convolutions
            # Adjust if input size differs!
            layer_init(nn.Linear(3136, 512)), # Example: 3136 = 7*7*64 for 84x84 input
            nn.ReLU()
        )
        # Actor head
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        # Critic heads (separate for intrinsic and extrinsic)
        self.critic_int = layer_init(nn.Linear(512, 1), std=1)
        self.critic_ext = layer_init(nn.Linear(512, 1), std=1)

        # Calculate flattened size properly (Example for 84x84 input)
        # Replace 3136 with the actual calculated size based on your input_channels and obs shape
        def conv_out_size(in_size, kernel, stride, padding=0):
            return (in_size + 2 * padding - (kernel - 1) - 1) // stride + 1
        h, w = 84, 84 # Assuming 84x84 input
        h = conv_out_size(h, rf[0], stride[0])
        w = conv_out_size(w, rf[0], stride[0])
        h = conv_out_size(h, rf[1], stride[1])
        w = conv_out_size(w, rf[1], stride[1])
        h = conv_out_size(h, rf[2], stride[2])
        w = conv_out_size(w, rf[2], stride[2])
        self.flattened_size = h * w * nf[2]
        # Redefine linear layers with correct input size
        self.network[7] = layer_init(nn.Linear(self.flattened_size, 512)) # Index 7 is the first Linear layer
        print(f"ActorCritic using flattened CNN output size: {self.flattened_size}")

    def _preprocess_ac_input(self, x):
        # Input x shape: (NumEnvs, FrameStack, H, W, C=1) from FrameStackObservation
        # Permute to (NumEnvs, C=1, FrameStack, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        # Reshape to (NumEnvs, C*FrameStack, H, W) -> (NumEnvs, FrameStack, H, W) as C=1
        x = x.reshape(x.size(0), -1, x.size(3), x.size(4))
        # Normalize 0-255 -> 0-1
        return x / 255.0
    
    def get_value(self, x):
        """Gets both intrinsic and extrinsic value estimates."""
        processed_x = self._preprocess_ac_input(x)
        hidden = self.network(processed_x)
        v_int = self.critic_int(hidden)
        v_ext = self.critic_ext(hidden)
        return v_int, v_ext

    def get_action_and_value(self, x, action=None):
        """Gets action, logprob, entropy, and value estimates."""
        processed_x = self._preprocess_ac_input(x)
        hidden = self.network(processed_x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        v_int = self.critic_int(hidden)
        v_ext = self.critic_ext(hidden)
        return action, log_prob, entropy, v_int, v_ext


class PPOAgent:
    # Corrected __init__ to accept 'envs' (vectorized) instead of 'env' (single)
    def __init__(self, envs, config, device): # <--- Changed 'env' to 'envs'
        self.envs = envs # <--- Store the vectorized envs object
        self.config = config
        self.device = device

        # Get observation/action space info from the vectorized env
        self.obs_shape = envs.single_observation_space.shape # <--- Use single_observation_space
        self.num_actions = envs.single_action_space.n # <--- Use single_action_space
        self.num_envs = envs.num_envs # <--- Get number of parallel envs

        frame_stack_k = config['frame_stack'] # Get frame stack 

        self.actor_critic = ActorCritic(frame_stack_k, self.num_actions).to(device)
        single_frame_shape_for_rnd = (1, 84, 84) # Hardcode for typical Atari setup
        self.rnd_module = RNDIntrinsicRewardModule(
            observation_shape=single_frame_shape_for_rnd, # Pass single frame shape!
            action_space=envs.single_action_space,
            device=device,
            learning_rate=config['rnd_lr'],
            update_proportion=config['rnd_update_proportion'],
            ema_alpha=config['rnd_ema_alpha'],
            gating_kappa=config['rnd_gating_kappa']
        )

        self.optimizer = optim.Adam(
            list(self.actor_critic.parameters()),
            lr=config['policy_lr'],
            eps=1e-5
        )

        # Storage setup (uses self.num_envs, which is now correctly derived)
        self.obs_shape = envs.single_observation_space.shape # (Stack, H, W, C)
        self.obs = torch.zeros((config['n_steps'], self.num_envs) + self.obs_shape).to(device)
        self.actions = torch.zeros((config['n_steps'], self.num_envs)).to(device) # Assuming discrete actions
        self.logprobs = torch.zeros((config['n_steps'], self.num_envs)).to(device)
        self.rewards_ext = torch.zeros((config['n_steps'], self.num_envs)).to(device)
        self.rewards_int_raw = torch.zeros((config['n_steps'], self.num_envs)).to(device)
        self.dones = torch.zeros((config['n_steps'], self.num_envs)).to(device)
        self.values_int = torch.zeros((config['n_steps'], self.num_envs)).to(device)
        self.values_ext = torch.zeros((config['n_steps'], self.num_envs)).to(device)

    def save_checkpoint(self, path, update, global_step):
        """Saves the agent's state to a checkpoint file."""
        checkpoint = {
            'update': update,
            'global_step': global_step,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rnd_predictor_state_dict': self.rnd_module.predictor.state_dict(),
            'rnd_optimizer_state_dict': self.rnd_module.optimizer.state_dict(),
            'rnd_obs_rms_state': self.rnd_module.ob_rms.get_state(),
            'rnd_rew_rms_state': self.rnd_module.rew_rms.get_state(),
            'rnd_novelty_ema': self.rnd_module.novelty_ema.cpu().item(), # Save as float
            # Add other states if needed, e.g., random states, LR scheduler
        }
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {e}")


    def load_checkpoint(self, path):
        """Loads the agent's state from a checkpoint file."""
        start_update = 1
        global_step = 0
        if path is None or not os.path.exists(path):
             print(f"Warning: Checkpoint path '{path}' not found or not specified. Starting from scratch.")
             return start_update, global_step # Return default start steps

        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.rnd_module.predictor.load_state_dict(checkpoint['rnd_predictor_state_dict'])
            self.rnd_module.optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
            self.rnd_module.ob_rms.set_state(checkpoint['rnd_obs_rms_state'], self.device)
            self.rnd_module.rew_rms.set_state(checkpoint['rnd_rew_rms_state'], self.device)
            if 'rnd_novelty_ema' in checkpoint: # For backward compatibility
                self.rnd_module.novelty_ema = torch.tensor(checkpoint['rnd_novelty_ema'], device=self.device)

            start_update = checkpoint.get('update', 1) + 1 # Resume from the *next* update
            global_step = checkpoint.get('global_step', 0)

            print(f"Checkpoint loaded from {path}. Resuming from update {start_update}, global_step {global_step}")

        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}. Starting from scratch.")
            start_update = 1
            global_step = 0

        return start_update, global_step
    
    # Inside ppo_rnd_impl.py -> PPOAgent class

    def _compute_advantages_returns(self, next_obs, next_done, norm_intrinsic_rewards):
        """Computes GAE and returns for both reward streams.
           Handles non-episodic intrinsic rewards based on config['use_done_intrinsic'].
        """
        with torch.no_grad():
            # Get value estimates for the *next* state (after the last step of the rollout)
            next_value_int, next_value_ext = self.actor_critic.get_value(next_obs)
            # Ensure correct shape (num_envs,)
            next_value_int = next_value_int.view(-1)
            next_value_ext = next_value_ext.view(-1)

            advantages_int = torch.zeros_like(norm_intrinsic_rewards).to(self.device)
            advantages_ext = torch.zeros_like(self.rewards_ext).to(self.device)
            lastgaelam_int = 0
            lastgaelam_ext = 0

            for t in reversed(range(self.config['n_steps'])):
                # Determine the correct 'done' signal and 'next value' for this timestep t
                if t == self.config['n_steps'] - 1:
                    # For the last step, the 'next done' is the input next_done
                    # The 'next value' is the bootstrapped value from the policy
                    actual_next_done = next_done
                    nextvalues_int = next_value_int
                    nextvalues_ext = next_value_ext
                else:
                    # For intermediate steps, the 'next done' is from the stored buffer
                    # The 'next value' is the stored value estimate from the buffer
                    actual_next_done = self.dones[t + 1]
                    nextvalues_int = self.values_int[t + 1]
                    nextvalues_ext = self.values_ext[t + 1]

                # --- Determine nextnonterminal based on config ---
                # Extrinsic rewards always use the actual done signal
                nextnonterminal_ext = 1.0 - actual_next_done

                # Intrinsic rewards conditionally ignore the done signal
                if self.config.get('use_done_intrinsic', False): # Default is False (non-episodic)
                    # Episodic intrinsic: Use the actual done signal
                    nextnonterminal_int = nextnonterminal_ext
                else:
                    # Non-episodic intrinsic: Ignore the done signal (always treat as non-terminal)
                    nextnonterminal_int = torch.ones_like(actual_next_done) # Vector of 1.0s
                # ---------------------------------------------

                # GAE for intrinsic rewards
                # Use nextnonterminal_int in both delta and recurrence (matches original TF code)
                delta_int = norm_intrinsic_rewards[t] + self.config['gamma_int'] * nextvalues_int * nextnonterminal_int - self.values_int[t]
                advantages_int[t] = lastgaelam_int = delta_int + self.config['gamma_int'] * self.config['lambda'] * nextnonterminal_int * lastgaelam_int

                # GAE for extrinsic rewards
                # Use nextnonterminal_ext in both delta and recurrence
                delta_ext = self.rewards_ext[t] + self.config['gamma_ext'] * nextvalues_ext * nextnonterminal_ext - self.values_ext[t]
                advantages_ext[t] = lastgaelam_ext = delta_ext + self.config['gamma_ext'] * self.config['lambda'] * nextnonterminal_ext * lastgaelam_ext

            # Calculate returns by adding advantages to value estimates
            returns_int = advantages_int + self.values_int
            returns_ext = advantages_ext + self.values_ext

            # Combine advantages using coefficients
            advantages = self.config['int_coeff'] * advantages_int + self.config['ext_coeff'] * advantages_ext

        return advantages, returns_int, returns_ext



    def train(self):
        """Main training loop."""
        global_step = 0
        start_time = time.time()
        # Use self.envs which is the vectorized environment object
        next_obs, _ = self.envs.reset(seed=self.config['seed']) # Seed reset
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        num_updates = self.config['total_timesteps'] // (self.config['n_steps'] * self.num_envs)


        print(f"Starting training for {num_updates} updates...")

        for update in range(1, num_updates + 1):
            # --- Rollout Phase ---
            all_rollout_obs = [] # Collect all observations in rollout for RMS update
            all_rollout_next_obs = [] # Collect next_obs for RND predictor update

            for step in range(self.config['n_steps']):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                all_rollout_obs.append(next_obs.cpu()) # Store CPU copy for RMS update

                with torch.no_grad():
                    action, logprob, _, value_int, value_ext = self.actor_critic.get_action_and_value(next_obs)
                    self.values_int[step] = value_int.flatten()
                    self.values_ext[step] = value_ext.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # Execute action
                next_obs_np, reward, done, _ = self.env.step(action.cpu().numpy()[0]) # Get action for single env
                self.rewards_ext[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(self.device).unsqueeze(0)
                next_done = torch.Tensor([done]).to(self.device) # Ensure correct shape

                # --- RND Integration: Compute Raw Intrinsic Reward ---
                # Compute reward based on the *next* observation
                with torch.no_grad():
                    raw_int_reward = self.rnd_module.compute_reward(next_obs) # Pass next_obs (B=1, C, H, W)
                    self.rewards_int_raw[step] = raw_int_reward.flatten() # Store raw reward
                    all_rollout_next_obs.append(next_obs) # Store next_obs for predictor update

                if done:
                    print(f"GStep: {global_step}, Update: {update}, Episode Finished.")
                    next_obs = torch.Tensor(self.env.reset()).to(self.device).unsqueeze(0)
                    next_done = torch.zeros(self.num_envs).to(self.device)


            # --- Post-Rollout Processing ---
            # RND: Update observation normalization
            self.rnd_module.update_obs_rms(torch.cat(all_rollout_obs, dim=0)) # Update with all obs from rollout

            # RND: Normalize intrinsic rewards for the full rollout
            norm_intrinsic_rewards = self.rnd_module.normalize_reward(self.rewards_int_raw)

            # Calculate advantages and returns
            advantages, returns_int, returns_ext = self._compute_advantages_returns(
                next_obs, next_done, norm_intrinsic_rewards
            )

            # Flatten batch (treat steps and envs as batch dimension)
            b_obs = self.obs.reshape((-1,) + self.obs_shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns_int = returns_int.reshape(-1)
            b_returns_ext = returns_ext.reshape(-1)
            b_values_int = self.values_int.reshape(-1) # For clipping
            b_values_ext = self.values_ext.reshape(-1) # For clipping
            b_next_obs = torch.cat(all_rollout_next_obs, dim=0).reshape((-1,) + self.obs_shape) # All next obs collected during rollout


            # --- Optimization Phase ---
            batch_size = self.config['n_steps'] * self.num_envs
            minibatch_size = batch_size // self.config['n_minibatches']
            inds = np.arange(batch_size)

            for epoch in range(self.config['n_epochs']):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = inds[start:end]

                    # Get data for minibatch
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns_int = b_returns_int[mb_inds]
                    mb_returns_ext = b_returns_ext[mb_inds]
                    # mb_values_int = b_values_int[mb_inds] # For value clipping if used
                    # mb_values_ext = b_values_ext[mb_inds] # For value clipping if used
                    mb_next_obs = b_next_obs[mb_inds] # Next obs for RND update

                    # --- RND Integration: Update Predictor ---
                    rnd_loss = self.rnd_module.update_predictor(mb_next_obs)

                    # --- PPO Loss Calculation ---
                    _, newlogprob, entropy, new_values_int, new_values_ext = self.actor_critic.get_action_and_value(
                        mb_obs, mb_actions
                    )
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    # Policy loss (clipped)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config['clip_coef'], 1 + self.config['clip_coef'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (separate for int/ext)
                    v_loss_int = F.mse_loss(new_values_int.view(-1), mb_returns_int)
                    v_loss_ext = F.mse_loss(new_values_ext.view(-1), mb_returns_ext)
                    v_loss = v_loss_int + v_loss_ext # Combine value losses

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = (pg_loss
                           - self.config['ent_coef'] * entropy_loss
                           + self.config['vf_coef'] * v_loss
                           + self.config['rnd_loss_coeff'] * rnd_loss # Add RND predictor loss
                           )

                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()

            # --- Logging ---
            if update % self.config['log_interval'] == 0:
                 print(f"Update {update}/{num_updates}, GStep {global_step}")
                 print(f"  Loss: {loss.item():.4f}, PG Loss: {pg_loss.item():.4f}, V Loss: {v_loss.item():.4f}")
                 print(f"  Entropy: {entropy_loss.item():.4f}, RND Loss: {rnd_loss:.4f}")
                 # Add mean reward logging etc.
                 print(f"  Time: {time.time() - start_time:.2f}s")
                 print("-" * 20)

        print("Training finished.")
        self.env.close()


# --- Main Execution ---
if __name__ == "__main__":
    config = {
        'env_id': "ALE/MontezumaRevenge-v5", # Requires Atari + ALE installed
        # 'env_id': "CartPole-v1", # Simpler env for testing - WILL NEED NETWORK CHANGES (MLP not CNN)
        'total_timesteps': 10_000_000,
        'n_steps': 128,         # Steps per rollout
        'n_minibatches': 4,
        'n_epochs': 4,          # PPO epochs per update
        'gamma_int': 0.99,      # Discount for intrinsic rewards
        'gamma_ext': 0.999,     # Discount for extrinsic rewards (often higher)
        'lambda': 0.95,         # GAE lambda
        'clip_coef': 0.1,       # PPO clipping coefficient
        'ent_coef': 0.001,      # Entropy coefficient
        'vf_coef': 0.5,         # Value function loss coefficient
        'max_grad_norm': 0.5,
        'policy_lr': 1e-4,      # Learning rate for PPO networks
        'rnd_lr': 1e-4,         # Learning rate for RND predictor
        'int_coeff': 1.0,       # Intrinsic reward coefficient for advantage
        'ext_coeff': 2.0,       # Extrinsic reward coefficient for advantage
        'rnd_update_proportion': 0.25, # Proportion of experience for RND update
        'rnd_loss_coeff': 1.0,  # Weight of RND predictor loss in total loss
        'log_interval': 10      # Log every N updates
    }

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment Setup ---
    # NOTE: Atari environments often require wrappers for preprocessing (frame stack, grayscale, etc.)
    # This basic example assumes the environment directly outputs usable frames (C, H, W)
    # Consider using Stable-Baselines3 wrappers or similar if needed.
    env = gym.make(config['env_id'])
    # env = gym.wrappers.RecordEpisodeStatistics(env) # Helpful wrapper

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # --- Agent Initialization and Training ---
    agent = PPOAgent(env=env, config=config, device=device)
    agent.train()
