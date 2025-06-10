# env_utils.py
import gymnasium as gym
from gymnasium import spaces
# Corrected import: Use FrameStackObservation instead of FrameStack
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, ClipReward, TransformReward, RecordEpisodeStatistics,RecordVideo
import numpy as np
from collections import deque
import os
import cv2 # Required for ResizeObservation if using INTER_AREA

# Optional: Define StickyAction wrapper if needed (from original code)
class StickyActionEnv(gym.Wrapper):
    """
    Applies sticky actions, where the previous action is repeated with probability p.
    """
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = 0
        # Ensure the action space is discrete before proceeding
        assert isinstance(env.action_space, gym.spaces.Discrete), "StickyActionEnv requires a discrete action space."


    def reset(self, **kwargs):
        self.last_action = 0
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        if self.env.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

# Optional: Define MaxAndSkip wrapper if needed (from original code, adapted for Gymnasium)
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame, taking the max over the skipped frames.
    And repeat action during skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        # Ensure observation space is Box and uint8 for buffer
        assert isinstance(env.observation_space, gym.spaces.Box), "MaxAndSkipEnv requires Box observation space."
        assert env.observation_space.dtype == np.uint8, "MaxAndSkipEnv requires uint8 observation space."
        self._obs_buffer = deque(maxlen=2) # Stores only the last 2 frames for max pooling

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}

        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break

        # Ensure buffer has items before max pooling
        if len(self._obs_buffer) > 0:
            max_frame = np.max(np.stack(list(self._obs_buffer)), axis=0)
        else:
             # This should ideally not happen if reset logic is correct,
             # but handle defensively. Return the last observation received.
             max_frame = obs

        # The done condition depends on the final step outcome
        done = terminated or truncated

        return max_frame, total_reward, terminated, truncated, info


# env_utils.py
# ... (Keep existing imports and wrappers like StickyAction, MaxAndSkip) ...
import gymnasium as gym
from gymnasium import spaces

import numpy as np
from collections import deque, defaultdict # Added defaultdict
import os
from copy import copy # Added copy

# --- Add this Helper Function ---
def get_ram(env):
    """Helper to recursively find the ALE environment and get RAM."""
    # Loop through wrapped environments
    while hasattr(env, "env"):
        # Check if the current env is the ALE environment
        if hasattr(env, "ale"):
            return env.ale.getRAM()
        env = env.env
    # Check the final unwrapped environment
    if hasattr(env, "ale"):
        return env.ale.getRAM()
    # Handle cases where env structure might differ or ALE is not found
    # print("Warning: Could not find ALE environment to get RAM.")
    return None

# --- Add this Wrapper ---
class MontezumaRoomsWrapper(gym.Wrapper):
    def __init__(self, env, room_ram_address=3):
        super().__init__(env)
        self.room_ram_address = room_ram_address
        self.visited_rooms_current_episode = set()
        print(f"[Wrapper Init] MontezumaRoomsWrapper applied.") # Debug print

    def get_current_room(self):
        ram = get_ram(self.env)
        if ram is not None and len(ram) > self.room_ram_address:
            return int(ram[self.room_ram_address])
        return None # Indicate failure to read room

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_room = self.get_current_room()
        if current_room is not None:
            if current_room not in self.visited_rooms_current_episode:
                # print(f"Entered new room: {current_room}") # Debug Print
                self.visited_rooms_current_episode.add(current_room)

        # Add room count to info when episode finishes (check if stats wrapper added 'episode' key)
        if terminated or truncated:
            if "episode" not in info:
                # This might happen if RecordEpisodeStatistics is wrapped AFTER this
                # It's better practice to wrap RecordEpisodeStatistics first or last consistently
                info["episode"] = {}
            info["episode"]["num_rooms"] = len(self.visited_rooms_current_episode)
            info["episode"]["visited_rooms_set"] = copy(self.visited_rooms_current_episode) # Store the set too
            # print(f"Episode End: Visited {info['episode']['num_rooms']} rooms.") # Debug Print

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Clear rooms before starting new episode
        self.visited_rooms_current_episode.clear()
        # Add initial room after reset
        observation, info = self.env.reset(**kwargs)
        current_room = self.get_current_room()
        if current_room is not None:
             self.visited_rooms_current_episode.add(current_room)
             # print(f"Reset: Starting in room {current_room}") # Debug Print
        # else:
             # print("Reset: Could not determine starting room.") # Debug Print

        # Add initial room info if needed by downstream wrappers, though usually done at end.
        # if "episode" not in info: info["episode"] = {}
        # info["episode"]["num_rooms"] = len(self.visited_rooms_current_episode)

        return observation, info

# --- Modify make_env function ---
def make_env(env_id, seed, idx, capture_video, run_name, frame_stack_k=4, clip_rewards=True, max_episode_steps=None):
    """
    Utility function for simplifying environment creation and wrapping.
    """
    def thunk():
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except Exception as e:
             print(f"Error creating env {env_id}: {e}")
             raise

        # Apply RecordEpisodeStatistics relatively early to capture base stats
        env = RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            video_folder = f"videos/{run_name}"
            if not os.path.exists(video_folder):
                os.makedirs(video_folder, exist_ok=True)
            try:
                env = RecordVideo(env, video_folder, episode_trigger=lambda x: x % 50 == 0)
            except Exception as e:
                print(f"Error setting up RecordVideo: {e}. Continuing without video capture.")

        # --- Add Montezuma Wrapper Conditionally ---
        # Apply AFTER RecordEpisodeStatistics so it can add to the 'episode' dict
        if "MontezumaRevenge" in env_id:
             print(f"Applying MontezumaRoomsWrapper for env {idx}")
             env = MontezumaRoomsWrapper(env, room_ram_address=3)
        # ------------------------------------------

        # Apply Atari common wrappers AFTER potential game-specific wrappers
        if hasattr(env, 'spec') and env.spec is not None and 'NoFrameskip' in env.spec.id:
            # env = StickyActionEnv(env) # Optional
            env = MaxAndSkipEnv(env, skip=4)

        env = ResizeObservation(env, (84, 84))
        if len(env.observation_space.shape) == 3 and env.observation_space.shape[2] in [1, 3]:
             env = GrayscaleObservation(env, keep_dim=True)
        elif len(env.observation_space.shape) == 2:
             env = gym.wrappers.TransformObservation(env, lambda obs: obs[..., np.newaxis])

        if clip_rewards:
            env = TransformReward(env, lambda reward: float(np.sign(reward)))

        if frame_stack_k > 0:
             env = FrameStackObservation(env, frame_stack_k)

        # Apply TimeLimit relatively late, but check its interaction with stat wrappers
        if max_episode_steps is not None:
             env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # Seed action space last
        env.action_space.seed(seed + idx)
        return env
    return thunk