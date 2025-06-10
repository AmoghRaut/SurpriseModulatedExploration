# templates/ppo_rnd_atari/experiment.py
import argparse
import os
import random
import time
from distutils.util import strtobool
from tqdm import tqdm
import json # <-- Import JSON
import sys # <-- Import sys for logging redirection

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Assuming ppo_rnd.py contains PPOAgent, ActorCritic, RND modules, RunningMeanStdTorch
from ppo_rnd import PPOAgent # <-- Relative import assumes ppo_rnd.py is in the same dir
from env_utils import make_env # <-- Relative import assumes env_utils.py is in the same dir
import ale_py
gym.register_envs(ale_py)

def parse_args():
    parser = argparse.ArgumentParser()
    # --- Keep most args, modify/remove some ---
    # parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment") # REMOVED - Handled by AI Scientist
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    # parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases") # REMOVED
    # parser.add_argument("--wandb-project-name", type=str, default="ppo-rnd-pytorch", help="the wandb's project name") # REMOVED
    # parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project") # REMOVED
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to capture videos of the agent performances")

    # --- ADDED ---
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save results (run_0, etc.)")

    # Algorithm specific arguments (KEEP AS IS or adjust defaults)
    parser.add_argument("--env-id", type=str, default="ALE/MontezumaRevenge-v5", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="total timesteps of the experiments") # Reduced for faster testing/baseline runs if needed
    parser.add_argument("--policy-lr", type=float, default=1e-4)
    parser.add_argument("--rnd-lr", type=float, default=1e-4)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--gamma-int", type=float, default=0.99)
    parser.add_argument("--gamma-ext", type=float, default=0.999)
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--n-minibatches", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--clip-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--int-coeff", type=float, default=1.0)
    parser.add_argument("--ext-coeff", type=float, default=2.0)
    parser.add_argument("--rnd-update-proportion", type=float, default=1) # Adjusted default
    parser.add_argument("--rnd-loss-coeff", type=float, default=1.0)
    parser.add_argument("--use-done-intrinsic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--rnd_ema_alpha", type=float, default=0.01, help="EMA alpha for RND novelty gating")
    parser.add_argument("--rnd_gating_kappa", type=float, default=1.0, help="Kappa for RND novelty gating sigmoid")
    # parser.add_argument("--save-interval", type=int, default=50, help="frequency of saving checkpoints (in updates)") # Maybe REMOVE or adjust
    # parser.add_argument("--save-path", type=str, default="checkpoints", help="directory to save checkpoints") # REMOVED - Use out_dir
    # parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="path to checkpoint to resume training from") # REMOVED - AI Scientist runs are fresh

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.n_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatches)
    return args


if __name__ == "__main__":
    args = parse_args()

    # --- Create Output Directories ---
    os.makedirs(args.out_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.out_dir, "tb_logs")
    checkpoint_dir = os.path.join(args.out_dir, "checkpoints") # Use subfolder
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True) # Make checkpoints dir

    # --- Logging Setup (Redirect stdout/stderr) ---
    log_path = os.path.join(args.out_dir, "log.txt")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        f = open(log_path, 'w')
        sys.stdout = f
        sys.stderr = f

        print(f"Starting experiment run in: {args.out_dir}")
        print(f"Full command: {' '.join(sys.argv)}")
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # --- TensorBoard Setup ---
        writer = SummaryWriter(tb_log_dir) # Log to subfolder
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # --- Seeding ---
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "mps")
        print(f"Using device: {device}")

        # --- Environment Setup ---
        # Define run_name based on out_dir for video saving if needed
        run_name_for_video = os.path.basename(args.out_dir) # Use output dir name
        envs = SyncVectorEnv(
            [make_env(args.env_id, args.seed, i, args.capture_video, run_name_for_video, args.frame_stack, args.clip_rewards)
             for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        print(f"Observation space: {envs.single_observation_space}")
        print(f"Action space: {envs.single_action_space}")


        # --- Agent Setup ---
        config = vars(args)
        agent = PPOAgent(envs=envs, config=config, device=device)

        # --- Observation Stat Initialization ---
        # (Keep this section as is)
        print("Collecting initial random rollouts for observation normalization...")
        INIT_STEPS = 1000 # Number of random steps to collect (adjust as needed)
        temp_obs_list = []
        obs, _ = envs.reset(seed=args.seed)
        for _ in tqdm(range(INIT_STEPS), desc="Init Rollout"):
            actions = envs.action_space.sample() # Sample random actions
            next_obs, _, _, _, _ = envs.step(actions)
            latest_frame = torch.Tensor(next_obs[:-1, :, :, :]) # Get latest frame
            if latest_frame.dim() == 4 and latest_frame.shape[-1] == 1:
                obs_for_update = latest_frame.permute(0, 3, 1, 2)
            else:
                obs_for_update = latest_frame
            temp_obs_list.append(obs_for_update.cpu())
            obs = next_obs
        if temp_obs_list:
            initial_obs_batch = torch.cat(temp_obs_list, dim=0)
            agent.rnd_module.update_obs_rms(initial_obs_batch)
            print(f"Initialized Obs RMS with {initial_obs_batch.shape[0]} random steps.")
            del initial_obs_batch
            del temp_obs_list
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        # -----------------------------------------------

        # --- Training Loop ---
        start_update = 1
        global_step = 0
        # Remove checkpoint loading logic - AI Scientist runs are typically fresh
        # if args.resume_from_checkpoint: ...

        start_time = time.time()
        total_start_time = time.time() # For overall wall time

        print("Performing initial environment reset...")
        next_obs, _ = envs.reset(seed=args.seed + start_update)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        num_updates = args.total_timesteps // args.batch_size
        print(f"Targeting {num_updates} updates.")

        recent_returns = deque(maxlen=100)
        recent_lengths = deque(maxlen=100) # Track lengths
        recent_rooms = deque(maxlen=100) # Track rooms (if applicable)
        best_mean_reward = -np.inf
        best_mean_rooms = 0.0 # <-- ADDED: Track max of the smoothed room count
        max_rooms_ever = 0 # <-- ADDED: Track peak raw rooms
        max_reward_ever = -np.inf # <-- ADDED: Track peak raw reward

        final_metrics = {} # Dictionary to store final results

        best_reward_ckpt_path = os.path.join(checkpoint_dir, "best_reward_model.pth")
        best_rooms_ckpt_path = os.path.join(checkpoint_dir, "best_rooms_model.pth")
        final_ckpt_path = os.path.join(checkpoint_dir, "final_model.pth")
        for update in tqdm(range(1, num_updates + 1), desc="Training Updates"):
            # --- Rollout Phase (Keep mostly as is, ensure logging works) ---
            all_rollout_obs = []
            all_rollout_next_obs = []
            # Initialize accumulators for this update's rewards
            update_total_raw_int_reward = 0.0
            update_total_norm_int_reward = 0.0
            num_int_reward_samples = 0

            for step in range(args.n_steps):
                global_step += args.num_envs
                agent.obs[step] = next_obs
                agent.dones[step] = next_done
                all_rollout_obs.append(next_obs.cpu())

                with torch.no_grad():
                    action, logprob, _, value_int, value_ext = agent.actor_critic.get_action_and_value(next_obs)
                    agent.values_int[step] = value_int.flatten()
                    agent.values_ext[step] = value_ext.flatten()
                agent.actions[step] = action
                agent.logprobs[step] = logprob

                next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = terminated | truncated
                agent.rewards_ext[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(device)
                next_done = torch.Tensor(done).to(device)

                with torch.no_grad():
                    raw_int_reward = agent.rnd_module.compute_reward(next_obs)
                    agent.rewards_int_raw[step] = raw_int_reward.flatten()
                    all_rollout_next_obs.append(next_obs)
                    # Accumulate intrinsic rewards for update-level average
                    update_total_raw_int_reward += raw_int_reward.sum().item()
                    num_int_reward_samples += raw_int_reward.numel()


                # --- Logging Episodic Info ---
                if "_episode" in infos:
                    finished_mask = infos["_episode"]
                    finished_indices = np.where(finished_mask)[0]

                    if len(finished_indices) > 0:
                        ep_returns = infos["episode"]["r"][finished_mask] if "r" in infos.get("episode", {}) else []
                        ep_lengths = infos["episode"]["l"][finished_mask] if "l" in infos.get("episode", {}) else []

                        # Handle num_rooms carefully - check if the key exists first
                        ep_num_rooms_values = [] # Default to empty list
                        if "episode" in infos and "num_rooms" in infos["episode"]:
                            # Ensure accessing only if the key exists
                            try:
                                # Get the array for all envs, then filter with the mask
                                rooms_array = infos["episode"]["num_rooms"]
                                ep_num_rooms_values = rooms_array[finished_mask]
                            except KeyError:
                                print("Warning: 'num_rooms' key found in infos['episode'] but failed to access.")
                        # Simplified room extraction - might need refinement based on wrapper exact output
                        try:
                           ep_num_rooms = [infos["episode"]["num_rooms"][i] for i in finished_indices if "num_rooms" in infos.get("episode", {})]
                        except KeyError: # Handle case where num_rooms might not exist for all finished episodes
                           ep_num_rooms = [None] * len(finished_indices)


                        for i, env_idx in enumerate(finished_indices):
                            ep_ret = ep_returns[i]
                            ep_len = ep_lengths[i]
                            ep_rooms = ep_num_rooms[i] if i < len(ep_num_rooms) else None # Safety check index

                            # --- Update Peak Trackers ---
                            if not np.isnan(ep_ret):
                                max_reward_ever = max(max_reward_ever, ep_ret) # <-- Update max reward
                            if ep_rooms is not None:
                                max_rooms_ever = max(max_rooms_ever, ep_rooms) # <-- Update max rooms
                            # --- End Peak Update ---
                            log_str = f"[Env {env_idx}] GStep={global_step}, EpReturn={ep_ret:.2f}, EpLength={ep_len}"
                            if ep_rooms is not None:
                                log_str += f", Rooms={ep_rooms}"
                            print(log_str)

                            if not np.isnan(ep_ret):
                                writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                                recent_returns.append(ep_ret)
                            if not np.isnan(ep_len):
                                writer.add_scalar("charts/episodic_length", ep_len, global_step)
                                recent_lengths.append(ep_len)
                            if ep_rooms is not None:
                                writer.add_scalar("charts/episodic_num_rooms", ep_rooms, global_step)
                                recent_rooms.append(ep_rooms)

            # --- Post-Rollout Processing ---
            agent.rnd_module.update_obs_rms(torch.cat(all_rollout_obs, dim=0))
            norm_intrinsic_rewards = agent.rnd_module.normalize_reward(agent.rewards_int_raw)

            # Log average intrinsic rewards for the update
            if num_int_reward_samples > 0:
                 avg_raw_int_reward_update = update_total_raw_int_reward / num_int_reward_samples
                 # Calculate normalized sum similarly (or just log mean of tensor)
                 avg_norm_int_reward_update = norm_intrinsic_rewards.mean().item()
                 writer.add_scalar("rnd/mean_raw_intrinsic_reward", avg_raw_int_reward_update, global_step)
                 writer.add_scalar("rnd/mean_norm_intrinsic_reward", avg_norm_int_reward_update, global_step)
                 writer.add_scalar("rnd/novelty_ema", agent.rnd_module.novelty_ema.item(), global_step) # Log novelty_ema

            writer.add_scalar("rnd/reward_rms_std", agent.rnd_module.rew_rms.std.item(), global_step)
            writer.add_scalar("rnd/obs_rms_mean_norm", torch.norm(agent.rnd_module.ob_rms.mean).item(), global_step)

            advantages, returns_int, returns_ext = agent._compute_advantages_returns(
                next_obs, next_done, norm_intrinsic_rewards
            )

            b_obs = agent.obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = agent.logprobs.reshape(-1)
            b_actions = agent.actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns_int = returns_int.reshape(-1)
            b_returns_ext = returns_ext.reshape(-1)
            try:
                 b_next_obs = torch.cat(all_rollout_next_obs, dim=0).reshape((-1,) + envs.single_observation_space.shape)
            except RuntimeError as e:
                 print(f"Error reshaping next_obs: {e}")
                 print(f"Num next_obs collected: {len(all_rollout_next_obs)}")
                 if all_rollout_next_obs: print(f"Shape of first next_obs: {all_rollout_next_obs[0].shape}")
                 # Handle error appropriately, maybe skip update?
                 continue # Skip to next update


            # --- Optimization Phase (Keep as is) ---
            inds = np.arange(args.batch_size)
            avg_pg_loss, avg_v_loss, avg_ent_loss, avg_rnd_loss = 0, 0, 0, 0
            for epoch in range(args.n_epochs):
                np.random.shuffle(inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = inds[start:end]
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns_int = b_returns_int[mb_inds]
                    mb_returns_ext = b_returns_ext[mb_inds]
                    mb_next_obs = b_next_obs[mb_inds]

                    rnd_loss = agent.rnd_module.update_predictor(mb_next_obs)
                    _, newlogprob, entropy, new_values_int, new_values_ext = agent.actor_critic.get_action_and_value(
                        mb_obs, mb_actions
                    )
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss_int = F.mse_loss(new_values_int.view(-1), mb_returns_int)
                    v_loss_ext = F.mse_loss(new_values_ext.view(-1), mb_returns_ext)
                    v_loss = v_loss_int + v_loss_ext
                    entropy_loss = entropy.mean()
                    loss = (pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + args.rnd_loss_coeff * rnd_loss)

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), args.max_grad_norm)
                    agent.optimizer.step()

                    avg_pg_loss += pg_loss.item()
                    avg_v_loss += v_loss.item()
                    avg_ent_loss += entropy_loss.item()
                    avg_rnd_loss += rnd_loss


            # --- Checkpointing Best Model ---
            if len(recent_returns) >= max(10, recent_returns.maxlen // 2):
                current_mean_reward = np.mean(recent_returns)
                current_mean_rooms = np.mean([r for r in recent_rooms if r is not None]) #

                writer.add_scalar("charts/mean_episodic_return_100", current_mean_reward, global_step)

                if not np.isnan(current_mean_rooms): # Avoid logging NaN
                    writer.add_scalar("charts/mean_episodic_num_rooms_100", current_mean_rooms, global_step) # <-- Log the smoothed room count

                if current_mean_reward > best_mean_reward:
                    best_mean_reward = current_mean_reward
                    print(f"\nNew best mean reward: {best_mean_reward:.2f} -> Saving checkpoint to {best_reward_ckpt_path}")
                    agent.save_checkpoint(best_reward_ckpt_path, update, global_step) # Overwrites previous best reward model
                
                            # --- Best Smoothed Rooms --- <--- ADDED BLOCK
                valid_rooms = [r for r in recent_rooms if r is not None]
                if len(valid_rooms) >= max(10, recent_rooms.maxlen // 2): # Check if enough valid room counts
                    current_mean_rooms = np.mean(valid_rooms)
                    writer.add_scalar("charts/mean_episodic_num_rooms_100", current_mean_rooms, global_step)
                    if current_mean_rooms > best_mean_rooms:
                        best_mean_rooms = current_mean_rooms # Update the tracked maximum smoothed value
                        print(f"\nNew best mean rooms (100): {best_mean_rooms:.2f} -> Saving checkpoint to {best_rooms_ckpt_path}")
                        agent.save_checkpoint(best_rooms_ckpt_path, update, global_step) # Overwrites previous best rooms model
                # --- END ADDED BLOCK ---


            # --- Logging after each update ---
            num_minibatches = args.batch_size // args.minibatch_size
            num_opt_steps = args.n_epochs * num_minibatches
            avg_pg_loss /= num_opt_steps
            avg_v_loss /= num_opt_steps
            avg_ent_loss /= num_opt_steps
            avg_rnd_loss /= num_opt_steps

            current_time = time.time()
            sps = int(args.batch_size / (current_time - start_time))
            print(f"Update {update}/{num_updates}, GStep {global_step}, SPS: {sps}")
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/total_loss", loss.item(), global_step)
            writer.add_scalar("losses/value_loss", avg_v_loss, global_step)
            writer.add_scalar("losses/policy_loss", avg_pg_loss, global_step)
            writer.add_scalar("losses/entropy", avg_ent_loss, global_step)
            writer.add_scalar("losses/rnd_loss", avg_rnd_loss, global_step)

            start_time = current_time

        # --- Final Steps ---
        print("Training finished.")
        total_run_time = time.time() - total_start_time
        print(f"Total wall clock time: {total_run_time:.2f} seconds")

        # Save final model
        final_checkpoint_path = os.path.join(checkpoint_dir, f"final_model_u{num_updates}_s{global_step}.pth")
        agent.save_checkpoint(final_checkpoint_path, num_updates, global_step)

        envs.close()
        writer.close()

        # --- Collect Final Metrics for JSON ---
        # Use the best recorded mean reward
        final_metrics["Best Mean Episodic Return (Extrinsic)"] = {"means": float(best_mean_reward) if best_mean_reward > -np.inf else None}

        final_metrics["Max Mean Num Rooms (100)"] = {"means": float(best_mean_rooms)} # <-- ADDED Use 
        # Use the average of the last N episodes for final length/rooms
        final_metrics["Max number of Rooms Reached"] = {"means": float(max_rooms_ever)} # <-- ADDED
        final_metrics["Max reward in an episode"] = {"means": float(max_reward_ever) if max_reward_ever > -np.inf else None} # <-- ADDED

        if recent_lengths:
             final_metrics["Mean Episodic Length (Last 100)"] = {"means": float(np.mean(recent_lengths))}
        if recent_rooms:
            # Filter None before calculating mean
            valid_rooms = [r for r in recent_rooms if r is not None]
            if valid_rooms:
                final_metrics["Mean Episodic Num Rooms (Last 100)"] = {"means": float(np.mean(valid_rooms))}
            else:
                final_metrics["Mean Episodic Num Rooms (Last 100)"] = {"means": None}
        # Add other metrics if available/logged (e.g., final avg intrinsic reward?)
        # final_metrics["Mean Norm Intrinsic Reward (Last Update)"] = {"means": avg_norm_int_reward_update}
        final_metrics["Total Global Steps"] = {"means": float(global_step)}
        final_metrics["Wall Clock Time (seconds)"] = {"means": float(total_run_time)}

    

        # Save final_info.json
        final_info_path = os.path.join(args.out_dir, "final_info.json")
        try:
            with open(final_info_path, 'w') as f_json:
                json.dump(final_metrics, f_json, indent=4)
            print(f"Final metrics saved to {final_info_path}")
        except Exception as e:
            print(f"Error saving final_info.json: {e}")

    except Exception as e:
         print(f"An error occurred: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback to log file
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if 'f' in locals() and not f.closed:
            f.close()
