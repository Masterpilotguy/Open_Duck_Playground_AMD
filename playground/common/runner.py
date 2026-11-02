"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
import time
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF CUDA/info warnings
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax
import optax

# Wrap optax.clip_by_global_norm to guard against any NaN gradients destroying policy weights
_orig_clip_by_global_norm = optax.clip_by_global_norm

def _safe_clip_by_global_norm(max_norm):
    return optax.chain(optax.zero_nans(), _orig_clip_by_global_norm(max_norm))

optax.clip_by_global_norm = _safe_clip_by_global_norm

from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        self.best_eval_reward = float("-inf")
        self.best_params = None
        self.best_step = 0
        self.latest_params = None
        self.latest_step = 0
        self.latest_checkpoint_path = ""
        self.orbax_checkpointer = ocp.PyTreeCheckpointer()

        # CSV log for tracking checkpoint performance
        self.summary_csv_path = self.output_dir / "summary.csv"
        if not self.summary_csv_path.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(self.summary_csv_path, "w") as f:
                    f.write("step,eval_reward,reward_std,checkpoint_dir,is_best\n")
            except Exception:
                pass
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        reward = float(metrics.get("eval/episode_reward", float("-inf")))
        reward_std = float(metrics.get("eval/episode_reward_std", 0.0))

        print("-----------")
        print(f"STEP: {num_steps} reward: {reward:.3f} reward_std: {reward_std:.3f}")

        is_best = False
        # Record best evaluation reward (skip step 0 initial untrained state)
        if num_steps > 0 and reward > self.best_eval_reward:
            self.best_eval_reward = reward
            self.best_params = self.latest_params
            self.best_step = num_steps
            is_best = True
            print(f"★ New best evaluation reward: {reward:.3f}! (Will export ONNX at end of run)")

        # Append to summary CSV
        try:
            with open(self.summary_csv_path, "a") as f:
                f.write(f"{num_steps},{reward:.4f},{reward_std:.4f},{self.latest_checkpoint_path},{is_best}\n")
        except Exception:
            pass

        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # Save raw Orbax checkpoints for training recovery/resuming
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        self.orbax_checkpointer.save(path, params, force=True, save_args=save_args)

        # Store latest params for progress_callback to record
        self.latest_params = params
        self.latest_step = current_step
        self.latest_checkpoint_path = path

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_envs"] = 4096
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps

        # Scale num_evals dynamically based on num_timesteps if not explicitly passed
        num_evals = getattr(self.args, "num_evals", None)
        if num_evals is not None:
            self.ppo_training_params["num_evals"] = num_evals
        elif self.num_timesteps <= 1_000_000:
            self.ppo_training_params["num_evals"] = 4
        elif self.num_timesteps <= 5_000_000:
            self.ppo_training_params["num_evals"] = 6
        elif self.num_timesteps <= 20_000_000:
            self.ppo_training_params["num_evals"] = 9

        # Prevent Welford normalizer std → 0 (0/0 = NaN in obs normalization)
        self.ppo_training_params["normalize_observations_std_eps"] = 1e-6
        self.ppo_training_params["unroll_length"] = 40
        self.ppo_training_params["num_minibatches"] = 16
        self.ppo_training_params["num_envs"] = 16384
        # Brax requires (batch_size * num_minibatches) % num_envs == 0
        num_envs = self.ppo_training_params["num_envs"]
        self.ppo_training_params["batch_size"] = num_envs // self.ppo_training_params["num_minibatches"]

        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        start_time = time.perf_counter()
        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )

        # Export models to ONNX at end of training
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        print("\n--> Exporting final trained policy to ONNX...")
        final_onnx_path = f"{self.output_dir}/{d}_{self.num_timesteps}_final.onnx"
        export_onnx(
            params,
            self.action_size,
            self.ppo_params,
            self.obs_size,
            output_path=final_onnx_path,
        )

        # Export best policy if recorded and different from final step
        if self.best_params is not None and self.best_step != self.num_timesteps:
            print(f"\n--> Exporting best policy (step: {self.best_step}, reward: {self.best_eval_reward:.3f}) to ONNX...")
            best_onnx_path = f"{self.output_dir}/{d}_{self.best_step}_best.onnx"
            export_onnx(
                self.best_params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=best_onnx_path,
            )

        elapsed_time = time.perf_counter() - start_time
        overall_sps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total Timesteps: {self.num_timesteps:,}")
        if self.best_eval_reward > float("-inf"):
            print(f"Best Reward:     {self.best_eval_reward:.3f}")
        if hours > 0:
            print(
                f"Elapsed Time:    {int(hours)}h {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f}s)"
            )
        elif minutes > 0:
            print(
                f"Elapsed Time:    {int(minutes)}m {seconds:.2f}s ({elapsed_time:.2f}s)"
            )
        else:
            print(f"Elapsed Time:    {elapsed_time:.2f}s")
        print(f"Overall SPS:     {overall_sps:,.0f} steps/second")
        print("=" * 60 + "\n")
