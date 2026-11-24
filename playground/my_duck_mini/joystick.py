# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    reward_feet_air_time,
    cost_orientation,
    cost_ang_vel_xy,
    cost_torques,
    cost_action_rate,
    cost_joint_pos_limits,
    cost_stand_still,
    cost_head_pos,
    cost_termination,
    reward_alive,
    safe_norm,
)
from playground.my_duck_mini.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.35,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=1.0,  # rad/s
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=2.5,
                tracking_ang_vel=2.5,
                tracking_coupled=7.0,
                orientation=-3.0,
                ang_vel_xy=-0.1,
                torques=-1.0e-3,
                action_rate=-0.15,
                stand_still=-1.5,
                stand_still_feet=-4.0,
                standing_back_step=6.0,
                standing_side_step=6.0,
                alive=2.5,
                imitation=3.5,
                feet_air_time=3.0,
                stall_ds=-3.0,
                head_pos=-0.5,
                dof_pos_limits=-1.0,
                feet_collision=-1.0,
                termination=-50.0,
            ),
            tracking_sigma=0.010,
            tracking_sigma_ang=0.025,
            tracking_y_weight=1.5,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[2.0, 4.5],
            mod_magnitude_range=[0.10, 0.45],
            mod_ang_magnitude_range=[0.05, 0.45],
            strong_magnitude_range=[0.45, 0.80],
            strong_ang_magnitude_range=[0.40, 0.80],
            strong_push_prob=0.5,
            z_magnitude_range=[-0.03, 0.03],
            recovery_window_steps=75,
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],  # [-1.0, 1.0]
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,  # to make it easier
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/my_duck_mini/data/polynomial_coefficients.pkl"
            )

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[
            self.actuator_joint_ids
        ].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)
        for idx, name in enumerate(self.actuator_names):
            if "_hip" in name:
                qpos_noise_scale[idx] = self._config.noise_config.scales.hip_pos
            elif "_knee" in name:
                qpos_noise_scale[idx] = self._config.noise_config.scales.knee_pos
            elif "_ankle" in name:
                qpos_noise_scale[idx] = self._config.noise_config.scales.ankle_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)
        self._hip_roll_actuator_idx = jp.array(
            [i for i, n in enumerate(self.actuator_names) if "_hip_roll" in n],
            dtype=jp.int32,
        )
        self._backlash_actuator_idx = jp.array(
            [i for i in range(self._actuators) if i not in self.backlash_idx_to_add],
            dtype=jp.int32,
        )

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        rng, key_yaw, key_rp = jax.random.split(rng, 3)
        yaw = jax.random.uniform(key_yaw, (), minval=-3.14, maxval=3.14)
        rp = jax.random.uniform(key_rp, (2,), minval=-0.08, maxval=0.08)
        q_yaw = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        q_pitch = math.axis_angle_to_quat(jp.array([0.0, 1.0, 0.0]), rp[1])
        q_roll = math.axis_angle_to_quat(jp.array([1.0, 0.0, 0.0]), rp[0])
        quat = math.quat_mul(q_yaw, math.quat_mul(q_pitch, q_roll))
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw + roll/pitch noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        rng, key_j, key_hr = jax.random.split(rng, 3)

        # multiply actual joints with noise (excluding floating base and backlash)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key_j, (self._actuators,), minval=0.85, maxval=1.15
        )
        hip_roll_jitter = jax.random.uniform(
            key_hr,
            (self._hip_roll_actuator_idx.shape[0],),
            minval=-0.06,
            maxval=0.06,
        )
        qpos_j = qpos_j.at[self._hip_roll_actuator_idx].add(hip_roll_jitter)
        qpos_j = jp.clip(qpos_j, self._soft_lowers, self._soft_uppers)
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)

        rng, key_vlin, key_vang = jax.random.split(rng, 3)
        init_qvel_lin = jax.random.uniform(
            key_vlin, (3,), minval=-0.15, maxval=0.15
        )
        init_qvel_ang = jax.random.uniform(
            key_vang, (3,), minval=-0.20, maxval=0.20
        )
        qvel = self.set_floating_base_qvel(
            jp.concatenate([init_qvel_lin, init_qvel_ang]), qvel
        )
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        rng, act_delay_rng, imu_delay_rng = jax.random.split(rng, 3)
        action_delay_idx = jax.random.randint(
            act_delay_rng,
            (),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        imu_delay_idx = jax.random.randint(
            imu_delay_rng,
            (),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.zeros(5),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "push_recovery_steps": jp.array(0, dtype=jp.int32),
            # History & delay related.
            "action_delay_idx": action_delay_idx,
            "imu_delay_idx": imu_delay_idx,
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.tile(
                jp.array([0.0, 0.0, -1.0]), self._config.noise_config.imu_max_delay
            ),
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
            "imu_tip_latch": jp.array(0, dtype=jp.int32),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        cmd_norm_step = safe_norm(state.info["command"][:3])
        if USE_IMITATION_REWARD:
            grav_step = state.data.site_xmat[self._site_id].T @ jp.array(
                [0.0, 0.0, -1.0]
            )
            gyro_step = self.get_gyro(state.data)
            imu_tipped = (
                (jp.abs(grav_step[0]) > 0.10)
                | (jp.abs(grav_step[1]) > 0.10)
                | (jp.abs(gyro_step[1]) > 0.45)
                | (jp.abs(gyro_step[0]) > 0.45)
            )
            state.info["imu_tip_latch"] = jp.where(
                imu_tipped,
                jp.array(30, dtype=jp.int32),
                jp.maximum(state.info["imu_tip_latch"] - 1, 0),
            )
            is_moving = (cmd_norm_step >= 0.01) | (state.info["imu_tip_latch"] > 0)
            next_imitation_i = (
                state.info["imitation_i"] + 1
            ) % self.PRM.nb_steps_in_period
            state.info["imitation_i"] = jp.where(is_moving, next_imitation_i, 0)
            phase_angle = (
                (state.info["imitation_i"] / self.PRM.nb_steps_in_period) * 2.0 * jp.pi
            )
            state.info["imitation_phase"] = jp.where(
                is_moving,
                jp.array([jp.cos(phase_angle), jp.sin(phase_angle)]),
                jp.zeros(2),
            )
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],
                state.info["command"][1],
                state.info["command"][2],
                state.info["imitation_i"],
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        prev_last_act = state.info["last_act"]

        (
            state.info["rng"],
            push_mix_rng,
            push_theta_rng,
            push_lin_rng,
            push_ang_rng,
            push_z_rng,
            push_noise_rng,
        ) = jax.random.split(state.info["rng"], 7)

        # Handle action delay (per-episode constant index)
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_w_delay = action_history.reshape((-1, self._actuators))[
            state.info["action_delay_idx"]
        ]  # action with delay

        # Coupled body-aligned pushes
        # Standing: 55% sagittal (70% front-push -Vx / pi, 30% back-push +Vx / 0), 45% uniform [0, 2*pi), 50% strong
        # Walking: 50% lateral (50/50 +pi/2 or -pi/2), 50% uniform [0, 2*pi), push_cfg.strong_push_prob strong
        push_cfg = self._config.push_config
        is_standing = cmd_norm_step < 0.01
        theta_mode_rng, theta_sign_rng, theta_jitter_rng, theta_unif_rng = (
            jax.random.split(push_theta_rng, 4)
        )
        is_directed = jax.random.bernoulli(
            theta_mode_rng, jp.where(is_standing, 0.55, 0.50)
        )
        directed_sign = jax.random.bernoulli(
            theta_sign_rng, jp.where(is_standing, 0.30, 0.50)
        )
        directed_base = jp.where(
            is_standing,
            jp.where(directed_sign, 0.0, jp.pi),
            jp.where(directed_sign, 0.5 * jp.pi, -0.5 * jp.pi),
        )
        directed_theta = (
            directed_base
            + jax.random.uniform(theta_jitter_rng, minval=-0.35, maxval=0.35)
        ) % (2.0 * jp.pi)
        unif_theta = jax.random.uniform(theta_unif_rng, maxval=2.0 * jp.pi)
        push_theta_body = jp.where(is_directed, directed_theta, unif_theta)
        is_strong = jax.random.bernoulli(
            push_mix_rng, jp.where(is_standing, 0.50, push_cfg.strong_push_prob)
        )
        strong_lin_max = jp.where(
            is_standing, 0.85, push_cfg.strong_magnitude_range[1]
        )
        strong_ang_max = jp.where(
            is_standing, 0.90, push_cfg.strong_ang_magnitude_range[1]
        )
        lin_min = jp.where(
            is_strong,
            push_cfg.strong_magnitude_range[0],
            push_cfg.mod_magnitude_range[0],
        )
        lin_max = jp.where(
            is_strong,
            strong_lin_max,
            push_cfg.mod_magnitude_range[1],
        )
        ang_min = jp.where(
            is_strong,
            push_cfg.strong_ang_magnitude_range[0],
            push_cfg.mod_ang_magnitude_range[0],
        )
        ang_max = jp.where(
            is_strong,
            strong_ang_max,
            push_cfg.mod_ang_magnitude_range[1],
        )
        lin_mag = jax.random.uniform(push_lin_rng, minval=lin_min, maxval=lin_max)
        ang_mag = jax.random.uniform(push_ang_rng, minval=ang_min, maxval=ang_max)
        dvz = jax.random.uniform(
            push_z_rng,
            minval=push_cfg.z_magnitude_range[0],
            maxval=push_cfg.z_magnitude_range[1],
        )
        dw_noise = jax.random.uniform(
            push_noise_rng, (2,), minval=-0.10, maxval=0.10
        )

        addr_qpos = self._floating_base_qpos_addr
        w, x, y, z = (
            state.data.qpos[addr_qpos + 3],
            state.data.qpos[addr_qpos + 4],
            state.data.qpos[addr_qpos + 5],
            state.data.qpos[addr_qpos + 6],
        )
        yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        dvx_world = jp.cos(yaw + push_theta_body) * lin_mag
        dvy_world = jp.sin(yaw + push_theta_body) * lin_mag
        dwx_body = -jp.sin(push_theta_body) * ang_mag + dw_noise[0]
        dwy_body = jp.cos(push_theta_body) * ang_mag + dw_noise[1]
        dvx_body = jp.cos(push_theta_body) * lin_mag
        dvy_body = jp.sin(push_theta_body) * lin_mag

        push_active = (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        ) & self._config.push_config.enable
        push_mask = push_active.astype(jp.float32)

        push_world = (
            jp.array([dvx_world, dvy_world, dvz, dwx_body, dwy_body]) * push_mask
        )
        push_body = jp.where(
            push_active,
            jp.array([dvx_body, dvy_body, dvz, dwx_body, dwy_body]),
            jp.zeros(5),
        )

        addr_qvel = self._floating_base_qvel_addr
        qvel = state.data.qvel.at[addr_qvel : addr_qvel + 5].add(push_world)
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        state.info["push"] = push_body
        state.info["push_recovery_steps"] = jp.where(
            push_active,
            jp.array(self._config.push_config.recovery_window_steps, dtype=jp.int32),
            jp.maximum(state.info["push_recovery_steps"] - 1, 0),
        )

        ####

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]

            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets
                - self._config.max_motor_velocity * self.dt,  # control dt
                prev_motor_targets
                + self._config.max_motor_velocity * self.dt,  # control dt
            )

        # motor_targets.at[5:9].set(state.info["command"][3:])  # head joints
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        state.info["motor_targets"] = motor_targets

        # Shift action history BEFORE _get_obs so obs has exact 1-step timing parity with inference
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = jp.nan_to_num(p_f[..., -1], nan=0.0)
        p_fz_rel = jp.maximum(0.0, p_fz - jp.min(p_fz))
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz_rel)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
            first_contact,
            contact,
            prev_last_act,
        )
        # FIXME
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, -0.5, 10000.0)
        reward = jp.nan_to_num(reward, nan=0.0)
        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["step"] += 1
        state.info["push_step"] += 1

        done_bool = done.astype(bool)
        state.info["last_last_last_act"] = jp.where(
            done_bool,
            jp.zeros_like(state.info["last_last_last_act"]),
            state.info["last_last_last_act"],
        )
        state.info["last_last_act"] = jp.where(
            done_bool,
            jp.zeros_like(state.info["last_last_act"]),
            state.info["last_last_act"],
        )
        state.info["last_act"] = jp.where(
            done_bool, jp.zeros_like(state.info["last_act"]), state.info["last_act"]
        )
        state.info["motor_targets"] = jp.where(
            done_bool, self._default_actuator, motor_targets
        )
        state.info["action_history"] = jp.where(
            done_bool, jp.zeros_like(state.info["action_history"]), action_history
        )
        state.info["imu_history"] = jp.where(
            done_bool,
            jp.tile(
                jp.array([0.0, 0.0, -1.0]), self._config.noise_config.imu_max_delay
            ),
            state.info["imu_history"],
        )
        state.info["push_recovery_steps"] = jp.where(
            done_bool,
            jp.array(0, dtype=jp.int32),
            state.info["push_recovery_steps"],
        )
        state.info["rng"], act_delay_rng, imu_delay_rng = jax.random.split(
            state.info["rng"], 3
        )
        new_action_delay_idx = jax.random.randint(
            act_delay_rng,
            (),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        new_imu_delay_idx = jax.random.randint(
            imu_delay_rng,
            (),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        state.info["action_delay_idx"] = jp.where(
            done_bool, new_action_delay_idx, state.info["action_delay_idx"]
        )
        state.info["imu_delay_idx"] = jp.where(
            done_bool, new_imu_delay_idx, state.info["imu_delay_idx"]
        )
        if USE_IMITATION_REWARD:
            state.info["imitation_i"] = jp.where(
                done_bool, 0, state.info["imitation_i"]
            )
            state.info["imitation_phase"] = jp.where(
                done_bool, jp.zeros(2), state.info["imitation_phase"]
            )
        state.info["imu_tip_latch"] = jp.where(
            done_bool, jp.array(0, dtype=jp.int32), state.info["imu_tip_latch"]
        )
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 200,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done_bool | (state.info["step"] > 200),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] = jp.where(
            done_bool, jp.zeros_like(state.info["feet_air_time"]), state.info["feet_air_time"] * ~contact
        )
        state.info["last_contact"] = jp.where(
            done_bool, jp.zeros_like(contact), contact
        )
        state.info["swing_peak"] = jp.where(
            done_bool, jp.zeros_like(state.info["swing_peak"]), state.info["swing_peak"] * ~contact
        )
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                v_clean = jp.nan_to_num(v, nan=0.0)
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v_clean
                else:
                    state.metrics[f"cost/{k}"] = -v_clean
        state.metrics["swing_peak"] = jp.nan_to_num(
            jp.mean(state.info["swing_peak"]), nan=0.0
        )

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = (self.get_gravity(data)[-1] < 0.45) | (
            data.qpos[self._floating_base_qpos_addr + 2] < 0.08
        )
        bad_state = (
            jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | jp.isinf(data.qpos).any()
            | jp.isinf(data.qvel).any()
        )
        return fall_termination | bad_state

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay (per-episode constant index)
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        noisy_gravity = imu_history.reshape((-1, 3))[info["imu_delay_idx"]]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        if self.backlash_joint_qpos_addr:
            joint_backlash_raw = self.get_actuator_backlash_qpos(data.qpos).reshape(-1)
            joint_backlash = (
                jp.zeros(self._actuators)
                .at[self._backlash_actuator_idx]
                .set(joint_backlash_raw)
            )
            joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)

        state = jp.hstack(
            [
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                noisy_gravity,  # 3
                info["command"],  # 7
                noisy_joint_angles - self._default_actuator,  # 14
                noisy_joint_vel * self._config.dof_vel_scale,  # 14
                info["last_act"],  # 14
                info["last_last_act"],  # 14
                info["last_last_last_act"],  # 14
                info["motor_targets"],  # 14
                contact,  # 2
                info["imitation_phase"],  # 2
            ]
        )

        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 14
                contact,  # 2
                feet_vel,  # 2*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
                info["push"],  # 5
                (info["push_recovery_steps"] > 0).astype(jp.float32),  # 1
            ]
        )

        return {
            "state": jp.clip(
                jp.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0
            ),
            "privileged_state": jp.clip(
                jp.nan_to_num(privileged_state, nan=0.0, posinf=0.0, neginf=0.0),
                -100.0,
                100.0,
            ),
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
        prev_last_act: Optional[jax.Array] = None,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        if prev_last_act is None:
            prev_last_act = info["last_act"]

        local_angvel = self.get_gyro(data)
        # Transform IMU site velocity to trunk CoM velocity (imu - trunk inertial pos)
        imu_offset = jp.array([-0.0541257, -0.00091211, 0.0016710])
        local_linvel = self.get_local_linvel(data) - jp.cross(local_angvel, imu_offset)
        gravity_vec = self.get_gravity(data)
        grav_body = data.site_xmat[self._site_id].T @ jp.array([0.0, 0.0, -1.0])
        joints_qpos = self.get_actuator_joints_qpos(data.qpos)
        joints_qvel = self.get_actuator_joints_qvel(data.qvel)

        in_push = info["push_recovery_steps"] > 0
        cmd_norm = safe_norm(info["command"][:3])
        standing_push = in_push & (cmd_norm < 0.01)
        is_upright_standing = (jp.abs(grav_body[0]) < 0.06) & (
            jp.abs(grav_body[1]) < 0.06
        )
        back_tip = standing_push & (grav_body[0] < -0.08)
        side_tip = in_push & (
            (jp.abs(grav_body[1]) > 0.08) | (jp.abs(local_angvel[0]) > 0.45)
        )
        trunk_x_axis = data.xmat[self._torso_body_id][:, 0]
        trunk_x_horiz = jp.array([trunk_x_axis[0], trunk_x_axis[1], 0.0])
        trunk_x_horiz = trunk_x_horiz / safe_norm(trunk_x_horiz)
        trunk_y_horiz = jp.array([-trunk_x_horiz[1], trunk_x_horiz[0], 0.0])
        foot_rel = data.site_xpos[self._feet_site_id] - data.xpos[self._torso_body_id]
        foot_x_body = foot_rel @ trunk_x_horiz
        foot_y_body = foot_rel @ trunk_y_horiz
        foot_behind = jp.clip((-jp.min(foot_x_body) - 0.02) / 0.08, 0.0, 1.0)
        pitch_recovery = jp.clip(local_angvel[1] / 1.5, 0.0, 1.0)
        tip_sign_y = jp.where(
            jp.abs(grav_body[1]) > 0.05,
            jp.sign(grav_body[1]),
            -jp.sign(local_angvel[0]),
        )
        foot_side = jp.clip(
            (jp.max(foot_y_body * tip_sign_y) - 0.06) / 0.06, 0.0, 1.0
        )
        roll_recovery = jp.clip((tip_sign_y * local_angvel[0]) / 1.5, 0.0, 1.0)
        single_contact = jp.logical_xor(contact[0], contact[1]).astype(jp.float32)
        rec_imit_scale = jp.where(side_tip, 0.0, jp.where(in_push, 0.2, 1.0))
        rec_stab_scale = jp.where(in_push, 0.25, 1.0)
        rec_rate_scale = jp.where(
            standing_push, 0.05, jp.where(side_tip, 0.10, jp.where(in_push, 0.35, 1.0))
        )
        active_mask = (~in_push).astype(jp.float32)

        r_lin = reward_tracking_lin_vel(
            info["command"],
            local_linvel,
            self._config.reward_config.tracking_sigma,
            y_weight=self._config.reward_config.tracking_y_weight,
        )
        r_ang = reward_tracking_ang_vel(
            info["command"],
            local_angvel,
            self._config.reward_config.tracking_sigma_ang,
        )
        tracking_coupled = r_lin * r_ang

        rec_track_scale = jp.where(
            standing_push | side_tip, 0.0, jp.where(in_push, 0.25, 1.0)
        )
        both_contact = jp.all(contact).astype(jp.float32)
        v_xy_sq = jp.sum(jp.square(local_linvel[:2]))
        wz_sq = jp.square(local_angvel[2])
        stall_ds = jp.where(
            (cmd_norm > 0.05) & (~in_push),
            jp.exp(-v_xy_sq / 0.002) * jp.exp(-wz_sq / 0.04) * both_contact,
            0.0,
        )

        ret = {
            "tracking_lin_vel": jp.where(
                standing_push,
                jp.where(is_upright_standing, jp.exp(-v_xy_sq / 0.15), 0.0),
                r_lin * rec_track_scale,
            ),
            "tracking_ang_vel": r_ang * rec_track_scale,
            "tracking_coupled": tracking_coupled * rec_track_scale,
            "orientation": cost_orientation(gravity_vec) * rec_stab_scale,
            "ang_vel_xy": cost_ang_vel_xy(local_angvel) * rec_stab_scale,
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, prev_last_act) * rec_rate_scale,
            "stand_still": cost_stand_still(
                info["command"],
                joints_qpos,
                joints_qvel,
                self._default_actuator,
                ignore_head=False,
                active_mask=active_mask,
            ),
            "stand_still_feet": jp.where(
                (cmd_norm < 0.01) & (~in_push), 1.0 - both_contact, 0.0
            ),
            "standing_back_step": jp.where(
                back_tip,
                0.35 * single_contact + 0.45 * foot_behind + 0.20 * pitch_recovery,
                0.0,
            ),
            "standing_side_step": jp.where(
                side_tip,
                jp.where(
                    standing_push,
                    0.35 * single_contact + 0.45 * foot_side + 0.20 * roll_recovery,
                    0.55 * foot_side + 0.45 * roll_recovery,
                ),
                0.0,
            ),
            "alive": reward_alive() * (1.0 - done),
            "imitation": reward_imitation(
                self.get_floating_base_qpos(data.qpos),
                local_linvel,
                local_angvel,
                joints_qpos,
                joints_qvel,
                contact,
                info["current_reference_motion"],
                info["command"],
                USE_IMITATION_REWARD,
                recovery_scale=rec_imit_scale,
            ),
            "feet_air_time": jp.where(
                side_tip,
                0.0,
                reward_feet_air_time(
                    info["feet_air_time"],
                    first_contact,
                    jp.where(
                        back_tip,
                        info["command"].at[0].set(-0.15),
                        jp.where(
                            standing_push,
                            info["command"].at[0].set(0.15),
                            info["command"],
                        ),
                    ),
                    threshold_min=0.06,
                    threshold_max=0.20,
                    swing_peak=info["swing_peak"],
                    min_swing_height=0.008,
                ),
            ),
            "stall_ds": stall_ds,
            "head_pos": cost_head_pos(
                joints_qpos,
                joints_qvel,
                info["command"],
            )
            * active_mask,
            "dof_pos_limits": cost_joint_pos_limits(
                joints_qpos,
                self._soft_lowers,
                self._soft_uppers,
            ),
            "feet_collision": geoms_colliding(
                data, self._feet_geom_id[0], self._feet_geom_id[1]
            ).astype(jp.float32),
            "termination": cost_termination(done),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        (
            rng_mode,
            rng_sx,
            rng_sy,
            rng_sz,
            rng_x,
            rng_y,
            rng_z,
            rng_cx,
            rng_cy,
            rng_cz,
        ) = jax.random.split(rng, 10)

        mode_u = jax.random.uniform(rng_mode, minval=0.0, maxval=1.0)
        sign_x = jp.where(jax.random.bernoulli(rng_sx, 0.5), 1.0, -1.0)
        sign_y = jp.where(jax.random.bernoulli(rng_sy, 0.5), 1.0, -1.0)
        sign_z = jp.where(jax.random.bernoulli(rng_sz, 0.5), 1.0, -1.0)

        pure_x = sign_x * jax.random.uniform(rng_x, minval=0.08, maxval=0.15)
        pure_y = sign_y * jax.random.uniform(rng_y, minval=0.08, maxval=0.20)
        pure_z = sign_z * jax.random.uniform(rng_z, minval=0.25, maxval=1.0)

        comb_x = jax.random.uniform(
            rng_cx, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        comb_y = jax.random.uniform(
            rng_cy, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        comb_z = jax.random.uniform(
            rng_cz,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # 5-mode distribution:
        # 35.0% standing (0, 0, 0)
        # 17.5% pure lateral (0, sign*U(0.08, 0.20), 0)
        # 15.0% pure sagittal (sign*U(0.08, 0.15), 0, 0)
        # 10.0% pure yaw (0, 0, sign*U(0.25, 1.0))
        # 22.5% combined (U(-0.15, 0.15), U(-0.20, 0.20), U(-1.0, 1.0))
        cmd_3 = jp.where(
            mode_u < 0.35,
            jp.zeros(3),
            jp.where(
                mode_u < 0.525,
                jp.array([0.0, pure_y, 0.0]),
                jp.where(
                    mode_u < 0.675,
                    jp.array([pure_x, 0.0, 0.0]),
                    jp.where(
                        mode_u < 0.775,
                        jp.array([0.0, 0.0, pure_z]),
                        jp.array([comb_x, comb_y, comb_z]),
                    ),
                ),
            ),
        )
        return jp.hstack([cmd_3, jp.zeros(4)])
