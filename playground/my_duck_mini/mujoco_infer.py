import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.my_duck_mini.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.35

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        policy_session = getattr(self.policy, "session", None) or getattr(
            self.policy, "ort_session", None
        )
        if policy_session is not None:
            try:
                expected_obs_dim = policy_session.get_inputs()[0].shape[-1]
                if expected_obs_dim == 101 or (
                    isinstance(expected_obs_dim, int) and expected_obs_dim != 104
                ):
                    print(
                        f"\n[WARNING] Loaded ONNX policy expects {expected_obs_dim} obs dims, "
                        f"but MjInfer outputs 104 dims (including 3-dim projected gravity). "
                        f"Please use a 104-dim policy (101-dim policies are incompatible)."
                    )
            except Exception:
                pass

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Independent multi-axis movement commands
        self.cmd_lin_vel_x = 0.0
        self.cmd_lin_vel_y = 0.0
        self.cmd_ang_vel = 0.0

        # Automatic heading lock (corrects rotational yaw drift)
        self.heading_hold = True
        self.heading_kp = 1.5
        self.target_yaw = None

        # Automatic straight-line course hold & lateral trim (corrects body-line crab drift)
        self.course_hold = True
        self.course_kp = 0.8
        self.track_origin = None
        self.track_yaw = None
        self.lateral_trim = 0.0

        self.imitation_i = 0
        self.imitation_phase = np.array([0.0, 0.0])
        self.imu_tip_latch = 0
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        # Push / stumble testing (thread-safe queue from GLFW callback)
        self.pending_push = None
        self.push_recovery_hold_steps = 0
        self.pending_reset = False
        self.fall_notified = False

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        gravity = self.get_gravity(data)
        # Training does not apply an x-axis bias; keep inference observation aligned.

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        if self.backlash_joint_qpos_addr:
            joint_backlash_raw = self.get_actuator_backlash_qpos(data.qpos).reshape(-1)
            joint_backlash = np.zeros(self.num_dofs)
            joint_backlash[self._backlash_actuator_idx] = joint_backlash_raw
            joint_angles = joint_angles + joint_backlash
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)

    def get_current_yaw(self, data):
        """Compute the robot's current yaw angle in radians around world Z-axis."""
        fb_qpos = self.get_floating_base_qpos(data.qpos)
        w, x, y, z = fb_qpos[3:7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def reset_environment(self):
        """Reset the robot and simulation state back to the initial upright standing configuration."""
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.model.keyframe("home").qpos

        self.data.ctrl[:] = self.default_actuator
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()

        # Run forward kinematics to initialize contact frames and sensors
        mujoco.mj_forward(self.model, self.data)

        # Reset action history buffers
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        # Reset velocity and motion commands to stopped
        self.cmd_lin_vel_x = 0.0
        self.cmd_lin_vel_y = 0.0
        self.cmd_ang_vel = 0.0
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.track_origin = None
        self.track_yaw = None
        self.target_yaw = self.get_current_yaw(self.data)
        self.imitation_i = 0
        self.imitation_phase = np.array([0.0, 0.0])
        self.imu_tip_latch = 0
        self.pending_push = None
        self.push_recovery_hold_steps = 0
        self.fall_notified = False

        print("\n[RESET] Robot restored to upright standing position (Home pose).")

    def key_callback(self, keycode):
        # Reset Environment: 'R' (82) / 'Backspace' (259)
        if keycode in (82, 259):
            self.pending_reset = True
            print("\n[RESET REQUESTED] Resetting to Home standing pose...")
            return

        # Interactive shove / stumble testing
        lin_push = 0.70
        ang_push = 0.80

        # Shove Forward: 'F' / 'f' (70) or '1' (49)
        if keycode in (70, 49):
            self.pending_push = (lin_push, 0.0, 0.0, 0.0, ang_push)
            print(f"\n[PUSH] Shove Forward (+Vx={lin_push:.2f} m/s, +Wpitch={ang_push:.2f} rad/s)")
            return

        # Shove Backward: 'B' / 'b' (66) or '2' (50)
        if keycode in (66, 50):
            self.pending_push = (-lin_push, 0.0, 0.0, 0.0, -ang_push)
            print(f"\n[PUSH] Shove Backward (-Vx=-{lin_push:.2f} m/s, -Wpitch=-{ang_push:.2f} rad/s)")
            return

        # Shove Left: 'G' / 'g' (71) or '3' (51)
        if keycode in (71, 51):
            self.pending_push = (0.0, lin_push, 0.0, -ang_push, 0.0)
            print(f"\n[PUSH] Shove Left (+Vy={lin_push:.2f} m/s, -Wroll=-{ang_push:.2f} rad/s)")
            return

        # Shove Right: 'J' / 'j' (74) or '4' (52)
        if keycode in (74, 52):
            self.pending_push = (0.0, -lin_push, 0.0, ang_push, 0.0)
            print(f"\n[PUSH] Shove Right (-Vy=-{lin_push:.2f} m/s, +Wroll={ang_push:.2f} rad/s)")
            return

        # Random Multi-Axis Stumble Shove: 'V' / 'v' (86) or '5' (53)
        if keycode in (86, 53):
            theta = np.random.uniform(0, 2 * np.pi)
            push_mag = np.random.uniform(0.20, 0.85)
            ang_mag = np.random.uniform(0.30, 0.90)
            dvx = float(np.cos(theta) * push_mag)
            dvy = float(np.sin(theta) * push_mag)
            dw_roll = float(-np.sin(theta) * ang_mag)
            dw_pitch = float(np.cos(theta) * ang_mag)
            dvz = float(np.random.uniform(-0.06, 0.06))
            self.pending_push = (dvx, dvy, dvz, dw_roll, dw_pitch)
            print(
                f"\n[PUSH] Random Stumble Shove (Vx={dvx:+.2f}, Vy={dvy:+.2f}, "
                f"Wroll={dw_roll:+.2f}, Wpitch={dw_pitch:+.2f})"
            )
            return

        if keycode == 72:  # 'h' / 'H'
            self.head_control_mode = not self.head_control_mode
            print(f"\n[MODE] {'HEAD CONTROL' if self.head_control_mode else 'LOCOMOTION'} mode active")
            return

        if keycode == 84:  # 't' / 'T'
            self.heading_hold = not self.heading_hold
            self.target_yaw = self.get_current_yaw(self.data)
            status = "ENABLED" if self.heading_hold else "DISABLED"
            print(f"\n[HEADING HOLD] {status} (Target: {np.degrees(self.target_yaw):.1f}°)")
            return

        if keycode in (32, 88):  # Spacebar or 'x' / 'X'
            self.cmd_lin_vel_x = 0.0
            self.cmd_lin_vel_y = 0.0
            self.cmd_ang_vel = 0.0
            self.track_origin = None
            self.track_yaw = None
            self.target_yaw = self.get_current_yaw(self.data)
            print("\n[CMD] STOP ALL (Vx=0, Vy=0, Wz=0)")
            return

        if not self.head_control_mode:
            # Longitudinal: Up Arrow (265) / 'w' (87) / Down Arrow (264) / 's' (83)
            if keycode in (265, 87):
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_lin_vel_x, self.COMMANDS_RANGE_X[1]):
                    self.cmd_lin_vel_x = 0.0
                else:
                    self.cmd_lin_vel_x = self.COMMANDS_RANGE_X[1]
            elif keycode in (264, 83):
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_lin_vel_x, self.COMMANDS_RANGE_X[0]):
                    self.cmd_lin_vel_x = 0.0
                else:
                    self.cmd_lin_vel_x = self.COMMANDS_RANGE_X[0]

            # Lateral: Left Arrow (263) / Right Arrow (262)
            elif keycode == 263:
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_lin_vel_y, self.COMMANDS_RANGE_Y[1]):
                    self.cmd_lin_vel_y = 0.0
                else:
                    self.cmd_lin_vel_y = self.COMMANDS_RANGE_Y[1]
            elif keycode == 262:
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_lin_vel_y, self.COMMANDS_RANGE_Y[0]):
                    self.cmd_lin_vel_y = 0.0
                else:
                    self.cmd_lin_vel_y = self.COMMANDS_RANGE_Y[0]

            # Turning: 'a' (65) / 'q' (81) / 'd' (68) / 'e' (69)
            elif keycode in (65, 81):
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_ang_vel, self.COMMANDS_RANGE_THETA[1]):
                    self.cmd_ang_vel = 0.0
                else:
                    self.cmd_ang_vel = self.COMMANDS_RANGE_THETA[1]
            elif keycode in (68, 69):
                self.track_origin = None
                self.track_yaw = None
                if np.isclose(self.cmd_ang_vel, self.COMMANDS_RANGE_THETA[0]):
                    self.cmd_ang_vel = 0.0
                else:
                    self.cmd_ang_vel = self.COMMANDS_RANGE_THETA[0]

            # Course Lock toggle: 'C' (67)
            elif keycode == 67:
                self.course_hold = not self.course_hold
                self.track_origin = None
                self.track_yaw = None
                print(f"\n[COURSE HOLD] {'ENABLED (tracks body-centerline)' if self.course_hold else 'DISABLED'}")

            # Lateral Trim: 'U' (85) / '[' (91) nudge left; 'I' (73) / ']' (93) nudge right; 'O' (79) / '\' (92) reset
            elif keycode in (85, 91):
                self.lateral_trim += 0.01
                print(f"[TRIM] Lateral trim: {self.lateral_trim:+.2f} m/s (nudge left)")
            elif keycode in (73, 93):
                self.lateral_trim -= 0.01
                print(f"[TRIM] Lateral trim: {self.lateral_trim:+.2f} m/s (nudge right)")
            elif keycode in (79, 92):
                self.lateral_trim = 0.0
                print(f"[TRIM] Lateral trim reset to 0.00 m/s")

            elif keycode == 80:  # 'p'
                self.phase_frequency_factor += 0.1
                print(f"[CADENCE] Phase factor: {self.phase_frequency_factor:.1f}")
            elif keycode in (59, 77):  # 'm' / ';'
                self.phase_frequency_factor -= 0.1
                print(f"[CADENCE] Phase factor: {self.phase_frequency_factor:.1f}")

            # Print active velocity state
            hh_target = f"{np.degrees(self.target_yaw):.1f}°" if self.target_yaw is not None else "Pending"
            hh_str = f"ON (Target: {hh_target})" if self.heading_hold else "OFF"
            ch_str = "ON" if self.course_hold else "OFF"
            print(
                f"[CMD] Vx: {self.cmd_lin_vel_x:+.2f} m/s | "
                f"Vy: {self.cmd_lin_vel_y:+.2f} (trim: {self.lateral_trim:+.2f}) | "
                f"Wz_user: {self.cmd_ang_vel:+.2f} rad/s | "
                f"HeadingHold: {hh_str} | CourseHold: {ch_str}"
            )
        else:
            neck_pitch = 0.0
            head_pitch = 0.0
            head_yaw = 0.0
            head_roll = 0.0
            if keycode == 265:  # arrow up
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode in (81, 65):  # a / q
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode in (69, 68):  # e / d
                head_roll = self.HEAD_ROLL_RANGE[0]

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll
            print(f"[HEAD] pitch={head_pitch:+.2f}, yaw={head_yaw:+.2f}, roll={head_roll:+.2f}")

    def _apply_pending_push(self):
        """Apply pending body-frame push by rotating XY linear impulse into world frame."""
        if self.pending_push is None:
            return
        dvx_b, dvy_b, dvz, dwx_b, dwy_b = self.pending_push
        current_yaw = self.get_current_yaw(self.data)
        cos_y, sin_y = np.cos(current_yaw), np.sin(current_yaw)
        dvx_w = cos_y * dvx_b - sin_y * dvy_b
        dvy_w = sin_y * dvx_b + cos_y * dvy_b
        addr = self._floating_base_qvel_addr
        self.data.qvel[addr : addr + 3] += [dvx_w, dvy_w, dvz]
        self.data.qvel[addr + 3 : addr + 5] += [dwx_b, dwy_b]
        self.pending_push = None
        self.push_recovery_hold_steps = 75

    def _update_step_commands(self):
        """Update active commands with heading hold, course hold, and push-recovery pause."""
        current_yaw = self.get_current_yaw(self.data)
        curr_pos = self.data.qpos[:2]

        if self.push_recovery_hold_steps > 0:
            self.track_origin = None
            self.track_yaw = None
            self.target_yaw = current_yaw
            active_ang_vel = self.cmd_ang_vel if abs(self.cmd_ang_vel) > 1e-3 else 0.0
            v_y_corr = 0.0
            self.push_recovery_hold_steps -= 1
        else:
            # 1. Determine active angular velocity command (Heading Lock)
            if abs(self.cmd_ang_vel) > 1e-3:
                active_ang_vel = self.cmd_ang_vel
                self.target_yaw = current_yaw
                self.track_origin = None
                self.track_yaw = None
            elif (
                (abs(self.cmd_lin_vel_x) > 1e-3 or abs(self.cmd_lin_vel_y) > 1e-3)
                and abs(self.cmd_ang_vel) <= 1e-3
                and self.push_recovery_hold_steps == 0
                and self.heading_hold
            ):
                if self.target_yaw is None:
                    self.target_yaw = current_yaw
                yaw_diff = self.target_yaw - current_yaw
                yaw_diff = (yaw_diff + np.pi) % (2.0 * np.pi) - np.pi
                active_ang_vel = float(
                    np.clip(self.heading_kp * yaw_diff, -0.6, 0.6)
                )
            else:
                active_ang_vel = 0.0
                self.target_yaw = current_yaw

            # 2. Determine active lateral velocity command (Course Hold & Lateral Trim)
            v_y_corr = 0.0
            if abs(self.cmd_lin_vel_y) > 1e-3:
                self.track_origin = None
                self.track_yaw = None
            elif abs(self.cmd_lin_vel_x) > 1e-3 and abs(self.cmd_ang_vel) <= 1e-3:
                if self.course_hold:
                    if self.track_origin is None:
                        self.track_origin = curr_pos.copy()
                        self.track_yaw = current_yaw

                    dx = curr_pos[0] - self.track_origin[0]
                    dy = curr_pos[1] - self.track_origin[1]
                    # Cross-track error perpendicular to line of travel (+ = left, - = right)
                    e_lat = -dx * np.sin(self.track_yaw) + dy * np.cos(self.track_yaw)
                    v_y_corr = float(np.clip(-self.course_kp * e_lat, -0.06, 0.06))
            else:
                self.track_origin = None
                self.track_yaw = None

        active_lin_vel_y = float(
            np.clip(
                self.cmd_lin_vel_y + self.lateral_trim + v_y_corr,
                self.COMMANDS_RANGE_Y[0],
                self.COMMANDS_RANGE_Y[1],
            )
        )

        self.commands[0] = self.cmd_lin_vel_x
        self.commands[1] = active_lin_vel_y
        self.commands[2] = active_ang_vel

    def _update_imitation_phase(self):
        if self.standing:
            return
        cmd_norm = float(np.linalg.norm(self.commands[:3]))
        grav_imu = self.get_gravity(self.data)
        gyro_imu = self.get_gyro(self.data)
        imu_tipped = (
            (abs(float(grav_imu[0])) > 0.10)
            or (abs(float(grav_imu[1])) > 0.10)
            or (abs(float(gyro_imu[1])) > 0.45)
            or (abs(float(gyro_imu[0])) > 0.45)
        )
        self.imu_tip_latch = 30 if imu_tipped else max(self.imu_tip_latch - 1, 0)
        if cmd_norm < 0.01 and self.imu_tip_latch == 0:
            self.imitation_i = 0
            self.imitation_phase = np.array([0.0, 0.0])
        else:
            self.imitation_i = (
                self.imitation_i + 1.0 * self.phase_frequency_factor
            ) % self.PRM.nb_steps_in_period
            self.imitation_phase = np.array(
                [
                    np.cos(
                        self.imitation_i
                        / self.PRM.nb_steps_in_period
                        * 2
                        * np.pi
                    ),
                    np.sin(
                        self.imitation_i
                        / self.PRM.nb_steps_in_period
                        * 2
                        * np.pi
                    ),
                ]
            )

    def run(self):
        print("\n" + "=" * 65)
        print("  MUJOCO INFERENCE - MULTI-INPUT & HEADING LOCK READY")
        print("=" * 65)
        print("  Locomotion Controls (Independent Toggle Keys):")
        print("    [R / Backspace]: Reset Environment to Home Standing Pose")
        print("    [Up / W]      : Toggle Forward  (+0.15 m/s)")
        print("    [Down / S]    : Toggle Backward (-0.15 m/s)")
        print("    [Left]        : Toggle Strafe Left (+0.20 m/s)")
        print("    [Right]       : Toggle Strafe Right (-0.20 m/s)")
        print("    [A / Q]       : Toggle Turn Left (+1.0 rad/s)")
        print("    [D / E]       : Toggle Turn Right (-1.0 rad/s)")
        print("    [Space / X]   : Full Stop (Zero all movement)")
        print("    [T]           : Toggle Heading Hold (Default: ON)")
        print("    [C]           : Toggle Course Hold (Straight-Line Body Track, Default: ON)")
        print("    [U / []       : Lateral Trim Left (+0.01 m/s)")
        print("    [I / ]]       : Lateral Trim Right (-0.01 m/s)")
        print("    [O / \\]       : Reset Lateral Trim (0.00 m/s)")
        print("    [H]           : Toggle Head Control Mode")
        print("    [P / M]       : Cadence / Step Frequency (+/- 0.1)")
        print("  Interactive Perturbation / Shove Controls:")
        print("    [F / 1]       : Shove Forward  (+Vx = +0.70 m/s, +Wpitch = +0.80 rad/s)")
        print("    [B / 2]       : Shove Backward (-Vx = -0.70 m/s, -Wpitch = -0.80 rad/s)")
        print("    [G / 3]       : Shove Left     (+Vy = +0.70 m/s, -Wroll = -0.80 rad/s)")
        print("    [J / 4]       : Shove Right    (-Vy = -0.70 m/s, +Wroll = +0.80 rad/s)")
        print("    [V / 5]       : Random Multi-Axis Stumble Shove")
        print("    [R / Backspace]: Reset Environment to Home Standing Pose")
        print("=" * 65 + "\n")

        self.target_yaw = self.get_current_yaw(self.data)

        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                # Ensure all geometry groups are visible (MuJoCo GLFW handler binds 0-5 to toggle geomgroups)
                viewer.opt.geomgroup[0] = 1  # Floor / static geoms
                viewer.opt.geomgroup[1] = 1
                viewer.opt.geomgroup[2] = 1  # Visual meshes
                viewer.opt.geomgroup[3] = 1  # Feet / collision geoms

                counter = 0
                step_start = time.time()
                control_dt = self.model.opt.timestep * self.decimation
                while viewer.is_running():
                    if self.pending_reset:
                        with viewer.lock():
                            self.reset_environment()
                            self.pending_reset = False
                            counter = 0

                    if counter % self.decimation == 0:
                        step_start = time.time()

                    if self.pending_push is not None:
                        with viewer.lock():
                            self._apply_pending_push()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        # Fall detection notice
                        root_z = self.data.qpos[self._floating_base_qpos_addr + 2]
                        if not self.fall_notified and root_z < 0.075:
                            print("\n[FALL DETECTED] Robot has fallen over! Press [R] or [Backspace] to reset.")
                            self.fall_notified = True

                        self._update_step_commands()
                        self._update_imitation_phase()
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                        # Keep floor and robot visual meshes explicitly visible even if 0-5 pressed in viewer
                        viewer.opt.geomgroup[0] = 1
                        viewer.opt.geomgroup[2] = 1
                        viewer.sync()

                        time_until_next_step = control_dt - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/my_duck_mini/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/my_duck_mini/xmls/scene_flat_terrain_backlash.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    )
    mjinfer.run()
