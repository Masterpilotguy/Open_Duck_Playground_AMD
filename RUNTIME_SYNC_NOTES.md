# Open_Duck_Mini_Runtime Sync Notes

Saved reference of all changes needed in `/home/masterpilotguy/Open_Duck_Mini_Runtime` to match the updated training environment in `/home/masterpilotguy/Open_Duck_Playground_AMD` (`playground/my_duck_mini`).

> **Status:** Pinned for later — no edits have been made to `Open_Duck_Mini_Runtime` yet.

---

## Summary of Required Changes (Items 1–6)

### 1. Observation Vector: `101` → `104` dims (Add `projected_gravity` at `obs[6:9]`)
- **Training reference:** `playground/my_duck_mini/joystick.py` (L719–839), `playground/my_duck_mini/mujoco_infer.py` (L107–149).
- **Runtime files:**
  - `/home/masterpilotguy/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/raw_imu.py`
  - `/home/masterpilotguy/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` (L123–172)
- **What to change:**
  - In `raw_imu.py`: BNO055 is already in `NDOF_MODE`. Read `self.imu.gravity`, guard against `None` / zero-norm, negate & normalize to unit length:
    ```python
    projected_gravity = -gravity / np.linalg.norm(gravity)
    ```
    Expose `"projected_gravity"` in `get_data()` (initialized to `[0.0, 0.0, -1.0]`).
    - **Hardware sign check required:** Run `python -m mini_bdx_runtime.raw_imu` on the robot and verify upright prints `projected_gravity ≈ [0, 0, -1]` and pitching **nose-down** makes `projected_gravity[0]` **positive** (`+X`).
  - In `v2_rl_walk_mujoco.py` (`get_obs()`): insert `imu_data["projected_gravity"]` at `obs[6:9]` and sanitize/clip:
    ```python
    obs = np.clip(
        np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0),
        -100.0,
        100.0,
    ).astype(np.float32)
    ```

#### Exact 104-Dim Observation Layout
| Slice | Dim | Field | Units / Formula |
|---|---:|---|---|
| `[0:3]` | 3 | `gyro` | Angular velocity `[wx, wy, wz]` in body frame (rad/s) |
| `[3:6]` | 3 | `accelerometer` | Proper linear accel `[ax, ay, az]` (m/s²); upright at rest = `[0, 0, +9.81]` |
| `[6:9]` | 3 | `projected_gravity` | Unit gravity in body frame `R_imu^T @ [0, 0, -1]`; upright = `[0, 0, -1]`, nose-down = `+X` |
| `[9:16]` | 7 | `command` | `[lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll]` |
| `[16:30]` | 14 | `dof_pos - init_pos` | Relative joint positions (rad) in 14-actuator order |
| `[30:44]` | 14 | `dof_vel * 0.05` | Joint velocities (rad/s) scaled by `0.05` |
| `[44:58]` | 14 | `last_action` | Raw ONNX action `a_{t-1}` (`[-1, 1]`, zeros at init) |
| `[58:72]` | 14 | `last_last_action` | Raw ONNX action `a_{t-2}` |
| `[72:86]` | 14 | `last_last_last_action` | Raw ONNX action `a_{t-3}` |
| `[86:100]` | 14 | `motor_targets` | Previous step's absolute motor targets (rad, initialized to `init_pos`) |
| `[100:102]` | 2 | `feet_contacts` | `[left_contact, right_contact]` as `0.0` / `1.0` |
| `[102:104]` | 2 | `imitation_phase` | `[cos(2π * i / 27), sin(2π * i / 27)]` when moving or `imu_tip_latch > 0`, else `[0.0, 0.0]` |

*(Observation normalization is baked into the exported ONNX graph — keep passing raw observations.)*

---

### 2. ONNX Input Dtype (`float64` → `float32`)
- **Runtime file:** `/home/masterpilotguy/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/onnx_infer.py` (L14–16)
- **What to change:** Cast `inputs` to `np.float32` in the `awd=True` path (`np.asarray(inputs, dtype=np.float32)[None, :]`) and return `np.float32` from `get_obs()`. Exported ONNX expects `tensor(float)` (`float32`), shape `[1, 104]`.

---

### 3. Action Scale: `0.25` → `0.35`
- **Training reference:** `playground/my_duck_mini/joystick.py` (L62), `playground/my_duck_mini/mujoco_infer.py` (L31).
- **Runtime file:** `/home/masterpilotguy/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` (L32, L355).
- **What to change:** Update default `action_scale` (both in `RLWalk.__init__` and CLI `-a / --action_scale`) from `0.25` to `0.35`.

---

### 4. Re-enable Motor Speed Limiting (`5.24 rad/s`)
- **Training reference:** `playground/my_duck_mini/joystick.py` (L53, L66, L550–559), `playground/my_duck_mini/mujoco_infer.py` (L14, L92, L588–599).
- **Runtime file:** `/home/masterpilotguy/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` (L292–298).
- **What to change:** Initialize `self.max_motor_velocity = 5.24` and uncomment the rate-limiting clip:
  ```python
  self.motor_targets = np.clip(
      self.motor_targets,
      self.prev_motor_targets - self.max_motor_velocity * (1 / self.control_freq),
      self.prev_motor_targets + self.max_motor_velocity * (1 / self.control_freq),
  )
  ```
  (At 50 Hz, max step delta is `0.1048` rad/step.)

---

### 5. Imitation Phase Gating, `imu_tip_latch`, and Update Order
- **Training reference:** `playground/my_duck_mini/joystick.py` (L380–402), `playground/my_duck_mini/mujoco_infer.py` (L461–493).
- **Runtime file:** `/home/masterpilotguy/Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py` (L103–109, L253–270).
- **What to change:**
  1. Update `imitation_phase` **before** calling `get_obs()` (removes 1-step lag).
  2. Add `self.imu_tip_latch = 0` and gate phase advancement when standing:
     ```python
     cmd_norm = np.linalg.norm(self.last_commands[:3])
     imu_tipped = (
         (abs(projected_gravity[0]) > 0.10)
         or (abs(projected_gravity[1]) > 0.10)
         or (abs(gyro[1]) > 0.45)
         or (abs(gyro[0]) > 0.45)
     )
     self.imu_tip_latch = 30 if imu_tipped else max(self.imu_tip_latch - 1, 0)

     if cmd_norm < 0.01 and self.imu_tip_latch == 0:
         self.imitation_i = 0
         self.imitation_phase = np.array([0.0, 0.0])
     else:
         self.imitation_i = (
             self.imitation_i
             + 1.0 * (self.phase_frequency_factor + self.phase_frequency_factor_offset)
         ) % self.PRM.nb_steps_in_period
         theta = (self.imitation_i / self.PRM.nb_steps_in_period) * 2.0 * np.pi
         self.imitation_phase = np.array([np.cos(theta), np.sin(theta)])
     ```
  3. **Verify** `/home/masterpilotguy/Open_Duck_Mini_Runtime/scripts/polynomial_coefficients.pkl` yields `nb_steps_in_period == 27` (matching `playground/my_duck_mini/data/polynomial_coefficients.pkl`: `period = 0.54`, `fps = 50`).

---

### 6. Update Default Standing Joint Angles (`init_pos`)
- **Training reference:** `playground/my_duck_mini/xmls/scene_flat_terrain_backlash.xml` (L40–79).
- **Runtime file:** `/home/masterpilotguy/Open_Duck_Mini_Runtime/mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py` (L52–69).
- **What to change:** Update `HWI.init_pos` to the exact symmetric training `home` angles:

| Idx | Joint Name | Old Runtime `init_pos` | New Training `home` |
|---:|---|---:|---:|
| 0 | `left_hip_yaw` | `0.002` | `0.0` |
| 1 | `left_hip_roll` | `0.053` | `0.0` |
| 2 | `left_hip_pitch` | `-0.63` | `-0.63` |
| 3 | `left_knee` | `1.368` | `1.37` |
| 4 | `left_ankle` | `-0.784` | `-0.74` |
| 5 | `neck_pitch` | `0.0` | `0.0` |
| 6 | `head_pitch` | `0.0` | `0.0` |
| 7 | `head_yaw` | `0.0` | `0.0` |
| 8 | `head_roll` | `0.0` | `0.0` |
| 9 | `right_hip_yaw` | `-0.003` | `0.0` |
| 10 | `right_hip_roll` | `-0.065` | `0.0` |
| 11 | `right_hip_pitch` | `0.635` | `0.63` |
| 12 | `right_knee` | `1.379` | `1.37` |
| 13 | `right_ankle` | `-0.796` | `-0.74` |

---

## Open Decision (Item 7)

### 7. Head Command Override (`v2_rl_walk_mujoco.py` L310–311)
The policy always controls all 14 joints (including the 4 neck/head servos). During normal walking (without pressing `Y`), `last_commands[3:7]` is `[0, 0, 0, 0]`, so the override is a no-op. Decide what should happen if `Y` (head-control mode) is pressed:
1. **Keep as-is:** Adds joystick head offset to `motor_targets[5:9]` and passes non-zero `last_commands[3:7]` into `obs[12:16]` (out-of-distribution for policy).
2. **Keep manual head offset, zero `obs[12:16]`:** Lets you puppet head on top of policy output without feeding out-of-distribution values into `obs[12:16]`.
3. **Disable override:** Strictly matches training/MuJoCo; policy drives all 14 joints directly.

---

## Already Aligned (No Changes Needed)
- Control frequency: **50 Hz** (`dt = 0.02` s)
- 14-actuator joint ordering
- Joint velocity scaling: `dof_vel * 0.05`
- Action history timing (`get_obs()` runs before updating `last_*_action` and `motor_targets`)
- Low-pass action filter: disabled by default in both
- Velocity command ranges: `X: [-0.15, 0.15]`, `Y: [-0.20, 0.20]`, `Yaw: [-1.0, 1.0]`
- Feet contacts: `[left, right]` as `0.0` / `1.0`
- ONNX observation normalization: baked into the ONNX model

---

## Best Trained Policies to Try on the Robot (104-dim, `action_scale = 0.35`)

All three policies below require the runtime updates in Items 1–6 above (104-dim obs with `projected_gravity`, `action_scale = 0.35`, `5.24 rad/s` motor speed limit, `imu_tip_latch` with pitch + roll, symmetric `init_pos`).

| Priority | ONNX Policy Path | Strengths & MuJoCo Benchmark | Notes |
|---:|---|---|---|
| **1 (Primary)** | `checkpoints/2026_11_23_174751_100000000_final.onnx` | **Best walking side-push resilience & lowest fall rate** (`termination: 33.2`). Standing: Front `0.6`, Back `1.0`, Right `0.8`, Left `0.8` m/s. Walking 27-phase side push: **16/27 R & 16/27 L** @ `0.5 m/s`, **14/27 R & 15/27 L** @ `0.6 m/s`, **12/27 R & 13/27 L** @ `0.7 m/s` (`25/54` total). | Most balanced all-around push recovery (symmetric L/R). |
| **2 (Backup)** | `checkpoints/2026_11_23_174751_7208960_best.onnx` | **Slightly crisper walking tracking** (`tracking_coupled: 956.7` vs `777.3`, eval reward `109.1`). Standing: Front `0.6`, Back `1.0`, Right `0.8`, Left `0.8` m/s. Walking 27-phase side push: **17/27 R & 17/27 L** @ `0.5 m/s`, **15/27 R & 14/27 L** @ `0.6 m/s`, **10/27 R & 11/27 L** @ `0.7 m/s`. | Good alternative if #1 feels slightly looser on straight-line tracking. |
| **3 (Cleanest Gait)** | `checkpoints/2026_11_23_083905_100925440_best.onnx` | **Highest overall eval reward (`115.78`) & longest episode length (`664.7`)** (`termination: 30.9`, `action_rate: 116.4`). Standing: Front `0.6`, Back `1.0`, Right `0.8`, Left `0.8` m/s. | Pre-walking-side-push fine-tune; weaker on walking side shoves (`~2–3/6` phases @ `0.7 m/s`), great baseline to compare gait smoothness on hardware. |

### Matching Orbax Checkpoints (if resuming training later)
- `checkpoints/2026_11_23_083642_100925440` (matches #3, `115.78` reward)
- `checkpoints/2026_11_23_113546_7208960` (`111.11` reward, parent of #1/#2)
- Latest 100M run final checkpoint directory in `checkpoints/2026_11_23_17*` (matches #1)
