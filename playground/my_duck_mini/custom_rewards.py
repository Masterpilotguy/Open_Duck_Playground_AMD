import jax
import jax.numpy as jp

def safe_norm(x: jax.Array, axis: int = -1, keepdims: bool = False, eps: float = 1e-7) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(x), axis=axis, keepdims=keepdims) + eps)

def reward_imitation(
    base_qpos: jax.Array,
    local_linvel: jax.Array,
    local_angvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
    recovery_scale: float = 1.0,
) -> jax.Array:
    del base_qpos, local_linvel, local_angvel
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    cmd_norm = safe_norm(cmd[:3])

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # remove neck head and antennas
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    joint_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove neck head and antennas
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    joint_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]
    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )

    # joint_pos has 10 leg dims: [0, 5]=hip_yaw, [1, 6]=hip_roll
    hip_roll_w = jp.where(jp.abs(cmd[1]) > 0.08, 0.85, 1.0)
    hip_yaw_w = jp.where(jp.abs(cmd[2]) > 0.25, 0.85, 1.0)
    joint_weights = (
        jp.ones(10)
        .at[jp.array([1, 6])]
        .set(hip_roll_w)
        .at[jp.array([0, 5])]
        .set(hip_yaw_w)
    )
    err_sq = jp.sum(joint_weights * jp.square(joint_pos - ref_joint_pos))
    joint_pos_rew = (0.5 * jp.exp(-3.0 * err_sq) + 0.5 * jp.exp(-15.0 * err_sq)) * 2.0
    joint_vel_rew = (
        jp.exp(-0.03 * jp.sum(jp.square(joint_vel - ref_joint_vels))) * 0.5
    )

    exact_match = jp.all(contacts == ref_foot_contacts).astype(jp.float32)
    ref_is_single = jp.sum(ref_foot_contacts) < 1.5
    contact_rew = jp.where(ref_is_single, 1.5 * exact_match, 0.3 * exact_match)

    reward = (joint_pos_rew + joint_vel_rew + contact_rew) * recovery_scale

    reward = jp.where(cmd_norm > 0.01, reward, 0.0)
    return jp.nan_to_num(reward)