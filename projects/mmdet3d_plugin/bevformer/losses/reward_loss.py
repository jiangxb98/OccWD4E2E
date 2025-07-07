import torch
from typing import Dict


def compute_im_reward_loss(
    gt_trajectory: torch.Tensor,  # bs, num_traj, 2/3
    prediction_rewards,  # bs, num_traj, 1
    trajectory_samples,  # bs, sample_traj_nums, 2/3
) -> torch.Tensor:
    """
    Compute the reward loss for the reward model.
    Only use the last frame of the trajectory to calculate the reward loss
    Args:
        gt_trajectory: torch.Tensor
        prediction_rewards: torch.Tensor
        trajectory_samples: torch.Tensor
    Returns:
        torch.Tensor: The reward loss.
    """
    Bz = gt_trajectory.shape[0]
    # Get target trajectory
    if gt_trajectory.dim() == 3:
        target_trajectory = gt_trajectory[:, -1, :]  # the last frame
    else:
        target_trajectory = gt_trajectory
    
    target_trajectory = target_trajectory.reshape(Bz, -1).unsqueeze(1).float()

    # Calculate L2 distance between each of the 256 predefined trajectories and the target trajectory
    num_trajs = trajectory_samples.shape[1]
    trajectory_samples = trajectory_samples.reshape(num_trajs, -1).unsqueeze(0).repeat(Bz, 1, 1).to(target_trajectory.device)
    l2_distances = torch.cdist(trajectory_samples[:, :, :2], target_trajectory[:, :, :2], p=2)  # Shape: [batch_size, 256]
    l2_distances = l2_distances.squeeze(-1)

    # Apply softmax to L2 distances to get reward targets
    reward_targets = torch.softmax(-l2_distances, dim=-1)  # Shape: [batch_size, 256]

    # Compute loss using cross-entropy
    prediction_rewards = prediction_rewards.squeeze(-1).clamp(1e-6, 1 - 1e-6)
    im_reward_loss = -torch.sum(reward_targets * prediction_rewards.log()) / Bz

    # 输出reward_targets最大值对应的预测的轨迹索引
    max_reward_idx = torch.argmax(reward_targets, dim=-1)

    return im_reward_loss, reward_targets


def compute_sim_reward_loss(
    sim_reward,
    predicted_rewards):

    epsilon = 1e-6
    # Load precomputed target rewards
    batch_size = sim_reward.shape[0]
    if sim_reward.dim() == 3:
        target_rewards = sim_reward[:, -1] # the last frame
    else:
        target_rewards = sim_reward
    
    assert sim_reward.shape == predicted_rewards.shape, f'sim_reward.shape: {sim_reward.shape}, predicted_rewards.shape: {predicted_rewards.shape}'

    # Compute loss using binary cross-entropy # 5 is the number of metrics
    sim_reward_loss = -torch.mean(
        target_rewards * (predicted_rewards + epsilon).log() + (1 - target_rewards) * (1 - predicted_rewards + epsilon).log()
    )

    return sim_reward_loss