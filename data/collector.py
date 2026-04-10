import numpy as np
from tqdm import tqdm
from env.gym_wrapper import VoxelWorldEnv
from data.replay_buffer import ReplayBuffer


def collect_episodes(num_episodes=200, max_steps=500, forward_bias=0.3, seed_offset=0):
    env = VoxelWorldEnv(max_steps=max_steps)
    buffer = ReplayBuffer()

    total_transitions = 0

    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, info = env.reset(seed=seed_offset + ep)

        observations = [obs]
        actions = []
        rewards = []
        dones = []

        for step in range(max_steps):
            if np.random.random() < forward_bias:
                action = 0
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)

            if terminated or truncated:
                break

        buffer.add_episode(observations, actions, rewards, dones)
        total_transitions += len(actions)

    print(f"Collected {num_episodes} episodes, {total_transitions} total transitions")
    return buffer


if __name__ == "__main__":
    buffer = collect_episodes(num_episodes=200)
    print(f"Buffer size: {buffer.total_transitions} transitions")

    sample = buffer.sample_batch(batch_size=4, seq_len=10)
    print(f"Sample batch shapes:")
    for key, val in sample.items():
        print(f"  {key}: {val.shape}")
