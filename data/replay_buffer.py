import numpy as np
import os


class ReplayBuffer:
    def __init__(self, storage_dir="data/episodes", max_episodes=1000):
        self.storage_dir = storage_dir
        self.max_episodes = max_episodes
        self.episodes = []
        os.makedirs(storage_dir, exist_ok=True)

    def add_episode(self, observations, actions, rewards, dones):
        episode = {
            "observations": np.array(observations, dtype=np.uint8),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
        }
        self.episodes.append(episode)

        idx = len(self.episodes) - 1
        path = os.path.join(self.storage_dir, f"episode_{idx:05d}.npz")
        np.savez_compressed(path, **episode)

        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def sample_subsequence(self, seq_len):
        ep = self.episodes[np.random.randint(len(self.episodes))]
        ep_len = len(ep["actions"])

        if ep_len <= seq_len:
            start = 0
            end = ep_len
        else:
            start = np.random.randint(0, ep_len - seq_len)
            end = start + seq_len

        return {
            "observations": ep["observations"][start:end + 1],
            "actions": ep["actions"][start:end],
            "rewards": ep["rewards"][start:end],
            "dones": ep["dones"][start:end],
        }

    def sample_batch(self, batch_size, seq_len):
        batch = [self.sample_subsequence(seq_len) for _ in range(batch_size)]
        return {
            key: np.stack([b[key] for b in batch])
            for key in batch[0].keys()
        }

    def load_from_disk(self):
        self.episodes = []
        files = sorted(f for f in os.listdir(self.storage_dir) if f.endswith(".npz"))
        for f in files:
            data = np.load(os.path.join(self.storage_dir, f))
            self.episodes.append({k: data[k] for k in data.files})
        print(f"Loaded {len(self.episodes)} episodes from {self.storage_dir}")

    @property
    def total_transitions(self):
        return sum(len(ep["actions"]) for ep in self.episodes)
