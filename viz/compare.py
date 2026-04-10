import numpy as np
import matplotlib.pyplot as plt
import imageio
from env.gym_wrapper import VoxelWorldEnv
from viz.dreamer import imagine_rollout
import torch


def compare_rollout(model, num_steps=30, seed=42, device="cpu"):
    env = VoxelWorldEnv()
    obs, _ = env.reset(seed=seed)

    real_frames = [obs]
    actions = []

    for _ in range(num_steps):
        action = env.action_space.sample()
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        real_frames.append(obs)
        if terminated or truncated:
            break

    start_obs = torch.from_numpy(real_frames[0]).float().permute(2, 0, 1) / 255.0
    imagined_frames = imagine_rollout(model, start_obs, actions, device=device)

    return real_frames[1:], imagined_frames, actions


def save_comparison_gif(real_frames, imagined_frames, path="comparison.gif", fps=5):
    frames = []
    num = min(len(real_frames), len(imagined_frames))

    for i in range(num):
        real = real_frames[i]
        imag = imagined_frames[i]
        error = np.abs(real.astype(np.float32) - imag.astype(np.float32)).astype(np.uint8)

        scale = 4
        real_up = np.repeat(np.repeat(real, scale, axis=0), scale, axis=1)
        imag_up = np.repeat(np.repeat(imag, scale, axis=0), scale, axis=1)
        error_up = np.repeat(np.repeat(error, scale, axis=0), scale, axis=1)

        combined = np.concatenate([real_up, imag_up, error_up], axis=1)
        frames.append(combined)

    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved comparison to {path}")


def plot_comparison(real_frames, imagined_frames, steps=None):
    if steps is None:
        num = min(len(real_frames), len(imagined_frames))
        steps = [0, num // 4, num // 2, 3 * num // 4, num - 1]

    fig, axes = plt.subplots(3, len(steps), figsize=(3 * len(steps), 9))

    for col, t in enumerate(steps):
        axes[0, col].imshow(real_frames[t])
        axes[0, col].set_title(f"Real t={t}")
        axes[0, col].axis("off")

        axes[1, col].imshow(imagined_frames[t])
        axes[1, col].set_title(f"Imagined t={t}")
        axes[1, col].axis("off")

        error = np.abs(real_frames[t].astype(np.float32) - imagined_frames[t].astype(np.float32))
        axes[2, col].imshow(error.astype(np.uint8))
        axes[2, col].set_title(f"Error t={t}")
        axes[2, col].axis("off")

    plt.suptitle("Real vs. Imagined Trajectories")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from models.world_model import WorldModel

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = WorldModel().to(device)
    model.load_state_dict(torch.load("checkpoints/world_model_epoch_100.pt", map_location=device))

    real, imagined, actions = compare_rollout(model, num_steps=30, device=device)
    plot_comparison(real, imagined)
    save_comparison_gif(real, imagined)
