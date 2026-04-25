import torch
import numpy as np
import matplotlib.pyplot as plt
from models.world_model import WorldModel
from env.gym_wrapper import VoxelWorldEnv
from env.voxel_world import NUM_ACTIONS

ACTION_NAMES = ["Forward", "Back", "Str Left", "Str Right", "Turn L", "Turn R"]


def branch_futures(model, state, depth=3, device="cpu"):
    if depth == 0:
        return None

    branches = {}
    with torch.no_grad():
        for action in range(NUM_ACTIONS):
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            new_state, _ = model.rssm.imagine_step(state, action_tensor)

            full_state = model.rssm.get_full_state(new_state)
            recon = model.decoder(full_state)
            frame = recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)

            children = branch_futures(model, new_state, depth - 1, device)
            branches[action] = {"frame": frame, "state": new_state, "children": children}

    return branches


def get_initial_state(model, seed=42, warmup_steps=10, device="cpu"):
    env = VoxelWorldEnv()
    obs, _ = env.reset(seed=seed)

    obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    obs_tensor = obs_tensor.to(device)

    with torch.no_grad():
        embed = model.encoder(obs_tensor)
        state = model.rssm.initial_state(1, device)
        state, _, _ = model.rssm.observe_step(
            state, torch.zeros(1, dtype=torch.long, device=device), embed
        )

        for _ in range(warmup_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            obs_tensor = obs_tensor.to(device)
            embed = model.encoder(obs_tensor)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            state, _, _ = model.rssm.observe_step(state, action_tensor, embed)

    return state, obs


def plot_multiverse(branches, root_frame, depth=2):
    if depth == 1:
        fig, axes = plt.subplots(1, NUM_ACTIONS + 1, figsize=(3 * (NUM_ACTIONS + 1), 3))
        axes[0].imshow(root_frame)
        axes[0].set_title("Current")
        axes[0].axis("off")

        for action in range(NUM_ACTIONS):
            axes[action + 1].imshow(branches[action]["frame"])
            axes[action + 1].set_title(ACTION_NAMES[action])
            axes[action + 1].axis("off")

        plt.suptitle("What happens next? (1-step lookahead)")
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(3, NUM_ACTIONS, figsize=(3 * NUM_ACTIONS, 9))

    for action in range(NUM_ACTIONS):
        branch = branches[action]

        axes[0, action].imshow(branch["frame"])
        axes[0, action].set_title(f"Step 1: {ACTION_NAMES[action]}")
        axes[0, action].axis("off")

        if branch["children"] is not None:
            child_actions = list(branch["children"].keys())
            first_child = branch["children"][child_actions[0]]
            last_child = branch["children"][child_actions[-1]]

            axes[1, action].imshow(first_child["frame"])
            axes[1, action].set_title(f"Then: {ACTION_NAMES[child_actions[0]]}")
            axes[1, action].axis("off")

            axes[2, action].imshow(last_child["frame"])
            axes[2, action].set_title(f"Then: {ACTION_NAMES[child_actions[-1]]}")
            axes[2, action].axis("off")
        else:
            axes[1, action].axis("off")
            axes[2, action].axis("off")

    plt.suptitle("Counterfactual Multiverse (2-step branching)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = WorldModel().to(device)
    model.load_state_dict(torch.load("checkpoints/world_model_epoch_100.pt", map_location=device))
    model.eval()

    state, root_frame = get_initial_state(model, seed=42, warmup_steps=10, device=device)

    depth = 2
    branches = branch_futures(model, state, depth=depth, device=device)
    plot_multiverse(branches, root_frame, depth=depth)
