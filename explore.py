import torch
import numpy as np
import matplotlib.pyplot as plt
from models.world_model import WorldModel
from env.gym_wrapper import VoxelWorldEnv
from data.replay_buffer import ReplayBuffer
from tqdm import tqdm


class WorldModelEnsemble:
    def __init__(self, num_models=3, device="cpu"):
        self.device = device
        self.models = [WorldModel().to(device) for _ in range(num_models)]

    def load_all(self, path_template="checkpoints/ensemble_{idx}_epoch_{epoch}.pt", epoch=100):
        for idx, model in enumerate(self.models):
            path = path_template.format(idx=idx, epoch=epoch)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
        print(f"Loaded {len(self.models)} ensemble members")

    def predict_frames(self, state_per_model, action):
        """
        Get predicted next frame from each model. Returns list of frames.
        """
        frames = []
        new_states = []
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for model, state in zip(self.models, state_per_model):
                state, _ = model.rssm.imagine_step(state, action_tensor)
                full_state = model.rssm.get_full_state(state)
                recon = model.decoder(full_state)
                frame = recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                frames.append(frame)
                new_states.append(state)

        return new_states, frames

    def compute_uncertainty(self, frames):
        """
        Pixel-wise variance across ensemble predictions.
        """
        stacked = np.stack(frames, axis=0)
        variance = stacked.var(axis=0)
        uncertainty = variance.mean(axis=-1)
        return uncertainty

    def initialize_states(self, start_obs):
        """
        Encode starting observation for each model.
        """
        obs_tensor = torch.from_numpy(start_obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        obs_tensor = obs_tensor.to(self.device)

        states = []
        with torch.no_grad():
            for model in self.models:
                embed = model.encoder(obs_tensor)
                state = model.rssm.initial_state(1, self.device)
                state, _, _ = model.rssm.observe_step(
                    state, torch.zeros(1, dtype=torch.long, device=self.device), embed
                )
                states.append(state)

        return states


def curiosity_action(ensemble, states, num_actions=6):
    """
    Pick the action that maximizes predicted uncertainty.
    """
    best_action = 0
    best_uncertainty = -1

    for action in range(num_actions):
        new_states, frames = ensemble.predict_frames(states, action)
        uncertainty = ensemble.compute_uncertainty(frames).mean()
        if uncertainty > best_uncertainty:
            best_uncertainty = uncertainty
            best_action = action

    return best_action, best_uncertainty


def explore_with_curiosity(ensemble, num_steps=50, seed=42):
    env = VoxelWorldEnv()
    obs, _ = env.reset(seed=seed)

    states = ensemble.initialize_states(obs)

    frames = []
    uncertainties = []
    actions_taken = []

    for step in tqdm(range(num_steps), desc="Curious exploration"):
        action, uncertainty = curiosity_action(ensemble, states)

        obs, reward, terminated, truncated, _ = env.step(action)

        new_states = []
        model_frames = []
        action_tensor = torch.tensor([action], dtype=torch.long, device=ensemble.device)
        with torch.no_grad():
            for model, state in zip(ensemble.models, states):
                state, _ = model.rssm.imagine_step(state, action_tensor)
                full_state = model.rssm.get_full_state(state)
                recon = model.decoder(full_state)
                frame = recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                model_frames.append(frame)
                new_states.append(state)

        states = new_states
        unc_map = ensemble.compute_uncertainty(model_frames)

        frames.append(obs)
        uncertainties.append(unc_map)
        actions_taken.append(action)

        if terminated or truncated:
            break

    return frames, uncertainties, actions_taken


def plot_uncertainty_timeline(frames, uncertainties, steps=None):
    if steps is None:
        num = len(frames)
        steps = [0, num // 4, num // 2, 3 * num // 4, num - 1]

    fig, axes = plt.subplots(2, len(steps), figsize=(3 * len(steps), 6))

    for col, t in enumerate(steps):
        axes[0, col].imshow(frames[t])
        axes[0, col].set_title(f"Frame t={t}")
        axes[0, col].axis("off")

        axes[1, col].imshow(uncertainties[t], cmap="hot", vmin=0)
        axes[1, col].set_title(f"Uncertainty t={t}")
        axes[1, col].axis("off")

    plt.suptitle("Curiosity-Driven Exploration: Uncertainty Maps")
    plt.tight_layout()
    plt.show()


def train_ensemble(num_models=3, num_epochs=100, batch_size=16, seq_len=50, lr=3e-4, device="cpu"):
    buffer = ReplayBuffer()
    buffer.load_from_disk()

    import os
    os.makedirs("checkpoints", exist_ok=True)

    for idx in range(num_models):
        print(f"\n--- Training ensemble member {idx+1}/{num_models} ---")
        torch.manual_seed(idx * 1000)
        model = WorldModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            num_batches = max(1, buffer.total_transitions // (batch_size * seq_len))
            epoch_loss = 0

            for _ in tqdm(range(num_batches), desc=f"Model {idx+1} Epoch {epoch+1}/{num_epochs}"):
                batch = buffer.sample_batch(batch_size, seq_len)
                obs = torch.from_numpy(batch["observations"]).float().permute(0, 1, 4, 2, 3) / 255.0
                actions = torch.from_numpy(batch["actions"]).long()
                rewards = torch.from_numpy(batch["rewards"]).float()

                loss, metrics = model(obs.to(device), actions.to(device), rewards.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                optimizer.step()
                epoch_loss += metrics["total_loss"]

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={epoch_loss / num_batches:.4f}")
                torch.save(model.state_dict(), f"checkpoints/ensemble_{idx}_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    import sys
    if "--train" in sys.argv:
        train_ensemble(device=device)
    else:
        ensemble = WorldModelEnsemble(num_models=3, device=device)
        ensemble.load_all()
        frames, uncertainties, actions = explore_with_curiosity(ensemble, num_steps=50)
        plot_uncertainty_timeline(frames, uncertainties)
