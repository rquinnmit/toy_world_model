import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.world_model import WorldModel
from data.replay_buffer import ReplayBuffer
import os


def train(num_epochs=100, batch_size=16, seq_len=50, lr=3e-4, device="cpu"):
    buffer = ReplayBuffer()
    buffer.load_from_disk()
    print(f"Training on {buffer.total_transitions} transitions")

    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter("runs/world_model")

    global_step = 0

    for epoch in range(num_epochs):
        epoch_metrics = {"total_loss": 0, "recon_loss": 0, "reward_loss": 0, "kl_loss": 0}
        num_batches = max(1, buffer.total_transitions // (batch_size * seq_len))

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = buffer.sample_batch(batch_size, seq_len)

            obs = torch.from_numpy(batch["observations"]).float().permute(0, 1, 4, 2, 3) / 255.0
            actions = torch.from_numpy(batch["actions"]).long()
            rewards = torch.from_numpy(batch["rewards"]).float()

            obs = obs.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)

            loss, metrics = model(obs, actions, rewards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
                writer.add_scalar(f"train/{k}", v, global_step)

            global_step += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        print(f"Epoch {epoch+1}: " + ", ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items()))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/world_model_epoch_{epoch+1}.pt")

    writer.close()
    print("Training complete.")
    return model


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    train(device=device)
