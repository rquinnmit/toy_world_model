import torch
import numpy as np
from models.world_model import WorldModel


def imagine_rollout(model, start_obs, actions, device="cpu"):
    """
    start_obs: (3, 64, 64) float tensor [0, 1]
    actions: list or 1D tensor of action ints, length T
    Returns: list of T numpy images (64, 64, 3) uint8
    """
    model.eval()
    with torch.no_grad():
        obs = start_obs.unsqueeze(0).to(device)
        embed = model.encoder(obs)

        state = model.rssm.initial_state(1, device)
        state, _, _ = model.rssm.observe_step(state, torch.zeros(1, dtype=torch.long, device=device), embed)

        frames = []
        for t in range(len(actions)):
            action = torch.tensor([actions[t]], dtype=torch.long, device=device)
            state, _ = model.rssm.imagine_step(state, action)

            full_state = model.rssm.get_full_state(state)
            recon = model.decoder(full_state)
            frame = recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frames.append(frame)

    return frames
