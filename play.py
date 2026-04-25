import torch
import numpy as np
import pygame
from models.world_model import WorldModel
from env.gym_wrapper import VoxelWorldEnv
from env.voxel_world import (
    ACTION_FORWARD, ACTION_BACKWARD, ACTION_STRAFE_LEFT,
    ACTION_STRAFE_RIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
)

KEY_MAP = {
    pygame.K_w: ACTION_FORWARD,
    pygame.K_s: ACTION_BACKWARD,
    pygame.K_a: ACTION_STRAFE_LEFT,
    pygame.K_d: ACTION_STRAFE_RIGHT,
    pygame.K_LEFT: ACTION_TURN_LEFT,
    pygame.K_RIGHT: ACTION_TURN_RIGHT,
}

DISPLAY_SCALE = 8
DISPLAY_SIZE = 64 * DISPLAY_SCALE


def play(checkpoint_path="checkpoints/world_model_epoch_100.pt", seed=42, device="cpu"):
    device = torch.device(device)

    model = WorldModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    env = VoxelWorldEnv()
    start_obs, _ = env.reset(seed=seed)
    env.close()

    obs_tensor = torch.from_numpy(start_obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    obs_tensor = obs_tensor.to(device)

    with torch.no_grad():
        embed = model.encoder(obs_tensor)
        state = model.rssm.initial_state(1, device)
        state, _, _ = model.rssm.observe_step(
            state, torch.zeros(1, dtype=torch.long, device=device), embed
        )

    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    pygame.display.set_caption("Neural Game Engine -- Playing in Imagination")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    step_count = 0
    running = True

    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in KEY_MAP:
                    action = KEY_MAP[event.key]

        if action is None:
            keys = pygame.key.get_pressed()
            for key, act in KEY_MAP.items():
                if keys[key]:
                    action = act
                    break

        if action is not None:
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.long, device=device)
                state, _ = model.rssm.imagine_step(state, action_tensor)

                full_state = model.rssm.get_full_state(state)
                recon = model.decoder(full_state)
                frame = recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)

            step_count += 1

            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (DISPLAY_SIZE, DISPLAY_SIZE))
            screen.blit(surface, (0, 0))

            step_text = font.render(f"Dream Step: {step_count}", True, (255, 255, 255))
            hint_text = font.render("WASD=move  Arrows=turn  ESC=quit", True, (200, 200, 200))
            screen.blit(step_text, (10, 10))
            screen.blit(hint_text, (10, DISPLAY_SIZE - 30))

            pygame.display.flip()

        clock.tick(15)

    pygame.quit()
    print(f"Dream ended after {step_count} steps.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    play(device=device)
