import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder
from models.rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, state_dim=256, latent_dim=256, num_categories=32,
                num_classes=32, action_dim=6, kl_beta=1.0, kl_balance=0.8):
        super().__init__()
        self.kl_beta = kl_beta
        self.kl_balance = kl_balance

        self.encoder = Encoder(latent_dim=latent_dim)
        self.rssm = RSSM(
            state_dim=state_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_categories=num_categories,
            num_classes=num_classes
        )

        full_state_dim = state_dim + num_categories * num_classes

        self.decoder = Decoder(input_dim=full_state_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(full_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, observations, actions, rewards):
        """
        observations: (batch, seq_len+1, 3, 64, 64) float [0,1]
        actions:      (batch, seq_len) long
        rewards:      (batch, seq_len) float
        """
        batch_size, seq_len = actions.shape
        device = actions.device

        # Encode all observations at once
        obs_flat = observations.reshape(-1, 3, 64, 64)
        embeds_flat = self.encoder(obs_flat)
        embeds = embeds_flat.reshape(batch_size, seq_len + 1, -1)

        # Roll through time with RSSM
        state = self.rssm.initial_state(batch_size, device)
        all_prior_logits = []
        all_post_logits = []
        all_states = []

        for t in range(seq_len):
            state, prior_logits, post_logits = self.rssm.observe_step(
                state, actions[:, t], embeds[:, t + 1]
            )
            all_prior_logits.append(prior_logits)
            all_post_logits.append(post_logits)
            all_states.append(self.rssm.get_full_state(state))

        all_states = torch.stack(all_states, dim=1)
        all_prior_logits = torch.stack(all_prior_logits, dim=1)
        all_post_logits = torch.stack(all_post_logits, dim=1)

        # Decode predictions
        states_flat = all_states.reshape(-1, all_states.size(-1))
        recon_flat = self.decoder(states_flat)
        recon = recon_flat.reshape(batch_size, seq_len, 3, 64, 64)

        reward_pred = self.reward_head(states_flat).reshape(batch_size, seq_len)

        # LOSSES
        # Reconstruction
        target_obs = observations[:, 1:]
        recon_loss = F.mse_loss(recon, target_obs)

        # Reward Prediction
        reward_loss = F.mse_loss(reward_pred, rewards)

        # KL Divergence
        kl_loss = self._kl_loss(all_prior_logits, all_post_logits)

        total_loss = recon_loss + reward_loss + self.kl_beta * kl_loss

        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return total_loss, metrics


    def _kl_loss(self, prior_logits, post_logits):
        prior_probs = F.softmax(prior_logits, dim=-1) + 1e-8
        post_probs = F.softmax(post_logits, dim=-1) + 1e-8

        post_log = torch.log(post_probs)
        prior_log = torch.log(prior_probs)

        # KL(posterior || prior)
        kl = (post_probs * (post_log - prior_log)).sum(dim=-1).mean()

        # KL balancing
        kl_balanced = (
            self.kl_balance * (post_probs.detach() * (post_log.detach() - prior_log)).sum(dim=-1).mean()
            + (1 - self.kl_balance) * (post_probs * (post_log - prior_log.detach())).sum(dim=-1).mean()
        )
        
        return kl_balanced