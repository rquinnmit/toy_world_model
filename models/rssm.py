import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, state_dim=256, latent_dim=256, action_dim=6,
                    num_categories=32, num_classes=32, embed_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.stoch_dim = num_categories * num_classes
        self.embed_dim = embed_dim

        self.action_embed = nn.Linear(action_dim, embed_dim)

        # GRU Input: previous stochastic state + action embedding
        self.gru = nn.GRUCell(self.stoch_dim + embed_dim, state_dim)

        # Prior: deterministic state to categorical logits
        self.prior_net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.stoch_dim)
        )

        # Posterior: deterministic state + encoded observation to categorical logits
        self.posterior_net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.stoch_dim)
        )

    def initial_state(self, batch_size, device):
        return {
            "h": torch.zeros(batch_size, self.state_dim, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device)
        }

    def observe_step(self, prev_state, action, obs_embed):
        """
        Single step with observation (training). Returns posterior sample.
        """
        h = self._deterministic_step(prev_state, action)

        # Prior: for KL loss
        prior_logits = self.prior_net(h).reshape(-1, self.num_categories, self.num_classes)

        # Posterior: uses observations
        post_input = torch.cat([h, obs_embed], dim=-1)
        post_logits = self.posterior_net(post_input).reshape(-1, self.num_categories, self.num_classes)

        z = self._sample_categorical(post_logits)
        z_flat = z.reshape(-1, self.stoch_dim)

        state = {"h": h, "z": z_flat}
        return state, prior_logits, post_logits

    def imagine_step(self, prev_state, action):
        """
        Single step without observation (imagination). Uses prior only.
        """
        h = self._deterministic_step(prev_state, action)

        prior_logits = self.prior_net(h).reshape(-1, self.num_categories, self.num_classes)

        z = self._sample_categorical(prior_logits)
        z_flat = z.reshape(-1, self.stoch_dim)

        state = {"h": h, "z": z_flat}
        return state, prior_logits

    def _deterministic_step(self, prev_state, action):
        action_one_hot = F.one_hot(action.long(), num_classes=6).float()
        action_emb = self.action_embed(action_one_hot)
        gru_input = torch.cat([prev_state["z"], action_emb], dim=-1)
        h = self.gru(gru_input, prev_state["h"])
        return h

    def _sample_categorical(self, logits):
        """
        Sample from categorical distribution with straight-through gradients.
        """
        probs = F.softmax(logits, dim=-1)
        hard = F.one_hot(probs.argmax(dim=-1), self.num_classes).float()
        return hard + probs - probs.detach()

    def get_full_state(self, state):
        """
        Concatenate h and z for use by decoder and reward predictor.
        """
        return torch.cat([state["h"], state["z"]], dim=-1)
