"""Minimal RSSM world model (~100M params), Dreamer-style but stripped down.

- Encoder: stride-2 CNN -> embedding
- RSSM: GRU deterministic state h + Gaussian stochastic latent z
  posterior q(z|h, enc) / prior p(z|h), KL balancing
- Decoder: deconv reconstructing the frame from (h, z)
- Reward head: MLP predicting reward from (h, z)

Loss = reconstruction (Gaussian NLL) + beta * KL + reward MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.clamp(torch.abs(x), max=20))


class Encoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        ch = 96
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.ELU(),          # 128 -> 64
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ELU(),     # 64 -> 32
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.ELU(),  # 32 -> 16
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.ELU(),  # 16 -> 8
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.ELU(),  # 8 -> 4
        )
        self.fc = nn.Linear(ch * 4 * 4 * 4, embed_dim)

    def forward(self, x):  # x: (B, 3, H, W) in [-1, 1]
        h = self.net(x)
        return self.fc(h.flatten(1))


class Decoder(nn.Module):
    def __init__(self, deter_dim, stoch_dim, ch=896):
        super().__init__()
        self.fc = nn.Linear(deter_dim + stoch_dim, ch * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.ELU(),   # 4 -> 8
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1), nn.ELU(),  # 8 -> 16
            nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1), nn.ELU(),  # 16 -> 32
            nn.ConvTranspose2d(ch // 8, ch // 16, 4, 2, 1), nn.ELU(),  # 32 -> 64
            nn.ConvTranspose2d(ch // 16, 3, 4, 2, 1),             # 64 -> 128
        )

    def forward(self, h, z):  # (B, deter), (B, stoch)
        x = self.fc(torch.cat([h, z], dim=-1))
        return self.net(x.view(x.shape[0], -1, 4, 4))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class WorldModel(nn.Module):
    def __init__(self, action_dim=7, embed_dim=768, deter_dim=3072,
                 stoch_dim=32, beta=0.15):
        super().__init__()
        self.beta = beta
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(deter_dim, stoch_dim)

        in_dim = stoch_dim + action_dim
        self.gru = nn.GRUCell(in_dim, deter_dim)
        self.prior = MLP(deter_dim, 256, stoch_dim * 2)
        self.post = MLP(deter_dim + embed_dim, 256, stoch_dim * 2)

        self.reward = MLP(deter_dim + stoch_dim, 256, 1)

    # -- single RSSM step -------------------------------------------------
    def _gauss(self, x):
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = log_std.exp()
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1)

    def _gauss_stop(self, d):
        """Same distribution with gradients stopped (for KL balancing)."""
        return torch.distributions.Independent(
            torch.distributions.Normal(d.mean.detach(),
                                       d.stddev.detach()), 1)

    def observe_step(self, h, z, action, embed):
        """Posterior step: p = prior(h), q = posterior(h, embed), new h."""
        p = self._gauss(self.prior(h))
        q = self._gauss(self.post(torch.cat([h, embed], dim=-1)))
        z = q.rsample()
        x = torch.cat([z, action], dim=-1)
        h = self.gru(x, h)
        return h, z, p, q

    # -- full-sequence training forward -----------------------------------
    def forward(self, obs, actions):
        """obs: (B, T, 3, H, W) in [-1,1]; actions: (B, T-1, A).
        Returns dict of losses (already averaged)."""
        B, T = obs.shape[:2]
        embed = self.encoder(obs.flatten(0, 1)).unflatten(0, (B, T))

        h = torch.zeros(B, self.deter_dim, device=obs.device)
        z = torch.zeros(B, self.stoch_dim, device=obs.device)

        rec_loss, kl_loss, rew_loss = 0.0, 0.0, 0.0
        for t in range(T):
            if t > 0:
                h, z, p, q = self.observe_step(h, z, actions[:, t - 1],
                                               embed[:, t])
                # KL balancing: slow prior, fast posterior
                p_det = self._gauss_stop(p)
                q_det = self._gauss_stop(q)
                kl_loss = kl_loss + 0.8 * torch.distributions.kl_divergence(
                    p_det, q).mean() + 0.2 * torch.distributions.kl_divergence(
                    p, q_det).mean()
            recon = self.decoder(h, z)
            rec_loss = rec_loss + F.mse_loss(recon, obs[:, t])

        return {"rec": rec_loss / T, "kl": kl_loss / (T - 1),
                "total": rec_loss / T + self.beta * kl_loss / (T - 1)}

    @torch.no_grad()
    def imagine(self, obs0, actions):
        """Open-loop rollout from one frame under an action sequence.
        obs0: (B, 3, H, W); actions: (B, T, A). Returns frames (B, T+1, 3, H, W)."""
        B, T = actions.shape[:2]
        embed = self.encoder(obs0)
        h = torch.zeros(B, self.deter_dim, device=obs0.device)
        q = self._gauss(self.post(torch.cat([h, embed], dim=-1)))
        z = q.rsample()
        frames = [self.decoder(h, z)]
        for t in range(T):
            x = torch.cat([z, actions[:, t]], dim=-1)
            h = self.gru(x, h)
            p = self._gauss(self.prior(h))
            z = p.sample()
            frames.append(self.decoder(h, z))
        return torch.stack(frames, dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
