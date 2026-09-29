"""Train the minimal RSSM world model on LIBERO demos.
Example:
    python train.py --data ../datasets_libero/libero_goal --steps 20000 --batch 32
"""
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import LiberoDataset
from world_model import WorldModel, count_params
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../datasets_libero/libero_goal")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=17)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--out", default="wm.pt")
    p.add_argument("--smoke", action="store_true", help="run 5 steps and exit")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LiberoDataset(args.data, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    model = WorldModel().to(device)
    n = count_params(model)
    print(f"params: {n/1e6:.1f}M | episodes: {len(ds.episodes)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 5 if args.smoke else args.steps
    it = iter(dl)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            obs, act = next(it)
        except StopIteration:
            it = iter(dl)
            obs, act = next(it)
        obs, act = obs.to(device, non_blocking=True), act.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            losses = model(obs, act)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:>6} | rec {losses['rec'].item():.4f} "
                  f"| kl {losses['kl'].item():.4f} "
                  f"| {100*(time.time()-t0)/step:.0f} ms/it")
        if args.smoke and step >= 5:
            break
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")
    # quick sanity: open-loop imagination from one real frame
    model.eval()
    obs, act = next(iter(dl))
    obs, act = obs[:2, :1].to(device), act[:2, :8].to(device)
    frames = model.imagine(obs.squeeze(1), act)
    print("imagine output:", tuple(frames.shape))

if __name__ == "__main__":
    main()
