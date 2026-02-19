from pathlib import Path
import os
import argparse
from tensorboard.backend.event_processing import event_accumulator


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


def _parse_args():
    p = argparse.ArgumentParser(description="Plot TensorBoard scalars from latest IsaacLab run.")
    p.add_argument(
        "--base",
        type=str,
        default=str(Path("~/work2/IsaacLab/logs/rsl_rl/cartpole_direct").expanduser()),
        help="Base directory containing run folders.",
    )
    p.add_argument("--run", type=str, default=None, help="Specific run directory to plot (overrides --base latest).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (.png). Default: <run>/plots.png",
    )
    p.add_argument("--show", action="store_true", default=False, help="Show the plot interactively (requires DISPLAY).")
    return p.parse_args()


args = _parse_args()

# Use a non-interactive backend when running headless (prevents blocking on plt.show()).
if not args.show or not _has_display():
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# === run path ===
if args.run is not None:
    latest = Path(args.run).expanduser()
else:
    base = Path(args.base).expanduser()
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"Aucun run trouvé dans {base}")
    latest = max(runs, key=lambda p: p.stat().st_mtime)

print(f"[INFO] Latest run: {latest}")

# === charger events ===
ea = event_accumulator.EventAccumulator(str(latest))
ea.Reload()

print("[INFO] Scalars disponibles:")
available_scalars = set(ea.Tags().get("scalars", []))
print(sorted(list(available_scalars)))

def get_scalar(tag):
    s = ea.Scalars(tag)
    x = [e.step for e in s]
    y = [e.value for e in s]
    return x, y

# === données ===
data = {
    "Train/mean_reward": ("Mean reward", "Reward"),
    "Train/mean_reward/time": ("Mean reward (time)", "Reward"),
    "Train/mean_episode_length": ("Mean episode length", "Steps"),
    "Train/mean_episode_length/time": ("Mean episode length (time)", "Steps"),
    "Policy/mean_noise_std": ("Mean action noise std", "Std"),
    "Loss/value_function": ("Value function loss", "Loss"),
    "Loss/surrogate": ("Surrogate loss", "Loss"),
    "Loss/entropy": ("Entropy loss", "Loss"),
    "Loss/learning_rate": ("Learning rate", "LR"),
    "Perf/collection time": ("Collection time", "Time (s)"),
    "Perf/learning_time": ("Learning time", "Time (s)"),
    "Perf/total_fps": ("Total FPS", "FPS"),
}

# Keep only scalars that exist in this run
data = {k: v for k, v in data.items() if k in available_scalars}
if not data:
    raise RuntimeError("Aucun des tags attendus n'est disponible dans ce run.")

# === subplots automatiques ===
n = len(data)
cols = 3
rows = (n + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axs = axs.flatten()
fig.suptitle("Apprentissage PPO – Isaac Lab (tous les signaux)", fontsize=14)

for ax, (tag, (title, ylabel)) in zip(axs, data.items()):
    x, y = get_scalar(tag)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True)

# cacher les axes inutilisés
for ax in axs[len(data):]:
    ax.axis("off")

plt.tight_layout()

out_path = Path(args.out).expanduser() if args.out else (latest / "plots.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"[INFO] Saved plot to: {out_path}")

if args.show:
    if _has_display():
        plt.show()
    else:
        print("[WARN] --show demandé mais DISPLAY absent (headless). Image sauvegardée uniquement.")


