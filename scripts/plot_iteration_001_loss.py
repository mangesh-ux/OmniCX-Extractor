"""
Plot training loss curve for iteration 001 (from WSL terminal logs).
Run from repo root: python scripts/plot_iteration_001_loss.py
"""
from pathlib import Path

# Loss per step (steps 1--150 from iteration 001 training log)
LOSS = [
    2.154, 1.993, 2.022, 1.999, 2.059, 1.999, 1.75, 1.779, 1.616, 1.592,
    1.583, 1.466, 1.484, 1.325, 1.242, 1.212, 1.22, 1.084, 1.057, 1.041,
    1.018, 0.9102, 0.8995, 0.9836, 0.9458, 0.9693, 0.9665, 0.9328, 0.858, 0.8086,
    0.7924, 0.8857, 0.8083, 0.8047, 0.8504, 0.7946, 0.853, 0.7414, 0.8879, 0.7552,
    0.7493, 0.7123, 0.776, 0.7094, 0.6734, 0.7879, 0.815, 0.7316, 0.7883, 0.915,
    0.8375, 0.6868, 0.6582, 0.6933, 0.77, 0.8333, 0.746, 0.6856, 0.6507, 0.6546,
    0.6869, 0.6901, 0.64, 0.6681, 0.8434, 0.6541, 0.7577, 0.7198, 0.7264, 0.6272,
    0.6742, 0.8365, 0.6693, 0.6484, 0.6336, 0.5932, 0.6874, 0.6353, 0.786, 0.8596,
    0.5823, 0.5891, 0.7057, 0.6796, 0.6785, 0.7429, 0.6955, 0.589, 0.7011, 0.5688,
    0.6194, 0.7684, 0.8227, 0.6404, 0.6795, 0.799, 0.6922, 0.5898, 0.5569, 0.6509,
    0.5909, 0.646, 0.7615, 0.7199, 0.5729, 0.6753, 0.6189, 0.5495, 0.6599, 0.6658,
    0.6389, 0.6874, 0.5571, 0.6582, 0.8773, 0.6774, 0.7698, 0.7854, 0.6147, 0.6029,
    0.601, 0.5849, 0.5964, 0.5073, 0.665, 0.5905, 0.6543, 0.6016, 0.6652, 0.6313,
    0.555, 0.6285, 0.6918, 0.6687, 0.7426, 0.5656, 0.5869, 0.5454, 0.5791, 0.6571,
    0.5816, 0.5611, 0.6045, 0.5645, 0.5708, 0.6226, 0.5997, 0.68, 0.6114, 0.7095,
]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib")

out_dir = Path(__file__).resolve().parent.parent / "docs" / "training_logs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "iteration_001_loss.png"

steps = list(range(1, len(LOSS) + 1))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, LOSS, color="#2563eb", linewidth=1.2, alpha=0.9)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training loss — Iteration 001 (Qwen2.5-3B QLoRA)")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(LOSS) + 1)
fig.tight_layout()
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
