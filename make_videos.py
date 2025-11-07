import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# ==============================================================
# Configuration
# ==============================================================
x_file = "mesh.txt"
data_dir = "results"
video_dir = "videos"
duration_s = 10          # total video duration [s]
max_frames = 200         # limit to this many frames per video
dpi = 100                # lower = faster
os.makedirs(video_dir, exist_ok=True)

# ==============================================================
# Load X data
# ==============================================================
x = np.loadtxt(x_file)

# ==============================================================
# Collect data files
# ==============================================================
y_files = [f for f in os.listdir(data_dir)
           if f.endswith(".txt") and f.lower() != "mesh.txt"]

# ==============================================================
# Process each file
# ==============================================================
for fname in y_files:
    path = os.path.join(data_dir, fname)
    print(f"Processing {fname}...")

    y = np.loadtxt(path)
    if y.ndim == 1:
        print(f"  Skipped: single line file")
        continue

    n_frames_total, n_points = y.shape
    # Downsample frames uniformly if too many
    if n_frames_total > max_frames:
        step = int(np.ceil(n_frames_total / max_frames))
        y = y[::step, :]
        n_frames = y.shape[0]
    else:
        n_frames = n_frames_total

    fps = n_frames / duration_s

    # Temporary folder for PNGs
    tmp_dir = os.path.join(video_dir, "_tmp_" + os.path.splitext(fname)[0])
    os.makedirs(tmp_dir, exist_ok=True)

    # --- Generate reduced set of frames ---
    for i in range(n_frames):
        plt.figure(figsize=(7, 4))
        plt.plot(x, y[i, :], lw=2)
        plt.xlim(x.min(), x.max())
        plt.ylim(np.min(y), np.max(y))
        plt.title(f"{fname} (Frame {i+1}/{n_frames})")
        plt.xlabel("Axial length [m]")
        plt.ylabel("Value")
        plt.grid(True)
        out_png = os.path.join(tmp_dir, f"frame_{i:05d}.png")
        plt.savefig(out_png, dpi=dpi)
        plt.close()

    # --- Assemble video with ffmpeg ---
    out_mp4 = os.path.join(video_dir, os.path.splitext(fname)[0] + ".mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.2f}",
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_mp4
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # --- Cleanup temporary images ---
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print(f"  Saved -> {out_mp4}")

print("All videos created successfully.")
