import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from io import StringIO

def safe_loadtxt(filename, fill_value=-1e30):
    def parse_line(line):
        return (line.replace('-nan(ind)', str(fill_value))
                    .replace('nan', str(fill_value))
                    .replace('NaN', str(fill_value)))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    return np.loadtxt(StringIO(''.join(lines)))

# -------------------- Case discovery --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
root = script_dir
cases = []

for dirpath, dirnames, filenames in os.walk(root):
    base = os.path.basename(dirpath)
    parent = os.path.basename(os.path.dirname(dirpath))

    if base.startswith("case") or (parent.startswith("cases") and "case" in base):
        cases.append(dirpath)

cases = sorted(cases)

if len(cases) == 0:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(f"{i}: {os.path.relpath(c, root)}")

idx = int(input("Select case index: "))
case = cases[idx]

# -------------------- Files --------------------
time_file = os.path.join(case, "time.txt")

targets = [
    "continuity_res_x.txt",
    "continuity_res_v.txt",
    "momentum_res_x.txt",
    "momentum_res_v.txt",
    "temperature_res_x.txt",
    "temperature_res_v.txt",
]

y_files = [os.path.join(case, f) for f in targets]

if not os.path.isfile(time_file):
    print("Missing file:", time_file)
    sys.exit(1)

for f in y_files:
    if not os.path.isfile(f):
        print("Missing file:", f)
        sys.exit(1)

# -------------------- Load data --------------------
time = safe_loadtxt(time_file)
Y = [safe_loadtxt(f) for f in y_files]

names = [
    "Continuity\nresidual (wick)",
    "Continuity\nresidual (vapor)",
    "Momentum\nresidual (wick)",
    "Momentum\nresidual (vapor)",
    "Temperature\nresidual (wick)",
    "Temperature\nresidual (vapor)",
]


units = ["[-]"] * len(names)

# -------------------- Utils --------------------
def robust_ylim(y):
    lo, hi = np.percentile(y, [1, 99])
    if lo == hi:
        lo, hi = np.min(y), np.max(y)
    margin = 0.1 * (hi - lo) if hi > lo else 1.0
    return lo - margin, hi + margin

# -------------------- Figure --------------------
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.10, bottom=0.15, right=0.75)

line, = ax.plot([], [], lw=2, marker='o', markersize=4)
ax.grid(True)
ax.set_xlabel("Time [s]")

# -------------------- Buttons --------------------
buttons = []
button_width = 0.18
button_height = 0.08

panel_left = 0.78
panel_top = 0.80
row_gap = 0.10

for i, name in enumerate(names):
    y_pos = panel_top - i * row_gap
    b_ax = plt.axes([panel_left, y_pos, button_width, button_height])
    btn = Button(b_ax, name, hovercolor='0.975')
    buttons.append(btn)

current_idx = 0
ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(time.min(), time.max())
ax.set_ylim(*robust_ylim(Y[current_idx]))

# -------------------- Drawing --------------------
def draw():
    y = np.asarray(Y[current_idx])
    t = np.asarray(time)

    n = min(len(t), len(y))
    t = t[:n]
    y = y[:n]

    line.set_data(t, y)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(*robust_ylim(y))
    fig.canvas.draw_idle()


# -------------------- Variable change --------------------
def change_variable(idx):
    global current_idx
    current_idx = idx
    ax.set_title(f"{names[idx]} {units[idx]}")
    draw()

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# -------------------- Init --------------------
draw()
plt.show()
