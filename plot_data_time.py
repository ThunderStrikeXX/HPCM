import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from io import StringIO
import textwrap

def safe_loadtxt(filename, fill_value=-1e9):
    def parse_line(line):
        return (line.replace('-nan(ind)', str(fill_value))
                    .replace('nan', str(fill_value))
                    .replace('NaN', str(fill_value)))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    return np.loadtxt(StringIO(''.join(lines)))

root = os.getcwd()
cases = [d for d in os.listdir(root) if os.path.isdir(d) and "case" in d]

if len(cases) == 0:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(i, c)

idx = int(input("Select case index: "))
case = cases[idx]

x_file = os.path.join(case, "mesh.txt")
time_file = os.path.join(case, "time.txt")

targets = [
    "vapor_velocity.txt",
    "vapor_bulk_temperature.txt",
    "vapor_pressure.txt",
    "wick_velocity.txt",
    "wick_bulk_temperature.txt",
    "wick_pressure.txt",
    "wall_bulk_temperature.txt",
    "outer_wall_temperature.txt",
    "wall_wick_interface_temperature.txt",
    "wick_vapor_interface_temperature.txt",
    "outer_wall_heat_flux.txt",
    "wall_wick_heat_flux.txt",
    "wick_vapor_heat_flux.txt",
    "wick_vapor_mass_source.txt",
    "rho_vapor.txt"
]

y_files = [os.path.join(case, p) for p in targets]

for f in [x_file, time_file] + y_files:
    if not os.path.isfile(f):
        print("Missing file:", f)
        sys.exit(1)

x = safe_loadtxt(x_file)
time = safe_loadtxt(time_file)
Y = [safe_loadtxt(f) for f in y_files]

names = [
    "Vapor velocity", "Vapor bulk temperature", "Vapor pressure",
    "Wick velocity", "Wick bulk temperature", "Wick pressure",
    "Wall bulk temperature", "Outer wall temperature",
    "Wall-wick interface temperature", "Wick-vapor interface temperature",
    "Outer wall heat flux", "Wall-wick heat flux", "Wick-vapor heat flux",
    "Mass volumetric source", "Vapor density"
]

units = [
    "[m/s]", "[K]", "[Pa]", "[m/s]", "[K]", "[Pa]",
    "[K]", "[K]", "[K]", "[K]",
    "[W/m²]", "[W/m²]", "[W/m²]", "[kg/s m³]", "[kg/m³]"
]

def robust_ylim(y):
    vals = y.flatten() if y.ndim > 1 else y
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)
    margin = 0.1 * (hi - lo)
    return lo - margin, hi + margin

def pos_to_index(val):
    return np.searchsorted(x, val, side='left')

def index_to_pos(i):
    return x[i]

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.70)
line, = ax.plot([], [], lw=2)
ax.grid(True)
ax.set_xlabel("Time [s]")

ax_slider = plt.axes([0.15, 0.1, 0.55, 0.03])
slider = Slider(ax_slider, "Axial pos [m]", x.min(), x.max(), valinit=x[0])

buttons = []
n_vars = len(Y)
n_cols = 2
button_width = 0.12
button_height = 0.08
col_gap = 0.02
row_gap = 0.10
start_x = 0.73
start_y = 0.86

for i, name in enumerate(names):
    col = i % n_cols
    row = i // n_cols
    label = "\n".join(textwrap.wrap(name, 15))
    x_pos = start_x + col * (button_width + col_gap)
    y_pos = start_y - row * row_gap
    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, label, hovercolor='0.975')
    btn.label.set_fontsize(8)
    buttons.append(btn)

ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
btn_play = Button(ax_play, "Play", hovercolor='0.975')
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
btn_pause = Button(ax_pause, "Pause", hovercolor='0.975')
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset", hovercolor='0.975')

current_idx = 0
ydata = Y[current_idx]

n_nodes = len(x)
n_frames = len(time)

ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(time.min(), time.max())

paused = [False]
current_node = [0]

def draw_node(i, update_slider=True):
    y = Y[current_idx]
    line.set_data(time, y[:, i])
    ax.set_ylim(*robust_ylim(y[:, i]))
    if update_slider:
        slider.disconnect(slider_cid)
        slider.set_val(index_to_pos(i))
        connect_slider()
    return line,

def update_auto(i):
    if not paused[0]:
        current_node[0] = i
        draw_node(i)
    return line,

def slider_update(val):
    i = pos_to_index(val)
    current_node[0] = i
    draw_node(i, update_slider=False)
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

def change_variable(idx):
    global current_idx, ydata
    current_idx = idx
    ydata = Y[idx]
    ax.set_title(f"{names[idx]} {units[idx]}")
    draw_node(current_node[0])

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

def pause(event):
    paused[0] = True

def reset(event):
    paused[0] = True
    current_node[0] = 0
    draw_node(0)
    slider.set_val(x[0])
    fig.canvas.draw_idle()

def play(event):
    paused[0] = False

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

skip = max(1, n_nodes // 200)
ani = FuncAnimation(
    fig,
    update_auto,
    frames=range(0, n_nodes, skip),
    interval=10000 / (n_nodes/skip),
    blit=False,
    repeat=True
)

plt.show()
