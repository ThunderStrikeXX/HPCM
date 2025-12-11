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

# -------------------- Files --------------------
x_file = os.path.join(case, "mesh.txt")
time_file = os.path.join(case, "time.txt")

targets = [
    "vapor_velocity.txt",
    "vapor_pressure.txt",
    "vapor_bulk_temperature.txt",
    "rho_vapor.txt",

    "wick_velocity.txt",
    "wick_pressure.txt",
    "wick_bulk_temperature.txt",
    "rho_liquid.txt",

    "wick_vapor_interface_temperature.txt",
    "wall_wick_interface_temperature.txt",
    "outer_wall_temperature.txt",
    "wall_bulk_temperature.txt",

    "wick_vapor_mass_source.txt",

    "outer_wall_heat_source.txt",
    "wall_wx_heat_source.txt",
    "wick_wx_heat_source.txt",
    "wick_xv_heat_source.txt",
    "vapor_xv_heat_source.txt",

    "vapor_heat_source_mass.txt",
    "wick_heat_source_mass.txt",

    "saturation_pressure.txt",
    "sonic_velocity.txt"
]



y_files = [os.path.join(case, p) for p in targets]

for f in [x_file, time_file] + y_files:
    if not os.path.isfile(f):
        print("Missing file:", f)
        sys.exit(1)

# -------------------- Load data --------------------
x = safe_loadtxt(x_file)
time = safe_loadtxt(time_file) 
Y = [safe_loadtxt(f) for f in y_files]

names = [
    "Vapor velocity",
    "Vapor pressure",
    "Vapor bulk temperature",
    "Vapor density",

    "Wick velocity",
    "Wick pressure",
    "Wick bulk temperature",
    "Liquid density",

    "Wick-vapor interface temperature",
    "Wall-wick interface temperature",
    "Outer wall temperature",
    "Wall bulk temperature",

    "Wick-vapor mass source",

    "Outer wall heat-source (flux)",
    "Wall heat-source from wick (flux)",
    "Wick heat-source from wall (flux)",
    "Wick heat-source from vapor (flux)",
    "Vapor heat-source from wick (flux)",

    "Vapor heat-source (mass)",
    "Wick heat-source (mass)",

    "Saturation pressure",
    "Sonic speed"
]


units = [
    "[m/s]",
    "[Pa]",
    "[K]",
    "[kg/m³]",

    "[m/s]",
    "[Pa]",
    "[K]",
    "[kg/m³]",

    "[K]",
    "[K]",
    "[K]",
    "[K]",

    "[kg/(m³·s)]",

    "[W/m³]",
    "[W/m³]",
    "[W/m³]",
    "[W/m³]",
    "[W/m³]",

    "[W/m³]",
    "[W/m³]",

    "[Pa]",
    "[m/s]"
]


# -------------------- Utils --------------------
def robust_ylim(y):
    YMIN_CUTOFF = 50

    if y.ndim > 1:
        if y.shape[0] > YMIN_CUTOFF:
            vals = y[YMIN_CUTOFF:, :].flatten()
        else:
            vals = y.flatten()
    else:
        vals = y

    if vals.size == 0:
        return -1, 1

    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)

    margin = 0.1 * (hi - lo)
    return lo - margin, hi + margin

def time_to_index(t):
    return np.searchsorted(time, t, side='left')

def index_to_time(i):
    return time[i]

# -------------------- Figure --------------------
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.60)
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=1, linestyle='--')
ax.grid(True)
ax.set_xlabel("Axial length [m]")

# Slider con valori temporali reali
ax_slider = plt.axes([0.08, 0.10, 0.50, 0.03])
slider = Slider(ax_slider, "Time [s]", time.min(), time.max(), valinit=time[0])

# -------------------- Buttons list --------------------
buttons = []
n_vars = len(names)
n_cols = 3                # numero colonne
button_width = 0.11
button_height = 0.07
col_gap = 0.005   # pulsanti più vicini orizzontalmente

# area dedicata ai pulsanti
panel_left = 0.62
panel_right = 0.7
panel_top = 0.95
panel_bottom = 0.05

# calcolo righe totali necessarie
n_rows = int(np.ceil(n_vars / n_cols))

# altezza effettiva per ogni riga
row_height = (panel_top - panel_bottom) / (n_rows + 2.0)

for i, name in enumerate(names):
    col = i % n_cols
    row = i // n_cols

    x_pos = panel_left + col * (button_width + col_gap)
    # inverti asse y: riga 0 in alto
    y_pos = panel_top - (row + 1) * row_height

    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, "\n".join(textwrap.wrap(name, 12)), hovercolor='0.975')
    btn.label.set_fontsize(9)
    buttons.append(btn)

# Control buttons
ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
btn_play = Button(ax_play, "Play", hovercolor='0.975')
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
btn_pause = Button(ax_pause, "Pause", hovercolor='0.975')
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset", hovercolor='0.975')

current_idx = 0
ydata = Y[current_idx]
n_frames = len(time)

ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(*robust_ylim(ydata))

paused = [False]
current_frame = [0]

# -------------------- Drawing --------------------
def draw_frame(i, update_slider=True):
    y = Y[current_idx]

    # plot principale
    if y.ndim > 1:
        line.set_data(x, y[i, :])
    else:
        line.set_data(x, y)

    # sovrapposizione sonic velocity
    if names[current_idx] == "Vapor velocity":
        y_sonic = Y[names.index("Sonic speed")]
        if y_sonic.ndim > 1:
            line2.set_data(x, y_sonic[i, :])
        else:
            line2.set_data(x, y_sonic)
        line2.set_visible(True)
    else:
        line2.set_visible(False)

    if update_slider:
        slider.disconnect(slider_cid)
        slider.set_val(index_to_time(i))
        connect_slider()
    return line,

def update_auto(i):
    if not paused[0]:
        current_frame[0] = i
        draw_frame(i)
    return line,

def slider_update(val):
    i = time_to_index(val)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

# -------------------- Variable change --------------------
def change_variable(idx):
    global current_idx
    global ydata
    current_idx = idx
    ydata = Y[idx]

    ax.set_title(f"{names[idx]} {units[idx]}")
    ax.set_ylim(*robust_ylim(ydata))

    current_frame[0] = 0
    draw_frame(0)


for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# -------------------- Controls --------------------
def pause(event):
    paused[0] = True

def reset(event):
    paused[0] = True
    current_frame[0] = 0
    draw_frame(0)
    slider.set_val(time[0])
    fig.canvas.draw_idle()

def play(event):
    paused[0] = False

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

# -------------------- Animation --------------------
skip = max(1, n_frames // 200)
ani = FuncAnimation(
    fig,
    update_auto,
    frames=range(0, n_frames, skip),
    interval=10000 / (n_frames/skip),
    blit=False,
    repeat=True
)

plt.show()
