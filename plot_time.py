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
cases = []

# case_* nella cartella corrente
for d in os.listdir(root):
    full = os.path.join(root, d)
    if os.path.isdir(full) and d.startswith("case_"):
        cases.append(full)

# case_* dentro sottocartelle cases_*
for d in os.listdir(root):
    parent = os.path.join(root, d)
    if os.path.isdir(parent) and d.startswith("cases_"):
        for sub in os.listdir(parent):
            full = os.path.join(parent, sub)
            if os.path.isdir(full) and sub.startswith("case_"):
                cases.append(full)

cases = sorted(cases)

if not cases:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(f"{i}: {os.path.relpath(c, root)}")

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

    # --- HEAT SOURCES (NUOVI) ---
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
nT = len(time)

Y = [safe_loadtxt(f) for f in y_files]


Y_clean = []
for arr in Y:
    if arr.shape[0] > nT:
        arr = arr[:nT, :]          # taglia righe in eccesso
    elif arr.shape[0] < nT:
        pad = np.full((nT - arr.shape[0], arr.shape[1]), np.nan)
        arr = np.vstack([arr, pad])  # padding se troppo corto
    Y_clean.append(arr)

Y = Y_clean


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

    "Outer wall heat-source",
    "Wall-WX heat-source",
    "Wick-WX heat-source",
    "Wick-XV heat-source",
    "Vapor-XV heat-source",

    "Vapor heat-source mass",
    "Wick heat-source mass",

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

    "[kg/(m³·s)]",
    "[kg/(m³·s)]",

    "[Pa]",
    "[m/s]"
]

# -------------------- Utils --------------------
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

# -------------------- Figure --------------------
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.60)
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=1, linestyle='--')
ax.grid(True)
ax.set_xlabel("Time [s]")

# Slider posizione assiale
ax_slider = plt.axes([0.13, 0.10, 0.42, 0.03])
slider = Slider(ax_slider, "Axial pos [m]", x.min(), x.max(), valinit=x[0])

# -------------------- Buttons (layout come versione buona) --------------------
buttons = []
n_vars = len(names)
n_cols = 3                # numero colonne
button_width = 0.11
button_height = 0.07
col_gap = 0.005           # pulsanti più vicini orizzontalmente

panel_left = 0.62
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
    # riga 0 in alto
    y_pos = panel_top - (row + 1) * row_height

    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, "\n".join(textwrap.wrap(name, 12)), hovercolor='0.975')
    btn.label.set_fontsize(9)
    buttons.append(btn)

# -------------------- Control buttons --------------------
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

# -------------------- Drawing --------------------
def draw_node(i, update_slider=True):
    y = Y[current_idx]

    # curva principale: y(t, x_i)
    line.set_data(time, y[:, i])

    if names[current_idx] == "Vapor velocity":
        y_sonic = Y[names.index("Sonic speed")]
        line2.set_data(time, y_sonic[:, i])
        line2.set_visible(True)
    else:
        # dati coerenti per evitare errori nel draw
        line2.set_data(time, np.full_like(time, np.nan))
        line2.set_visible(False)

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

# -------------------- Variable change --------------------
def change_variable(idx):
    global current_idx, ydata
    current_idx = idx
    ydata = Y[idx]
    ax.set_title(f"{names[idx]} {units[idx]}")
    draw_node(current_node[0])

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# -------------------- Controls --------------------
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

# -------------------- Animation --------------------
skip = max(1, n_nodes // 200)
ani = FuncAnimation(
    fig,
    update_auto,
    frames=range(0, n_nodes, skip),
    interval=10000 / (n_nodes / skip),
    blit=False,
    repeat=True
)

plt.show()
