import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from io import StringIO
import textwrap

# ============================================================
# Utils
# ============================================================
def safe_loadtxt(filename, fill_value=-1e9):
    def parse_line(line):
        return (line.replace('-nan(ind)', str(fill_value))
                    .replace('nan', str(fill_value))
                    .replace('NaN', str(fill_value)))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    return np.loadtxt(StringIO(''.join(lines)))

def robust_ylim(y_list):
    vals = []
    for y in y_list:
        if y.ndim > 1:
            if y.shape[0] > 50:
                vals.append(y[50:, :].ravel())
            else:
                vals.append(y.ravel())
        else:
            vals.append(y)
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)
    m = 0.1 * (hi - lo)
    return lo - m, hi + m

def time_to_index(time, t):
    return np.searchsorted(time, t, side="left")

# ============================================================
# Case selection
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
root = script_dir

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


if not cases:
    print("No case folders found")
    sys.exit(1)

print("Available cases:")
for i, c in enumerate(cases):
    print(f"{i}: {os.path.relpath(c, root)}")

sel = input("Select case indices (comma separated): ")
idxs = [int(i.strip()) for i in sel.split(",")]
cases_sel = [cases[i] for i in idxs]
case_labels = [
    os.path.basename(c).split("case_", 1)[1]
    for c in cases_sel
]


# ============================================================
# Files and metadata
# ============================================================
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
    "sonic_velocity.txt",
    "reynolds_vapor.txt",
    "HTC.txt" 
]

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
    "Sonic speed",
    "Vapor Reynolds number",
    "Vapor HTC"
]

units = [
    "[m/s]", "[Pa]", "[K]", "[kg/m³]",
    "[m/s]", "[Pa]", "[K]", "[kg/m³]",
    "[K]", "[K]", "[K]", "[K]",
    "[kg/(m³·s)]",
    "[W/m³]", "[W/m³]", "[W/m³]", "[W/m³]", "[W/m³]",
    "[W/m³]", "[W/m³]",
    "[Pa]", "[m/s]", "[-]", "[W/(m2·K)]"
]

# ============================================================
# Load data
# ============================================================
X = []
TIME = None
Y = []   # Y[var][case]

for c in cases_sel:
    x = safe_loadtxt(os.path.join(c, "mesh.txt"))
    t = safe_loadtxt(os.path.join(c, "time.txt"))

    X.append(x)

    if TIME is None:
        TIME = t

    y_case = [safe_loadtxt(os.path.join(c, f)) for f in targets]
    Y.append(y_case)

# transpose -> Y[var][case]
Y = list(map(list, zip(*Y)))

n_cases = len(cases_sel)
n_frames = len(TIME)

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.60)

lines = []
for i in range(n_cases):
    (ln,) = ax.plot([], [], lw=2, marker='o', markersize=3,
                    label=case_labels[i])
    lines.append(ln)

ax.grid(True)
ax.set_xlabel("Axial length [m]")
ax.legend()

# Slider
ax_slider = plt.axes([0.08, 0.10, 0.50, 0.03])
slider = Slider(ax_slider, "Time [s]", TIME.min(), TIME.max(), valinit=TIME[0])

# Timebox
ax_timebox = plt.axes([0.70, 0.09, 0.15, 0.05])
time_box = TextBox(ax_timebox, "Set t [s] ", initial=f"{TIME[0]:.6g}")

# ============================================================
# Variable buttons
# ============================================================
buttons = []
n_vars = len(names)
n_cols = 3
button_width = 0.11
button_height = 0.07
col_gap = 0.005

panel_left = 0.62
panel_top = 0.95
panel_bottom = 0.05
n_rows = int(np.ceil(n_vars / n_cols))
row_height = (panel_top - panel_bottom) / (n_rows + 2.0)

for i, name in enumerate(names):
    col = i % n_cols
    row = i // n_cols
    x_pos = panel_left + col * (button_width + col_gap)
    y_pos = panel_top - (row + 1) * row_height
    bax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(bax, "\n".join(textwrap.wrap(name, 12)))
    btn.label.set_fontsize(9)
    buttons.append(btn)

# ============================================================
# State
# ============================================================
current_var = 0
current_frame = [0]
running = [False]
updating = [False]

ax.set_title(f"{names[current_var]} {units[current_var]}")
xmin = min(x.min() for x in X)
xmax = max(x.max() for x in X)
ax.set_xlim(xmin, xmax)
ax.set_ylim(*robust_ylim(Y[current_var]))

# ============================================================
# Drawing
# ============================================================
def draw_frame(i, update_slider=True):

    for c in range(n_cases):
        y = Y[current_var][c]
        if y.ndim > 1:
            ii = min(i, y.shape[0] - 1)
            lines[c].set_data(X[c], y[ii, :])
        else:
            lines[c].set_data(X[c], y)

    if update_slider:
        slider.disconnect(slider_cid)

        if len(TIME) > 0:
            slider.set_val(TIME[min(i, len(TIME) - 1)])
        connect_slider()

    return lines

def update_auto(i):
    current_frame[0] = i
    draw_frame(i)
    return lines

def slider_update(val):
    if updating[0]:
        return
    updating[0] = True
    i = time_to_index(TIME, val)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    time_box.set_val(f"{val:.6g}")
    updating[0] = False
    fig.canvas.draw_idle()

def submit_time(text):
    if updating[0]:
        return
    try:
        t = float(text)
    except ValueError:
        return

    t = max(TIME.min(), min(t, TIME.max()))
    updating[0] = True
    i = time_to_index(TIME, t)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    slider.set_val(t)
    updating[0] = False
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

time_box.on_submit(submit_time)

# ============================================================
# Variable change
# ============================================================
def change_variable(idx):
    global current_var
    current_var = idx
    ax.set_title(f"{names[idx]} {units[idx]}")


    ax.set_ylim(*robust_ylim(Y[idx]))
    ax.legend(handles=lines, loc='best')

    current_frame[0] = 0
    draw_frame(0)

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# ============================================================
# Controls
# ============================================================
ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
ax_save = plt.axes([0.51, 0.02, 0.10, 0.05])

btn_play = Button(ax_play, "Play")
btn_pause = Button(ax_pause, "Pause")
btn_reset = Button(ax_reset, "Reset")
btn_save = Button(ax_save, "Save")

def play(event):
    ani.event_source.start()
    running[0] = True

def pause(event):
    ani.event_source.stop()
    running[0] = False

def reset(event):
    ani.event_source.stop()
    running[0] = False
    current_frame[0] = 0
    draw_frame(0)
    slider.set_val(TIME[0])
    time_box.set_val(f"{TIME[0]:.6g}")
    fig.canvas.draw_idle()

def save_plot(event):
    fig.canvas.draw()

    # bounding box dell'axes
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    filename = os.path.join(
        desktop,
        f"{names[current_var].replace(' ', '_')}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches=bbox_inches,
        pad_inches=0.02
    )

    print(f"Saved on Desktop: {filename}")

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)
btn_save.on_clicked(save_plot)

# ============================================================
# Animation
# ============================================================
skip = max(1, n_frames // 200)

ani = FuncAnimation(
    fig,
    update_auto,
    frames=range(0, n_frames, skip),
    interval=10000 / (n_frames / skip),
    blit=False,
    repeat=True
)

# force start in paused state
ani.event_source.stop()
running[0] = False
current_frame[0] = 0
draw_frame(0)

change_variable(current_var)

plt.show()
