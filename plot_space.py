import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
from io import StringIO
import textwrap
from matplotlib.transforms import Bbox

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

def wrap_max_2_lines(text, start_width=20, max_width=60):
    """
    Wrappa il testo in modo che occupi al massimo 2 righe.
    Aumenta la larghezza finché ci riesce.
    """
    for w in range(start_width, max_width + 1):
        lines = textwrap.wrap(text, width=w)
        if len(lines) <= 2:
            return "\n".join(lines)

    # fallback: comunque solo 2 righe
    return "\n".join(textwrap.wrap(text, width=max_width)[:2])

def wrap_max_3_lines_full_text(text, start_width=10, max_width=35):
    """
    Trova una larghezza di wrap tale che tutto il testo
    stia in massimo 3 righe.
    """
    for w in range(start_width, max_width + 1):
        lines = textwrap.wrap(text, width=w)
        if len(lines) <= 3:
            return "\n".join(lines)
    # fallback: ultima possibilità (testo molto lungo)
    return "\n".join(textwrap.wrap(text, width=max_width))

def fit_text_in_button(button, min_fontsize=5, max_fontsize=9):
    fig = button.ax.figure
    label = button.label

    # Forza un primo draw per avere bbox valide
    fig.canvas.draw()

    ax_bbox = button.ax.get_window_extent()

    for fs in range(max_fontsize, min_fontsize - 1, -1):
        label.set_fontsize(fs)
        fig.canvas.draw()

        text_bbox = label.get_window_extent()

        if (text_bbox.width <= ax_bbox.width and
            text_bbox.height <= ax_bbox.height):
            return  # trovato font valido

    # fallback estremo
    label.set_fontsize(min_fontsize)


# ============================================================
# Case selection
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
root = script_dir

cases = []

for d in os.listdir(root):
    full = os.path.join(root, d)
    if os.path.isdir(full) and d.startswith("case_"):
        cases.append(full)

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

sel = input("Select case indices (comma separated): ")
idxs = [int(i.strip()) for i in sel.split(",")]
cases_sel = [cases[i] for i in idxs]
case_labels = [os.path.basename(c).split("case_", 1)[1] for c in cases_sel]

# ============================================================
# Files and metadata
# ============================================================
targets = [
    "vapor_velocity.txt",
    "vapor_pressure.txt",
    "vapor_bulk_temperature.txt",
    "rho_vapor.txt",
    "liquid_velocity.txt",
    "liquid_pressure.txt",
    "liquid_bulk_temperature.txt",
    "rho_liquid.txt",
    "liquid_vapor_interface_temperature.txt",
    "wall_liquid_interface_temperature.txt",
    "outer_wall_temperature.txt",
    "wall_bulk_temperature.txt",
    "liquid_vapor_mass_source.txt",
    "outer_wall_heat_source.txt",
    "wall_wx_heat_source.txt",
    "liquid_wx_heat_source.txt",
    "liquid_xv_heat_source.txt",
    "vapor_xv_heat_source.txt",
    "vapor_heat_source_mass.txt",
    "liquid_heat_source_mass.txt",
    "saturation_pressure.txt",
    "sonic_velocity.txt",
    "reynolds_vapor.txt",
    "HTC.txt"
]

names = [
    "Vapor velocity",
    "Vapor absolute pressure",
    "Vapor temperature",
    "Vapor density",
    "Liquid velocity",
    "Liquid gauge pressure",
    "Liquid bulk temperature",
    "Liquid density",
    "Liquid-vapor interface temperature",
    "Wall-liquid interface temperature",
    "Outer wall temperature",
    "Wall bulk temperature",
    "Mass flow rate from the liquid to the vapor",
    "Power from outside to the wall",
    "Power from the liquid to the wall due to conduction",
    "Power from the wall to the liquid due to conduction",
    "Power from the vapor to the liquid due to conduction",
    "Power from the liquid to the vapor due to convection",
    "Power from the liquid to the vapor due to phase change",
    "Power from the vapor to the liquid due to phase change",
    "Saturation pressure",
    "Sonic speed",
    "Vapor Reynolds number",
    "Vapor HTC"
]

units = [
    "[m/s]", "[Pa]", "[K]", "[kg/m³]",
    "[m/s]", "[Pa]", "[K]", "[kg/m³]",
    "[K]", "[K]", "[K]", "[K]",
    "[kg/s]",
    "[W]", "[W]", "[W]", "[W]", "[W]",
    "[W]", "[W]",
    "[Pa]", "[m/s]", "[-]", "[W/(m²·K)]"
]

# ============================================================
# Load data
# ============================================================
X = []
TIME = None
Y = []

for c in cases_sel:
    x = safe_loadtxt(os.path.join(c, "mesh.txt"))
    t = safe_loadtxt(os.path.join(c, "time.txt"))

    X.append(x)
    if TIME is None:
        TIME = t

    Y.append([safe_loadtxt(os.path.join(c, f)) for f in targets])

Y = list(map(list, zip(*Y)))

n_cases = len(cases_sel)
n_frames = len(TIME)

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.12, bottom=0.25, right=0.60, top=0.92)

y_label_top = fig.text(
    0.36, 0.985,
    "",
    ha="center", va="top",
    fontsize=11
)


lines = []

styles = [
    dict(lw=2.5, ls='-',  color='C0'),                # linea continua
    dict(lw=1.8, ls='--', color='C1'),                # linea tratteggiata
    dict(ls='None', marker='x', ms=6, color='C2'),    # marker
    dict(ls='None', marker='*', ms=8, color='C3'),    # marker
    dict(ls='None', marker='^', ms=6, color='C4'),    # marker
]

for i in range(n_cases):
    (ln,) = ax.plot(
        [],
        [],
        label=case_labels[i],
        **styles[i]
    )
    lines.append(ln)

ax.grid(True)
ax.set_xlabel("Axial length [m]")
ax.tick_params(axis='both', labelsize=9)

# Slider
ax_slider = plt.axes([0.12, 0.10, 0.46, 0.03])
slider = Slider(ax_slider, "Time [s]", TIME.min(), TIME.max(), valinit=TIME[0])

# Timebox
ax_timebox = plt.axes([0.70, 0.09, 0.15, 0.05])
time_box = TextBox(ax_timebox, "Set t [s]", initial=f"{TIME[0]:.6g}")

# ============================================================
# Variable buttons (FIXED: wrapping + font)
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
    wrapped = wrap_max_3_lines_full_text(name, start_width=10, max_width=30)

    btn = Button(bax, wrapped)
    btn.label.set_multialignment("center")

    fit_text_in_button(btn, min_fontsize=5, max_fontsize=9)

    buttons.append(btn)

# ============================================================
# State
# ============================================================
current_var = 0
current_frame = [0]
running = [False]
updating = [False]

ax.set_xlim(min(x.min() for x in X), max(x.max() for x in X))
ax.set_ylim(*robust_ylim(Y[current_var]))

y_label_top.set_text(
    wrap_max_2_lines(f"{names[current_var]} {units[current_var]}")
)


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
# Variable change (FIXED y-label)
# ============================================================
def change_variable(idx):
    global current_var
    current_var = idx
    y_label_top.set_text(
        wrap_max_2_lines(f"{names[idx]} {units[idx]}")
    )

    ax.set_ylim(*robust_ylim(Y[idx]))
    ax.legend(handles=lines, loc='best')
    current_frame[0] = 0
    draw_frame(0)

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

# ============================================================
# Controls
# ============================================================
ax_play  = plt.axes([0.15, 0.02, 0.10, 0.05])
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
ax_save  = plt.axes([0.51, 0.02, 0.10, 0.05])

btn_play  = Button(ax_play, "Play")
btn_pause = Button(ax_pause, "Pause")
btn_reset = Button(ax_reset, "Reset")
btn_save  = Button(ax_save, "Save")

def play(event):
    ani.event_source.start()

def pause(event):
    ani.event_source.stop()

def reset(event):
    ani.event_source.stop()
    current_frame[0] = 0
    draw_frame(0)
    slider.set_val(TIME[0])
    time_box.set_val(f"{TIME[0]:.6g}")
    fig.canvas.draw_idle()

def save_plot(event):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # bounding box dell'axes
    bbox_ax = ax.get_tightbbox(renderer)

    # bounding box della y-label (fig.text)
    bbox_label = y_label_top.get_window_extent(renderer)

    # UNIONE CORRETTA dei bounding box
    bbox = Bbox.union([bbox_ax, bbox_label])

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

ani.event_source.stop()
draw_frame(0)
change_variable(current_var)

plt.show()
