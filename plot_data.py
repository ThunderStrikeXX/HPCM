import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import textwrap, sys, os

def safe_loadtxt(filename, fill_value=-1e9):
    def parse_line(line):
        # sostituisce '-nan(ind)' o 'nan' o 'NaN' con fill_value
        return line.replace('-nan(ind)', str(fill_value)) \
                   .replace('nan', str(fill_value)) \
                   .replace('NaN', str(fill_value))
    with open(filename, 'r') as f:
        lines = [parse_line(l) for l in f]
    from io import StringIO
    return np.loadtxt(StringIO(''.join(lines)))

# --- Input check ---
if len(sys.argv) < 3:
    print("Usage: python plot_data.py x.txt y1.txt y2.txt ...")
    sys.exit(1)

x_file = sys.argv[1]
y_files = sys.argv[2:]

for f in [x_file] + y_files:
    if not os.path.isfile(f):
        print(f"Error: file not found -> {f}")
        sys.exit(1)

# --- Data ---
x = safe_loadtxt(x_file)
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

# --- Adjust y axis ---
def robust_ylim(y):
    vals = y.flatten() if y.ndim > 1 else y
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = np.min(vals), np.max(vals)
    margin = 0.1 * (hi - lo)
    return lo - margin, hi + margin

# --- Figure principale ---
fig, ax = plt.subplots(figsize=(11, 6))  # più larga
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.70)  # lascia spazio per 2 colonne
line, = ax.plot([], [], lw=2)
ax.grid(True)
ax.set_xlabel("Axial length [m]")

# --- Slider ---
ax_slider = plt.axes([0.15, 0.1, 0.55, 0.03])
slider = Slider(ax_slider, "Time step", 0, 1, valinit=0, valstep=1)

# --- Pulsanti variabili in due colonne visibili ---
buttons = []
n_vars = len(Y)
n_cols = 2
n_rows = int(np.ceil(n_vars / n_cols))

button_width = 0.12
button_height = 0.08
col_gap = 0.02
row_gap = 0.10
start_x = 0.73   # spostato a destra del grafico
start_y = 0.86

for i, name in enumerate(names[:n_vars]):
    col = i % n_cols
    row = i // n_cols
    label = "\n".join(textwrap.wrap(name, 15))
    x_pos = start_x + col * (button_width + col_gap)
    y_pos = start_y - row * row_gap
    b_ax = plt.axes([x_pos, y_pos, button_width, button_height])
    btn = Button(b_ax, label, hovercolor='0.975')
    btn.label.set_fontsize(8)
    buttons.append(btn)

# --- Pulsanti controllo ---
ax_play = plt.axes([0.15, 0.02, 0.10, 0.05])
btn_play = Button(ax_play, "Play", hovercolor='0.975')
ax_pause = plt.axes([0.27, 0.02, 0.10, 0.05])
btn_pause = Button(ax_pause, "Pause", hovercolor='0.975')
ax_reset = plt.axes([0.39, 0.02, 0.10, 0.05])
btn_reset = Button(ax_reset, "Reset", hovercolor='0.975')

# --- Stato iniziale ---
current_idx = 0
ydata = Y[current_idx]
n_frames = ydata.shape[0]
duration_ms = 10_000
interval = duration_ms / n_frames

ax.set_title(f"{names[current_idx]} {units[current_idx]}")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(*robust_ylim(ydata))
slider.valmax = n_frames - 1
slider.ax.set_xlim(slider.valmin, slider.valmax)

paused = [False]
current_frame = [0]

# --- Funzioni ---
def draw_frame(i, update_slider=True):
    y = Y[current_idx]
    if y.ndim > 1:
        line.set_data(x, y[i, :])
    else:
        line.set_data(x, y)
    if update_slider:
        slider.disconnect(slider_cid)
        slider.set_val(i)
        connect_slider()
    return line,

def update_auto(i):
    if not paused[0]:
        current_frame[0] = i
        draw_frame(i, update_slider=True)
    return line,

def slider_update(val):
    i = int(slider.val)
    current_frame[0] = i
    draw_frame(i, update_slider=False)
    fig.canvas.draw_idle()

def connect_slider():
    global slider_cid
    slider_cid = slider.on_changed(slider_update)

connect_slider()

def change_variable(idx):
    global current_idx, ydata, n_frames, interval
    current_idx = idx
    ydata = Y[idx]
    n_frames = ydata.shape[0]
    interval = 10_000 / n_frames
    ax.set_title(f"{names[idx]} {units[idx]}")
    ax.set_ylim(*robust_ylim(ydata))
    slider.valmax = n_frames - 1
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    current_frame[0] = 0
    draw_frame(0)
    ani.event_source.interval = interval

for i, btn in enumerate(buttons):
    btn.on_clicked(lambda event, j=i: change_variable(j))

def pause(event):
    paused[0] = True

def reset(event):
    paused[0] = True
    current_frame[0] = 0
    draw_frame(0)
    slider.set_val(0)
    ani.frame_seq = ani.new_frame_seq()  # forza restart dall’inizio
    fig.canvas.draw_idle()

def play(event):
    ani.frame_seq = ani.new_frame_seq()  # se era stato resettato, ricomincia
    paused[0] = False

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

# --- Animazione ---
skip = max(1, n_frames // 200)  # ~300 frame total
ani = FuncAnimation(fig, update_auto,
                    frames=range(0, n_frames, skip),
                    interval=duration_ms / (n_frames / skip),
                    blit=False, repeat=True)
plt.show()
