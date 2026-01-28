import os
import sys

# directory dello script, non del processo
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

T1 = 2000.0
T2 = 10000.0
T3 = 100000.0

def format_dhm(seconds):
    if seconds <= 0 or not (seconds < 1e12):
        return "0d 0h 0m"
    seconds = int(seconds)
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    return f"{d}d {h}h {m}m"

def estimate_remaining(t_last, c_last):
    if t_last <= 0:
        return None, None

    if t_last <= T1:
        target = T1
    elif t_last <= T2:
        target = T2
    elif t_last <= T3:
        target = T3
    else:
        return None, None  # oltre 100000 non stimiamo

    total_clock_est = c_last * (target / t_last)
    remaining_clock = max(total_clock_est - c_last, 0.0)
    return target, remaining_clock

for name in sorted(os.listdir(script_dir)):
    case_path = os.path.join(script_dir, name)
    if not (os.path.isdir(case_path) and name.startswith("case_")):
        continue

    time_file = os.path.join(case_path, "time.txt")
    clock_file = os.path.join(case_path, "clock_time.txt")

    if not os.path.isfile(time_file):
        print(f"{name}: time.txt NOT FOUND")
        continue

    if not os.path.isfile(clock_file):
        print(f"{name}: clock_time.txt NOT FOUND")
        continue

    try:
        with open(time_file, "r") as f:
            t_vals = [float(x) for x in f.read().split()]
        with open(clock_file, "r") as f:
            c_vals = [float(x) for x in f.read().split()]

        if not t_vals or not c_vals:
            print(f"{name}: EMPTY DATA")
            continue

        t_last = t_vals[-1]   # [s]
        c_last = c_vals[-1]   # [s]

        target, remaining = estimate_remaining(t_last, c_last)

        base_msg = (
            f"{name}: "
            f"t={t_last:.2f} s | "
            f"elapsed {format_dhm(c_last)}"
        )

        if target is None:
            print(base_msg + " | no further estimate")
        else:
            print(
                base_msg +
                f" | remaining to {int(target)} s: {format_dhm(remaining)}"
            )

    except Exception as e:
        print(f"{name}: ERROR ({e})")

input("Press ENTER to close...")
