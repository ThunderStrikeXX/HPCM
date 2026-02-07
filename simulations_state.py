import os
import sys

# directory dello script, non del processo
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

T1 = 2000.0
T2 = 5000.0
T3 = 20000.0


def format_dhm(seconds):
    if seconds <= 0 or not (seconds < 1e12):
        return "0d0h0m"
    seconds = int(seconds)
    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    return f"{d:2d}d{h:02d}h{m:02d}m"


def estimate_remaining(t_last, c_last):
    if t_last <= 0 or c_last <= 0:
        return None, None

    if t_last <= T1:
        target = T1
    elif t_last <= T2:
        target = T2
    elif t_last <= T3:
        target = T3
    else:
        return None, None

    total_clock_est = c_last * (target / t_last)
    remaining_clock = max(total_clock_est - c_last, 0.0)
    return target, remaining_clock


def scan_cases():
    rows = []

    for name in sorted(os.listdir(script_dir)):
        case_path = os.path.join(script_dir, name)
        if not (os.path.isdir(case_path) and name.startswith("case_")):
            continue

        time_file = os.path.join(case_path, "time.txt")
        clock_file = os.path.join(case_path, "clock_time.txt")

        if not os.path.isfile(time_file) or not os.path.isfile(clock_file):
            rows.append((name, "missing", "", "", ""))
            continue

        try:
            with open(time_file) as f:
                t_vals = [float(x) for x in f.read().split()]
            with open(clock_file) as f:
                c_vals = [float(x) for x in f.read().split()]

            if not t_vals or not c_vals:
                rows.append((name, "empty", "", "", ""))
                continue

            t_last = t_vals[-1]
            c_last = c_vals[-1]

            speedup = t_last / c_last if c_last > 0 else 0.0
            target, remaining = estimate_remaining(t_last, c_last)

            t_str = f"{t_last:7.0f}s"
            clk_str = f"{format_dhm(c_last)}"
            spd_str = f"{speedup:5.2f}x"

            if target is None:
                rem_str = "---"
            else:
                rem_str = f"{int(target):5d}s {format_dhm(remaining)}"

            rows.append((name, t_str, clk_str, spd_str, rem_str))

        except Exception as e:
            rows.append((name, "ERROR", str(e), "", ""))

    # larghezze colonne (calcolate automaticamente)
    w_name = max(len(r[0]) for r in rows)
    w_t    = max(len(r[1]) for r in rows)
    w_clk  = max(len(r[2]) for r in rows)
    w_spd  = max(len(r[3]) for r in rows)
    w_rem  = max(len(r[4]) for r in rows)

    lines = []
    for r in rows:
        lines.append(
            f"{r[0]:<{w_name}} | "
            f"{r[1]:>{w_t}} | "
            f"{r[2]:<{w_clk}} | "
            f"{r[3]:>{w_spd}} | "
            f"{r[4]:<{w_rem}}"
        )

    return lines


while True:
    os.system("cls" if os.name == "nt" else "clear")

    for line in scan_cases():
        print(line)

    try:
        input("\nENTER = refresh | CTRL+C = exit")
    except KeyboardInterrupt:
        break
