import os
import sys

# directory dello script, non del processo
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

for name in sorted(os.listdir(script_dir)):
    case_path = os.path.join(script_dir, name)
    if not (os.path.isdir(case_path) and name.startswith("case_")):
        continue

    time_file = os.path.join(case_path, "time.txt")

    if not os.path.isfile(time_file):
        print(f"{name}: time.txt NOT FOUND")
        continue

    try:
        with open(time_file, "r") as f:
            data = f.read().split()
            if data:
                print(f"{name}: {data[-1]}")
            else:
                print(f"{name}: EMPTY")
    except Exception as e:
        print(f"{name}: ERROR ({e})")

print("\n--- END ---")
input("Press ENTER to close...")
