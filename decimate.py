import os

files = [
    "mesh.txt",
    r"results\vapor_velocity.txt",
    r"results\vapor_bulk_temperature.txt",
    r"results\vapor_pressure.txt",
    r"results\wick_velocity.txt",
    r"results\wick_bulk_temperature.txt",
    r"results\wick_pressure.txt",
    r"results\wall_bulk_temperature.txt",
    r"results\outer_wall_temperature.txt",
    r"results\wall_wick_interface_temperature.txt",
    r"results\wick_vapor_interface_temperature.txt",
    r"results\outer_wall_heat_flux.txt",
    r"results\wall_wick_heat_flux.txt",
    r"results\wick_vapor_heat_flux.txt",
    r"results\wick_vapor_mass_source.txt",
    r"results\rho_vapor.txt",
]

def decimate(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    kept = [line for i, line in enumerate(lines) if i % 10 == 0]

    with open(file_path, "w") as f:
        f.writelines(kept)

for f in files:
    if os.path.exists(f):
        decimate(f)
