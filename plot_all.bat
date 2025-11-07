@echo off
REM Esegue lo script Python con tutti i file specificati
python plot_data.py ^
    mesh.txt ^
    results\vapor_velocity.txt ^
    results\vapor_bulk_temperature.txt ^
    results\vapor_pressure.txt ^
    results\wick_velocity.txt ^
    results\wick_bulk_temperature.txt ^
    results\wick_pressure.txt ^
    results\wall_bulk_temperature.txt ^
    results\outer_wall_temperature.txt ^
    results\wall_wick_interface_temperature.txt ^
    results\wick_vapor_interface_temperature.txt ^
    results\outer_wall_heat_flux.txt ^
    results\wall_wick_heat_flux.txt ^
    results\wick_vapor_heat_flux.txt ^
    results\wick_vapor_mass_source.txt ^
    results\rho_vapor.txt
pause
