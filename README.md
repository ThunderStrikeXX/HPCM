# Heat Pipe Coupled Multyphysics solver

This repository contains a 1D finite-volume solver for coupled **wall–wick–vapor** regions of a sodium heat pipe.  
The code is implemented in **C++17**, following the structure of **THROHPUT**, **HPTAM** and **Sockeye** frameworks.

---

## Features

- Transient simulation  
- Coupled mass, momentum, and energy equations for each region  
- Phase-change interface coupling via evaporation/condensation mass source  
- Adaptive time-step control (diffusion, Courant, source limits)  
- Temperature-dependent liquid/vapor sodium and steel properties (interpolated from tables)  
- Modular namespaces for physical properties: `vapor_sodium`, `liquid_sodium`, `steel`  
- Output files in `results/` for post-processing in Python  
- Optional animated plotting via `plot_data.py`  

---

## Repository Structure

```text

+-- HPCM.cpp
+-- results/
¦   +-- (data outputs)
+-- videos/
¦   +-- (video outputs)
+-- README.md

````

---

## Compilation

Requires a C++17 compiler.  
Example with **g++**:

```bash
g++ -std=c++17 -O3 coupling.cpp -Iinclude -o coupling
````

---

## Execution

Run the solver:

```bash
./HPCM
```

Optional plotting (Python 3.8+):

Just double-click the plot_data.vbs file

Optional videos creation (Python 3.8+):

Just double-click the make_videos.vbs file

---

## Dependencies

* **C++17**
* **Python 3.8+** (for post-processing)

  * `numpy`
  * `matplotlib`

---

## Example Output

Example plots show:

* Vapor, wick, and wall temperature profiles
* Interface heat fluxes and mass transfer rates
* Pressure and velocity distributions for vapor and wick

---

## References

* THROHPUT Technical Manual, NASA TM-2018-219886
* HPTAM: Heat Pipe Transient Analysis Model, NASA Glenn Research Center
* Faghri, A. *Heat Pipe Science and Technology*, 2nd Ed., 2016

---

## License

To be done.

```
