# Dynamics of the Transverse-Field Ising Model via Tensor Networks

This repository contains the code, data, and final report for the simulation of the 1D Transverse-Field Ising Model (TFIM) driven across a quantum phase transition. 

The project exploits the **Matrix Product State (MPS)** ansatz within the Tensor Network framework to simulate the time evolution of the system, overcoming the exponential memory limits of exact statevector simulations.

## Abstract
We investigate the non-equilibrium dynamics of the TFIM driven by a linearly increasing transverse magnetic field. By mapping the Hamiltonian dynamics onto a quantum circuit, we benchmark the MPS algorithm's scalability and determine the bond dimension requirements to surpass exact diagonalization limits ($N=20$). 

Starting from a ferromagnetic ground state, the system is driven towards the paramagnetic phase. The simulation captures the breakdown of adiabaticity predicted by the **Kibble-Zurek mechanism**. Analysis of the residual energy, coherent order parameter oscillations, and broadened spin-spin correlations confirms the generation of quasiparticle excitations. Furthermore, the continuous step-like growth of the Von Neumann entropy tracking a volume-law validates the system's evolution into a highly entangled non-equilibrium state.

## Key Physical Phenomena Explored
* **Quantum Phase Transition:** Crossing the critical point $g_c = 1$.
* **Kibble-Zurek Mechanism:** Breakdown of adiabaticity and defect formation due to critical slowing down.
* **Spin-Spin Correlations:** Transition from long-range order (ferromagnet) to exponential decay (paramagnet), highlighting the algebraic broadening at the critical point and interference patterns due to Open Boundary Conditions (OBC).
* **Entanglement Entropy:** Volume-law entanglement production ($S_{N/2} \propto t$) driven by the coherent propagation of entangled quasiparticles, successfully captured using a high bond dimension ($m=256$).

## Technologies & Libraries
* **[Qiskit](https://qiskit.org/):** Used for exact statevector simulations to benchmark the Tensor Network approach up to $N=20$ qubits.
* **[QTeaLeaves / QMatchaTea](https://quantumtealeaves.readthedocs.io/):** Used for the Tensor Network / MPS simulations, dynamic truncation via SVD, and measurement of advanced observables (e.g., bond entropy, correlation matrices).
* **NumPy & SciPy:** For numerical operations and sparse matrix exact diagonalization.
* **Matplotlib:** For rendering heatmap correlations and dynamically scaling observables.

## Repository Structure
* `main.ipynb`: The main Jupyter Notebook containing the full pipeline (circuit generation, Qiskit benchmark, MPS simulation, and plotting).
* `tfim.py`: Python module containing the `TFIMChain` class, which handles the Hamiltonian definition, Trotterized circuit building, and backend execution.
* `plots/`: Directory containing the generated PDF figures (Entropy step-like growth, 2D Correlation Heatmaps, Scaling benchmarks).
* `QCReport.pdf`: The final academic report detailing the theoretical background, methodology, and physical interpretation of the results.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/tfim-tensor-networks.git](https://github.com/yourusername/tfim-tensor-networks.git)
   cd tfim-tensor-networks
