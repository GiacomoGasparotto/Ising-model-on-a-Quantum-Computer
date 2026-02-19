import numpy as np
import gc
from time import perf_counter
import tracemalloc
from functools import reduce
from scipy import sparse
from scipy.sparse.linalg import eigsh

from debugger import message_checkpoints, matrix_checkpoints, vector_checkpoints

from qiskit import QuantumCircuit
from qtealeaves.observables import TNObservables, TNObsLocal, TNObsCorr, TNObsBondEntropy
from qmatchatea import run_simulation, QCBackend, QCConvergenceParameters

class TFIMChain:
    ### Constructor
    def __init__(self, n_qubits, T, n_steps):
        """
        Initializes the Transverse Field Ising Model (TFIM) Chain simulation object.

        Args:
            n_qubits (int): The number of qubits in the chain.
            T (float): Total time of the simulation/protocol.
            n_steps (int): Number of time steps for discretization.
        """
        
        self.n_qubits = n_qubits
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        self._Hzz = None
        self._Hx = None

    ### Hamiltonian definitions
    def circuit_hamiltonian(self, g, trotter_order=1, early_stop=None):
        """
        Constructs the quantum circuit implementing the time evolution of the TFIM 
        using Suzuki-Trotter decomposition.

        Args:
            g (float): The final strength of the transverse magnetic field.
            trotter_order (int, optional): The order of the Trotter approximation (1 or 2). Defaults to 1.
            early_stop (int, optional): The step index at which to stop the circuit generation (useful for snapshots). Defaults to None.

        Returns:
            QuantumCircuit: A Qiskit QuantumCircuit object representing the time evolution.
            
        Raises:
            ValueError: If `trotter_order` is not 1 or 2.
        """
        # Init the QuantumCircuit object
        qc = QuantumCircuit(self.n_qubits)

        # Compute z rotation angle
        theta_zz = -2 * self.dt
        
        # Provide an early stop option
        if early_stop is None:
            limit = self.n_steps
        else:
            limit = early_stop

        # Choose which Trotter order to use
        if trotter_order == 1:
            # Define the circuit
            for step in range(limit):
                # Compute transverse field parameter
                current_time = step * self.dt
                gk = (current_time / self.T) * g
                
                # Compute x rotation angle
                theta_x = -2 * gk * self.dt

                ## X interaction
                #  - Rx on qubit i
                for i in range(self.n_qubits): 
                    qc.rx(theta_x, i)
                
                ## ZZ interaction:
                #  1 - CNOT (control i, target i+1)
                #  2 - Rz on qubit i+1
                #  3 - CNOT (control i, target i+1)
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i+1)
                    qc.rz(theta_zz, i+1)
                    qc.cx(i, i+1)

        elif trotter_order == 2:
            # Define the circuit
            for step in range(limit):
                # Compute transverse field parameter
                current_time = (step + 0.5) * self.dt
                gk = (current_time/self.T) * g
                
                # Compute x rotation angle
                theta_x = -2*gk*(self.dt/2) # Half update

                ## X interaction
                #  - First half Rx on qubit i
                for i in range(self.n_qubits): 
                    qc.rx(theta_x, i)
                
                ## ZZ interaction:
                #  1 - CNOT (control i, target i+1)
                #  2 - Rz on qubit i+1
                #  3 - CNOT (control i, target i+1)
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i+1)
                    qc.rz(theta_zz, i+1)
                    qc.cx(i, i+1)
                
                ## X interaction
                #  - Second half Rx on qubit i
                for i in range(self.n_qubits): 
                    qc.rx(theta_x, i)
        
        else:
            raise ValueError("ERROR: Trotter decomposition available only at 1st and 2nd order")
            
        return qc
    
    def draw_circuit(self, g=0.5, trotter_order=1, style="mpl"):
        """
        Draws the quantum circuit for the Hamiltonian evolution.

        Args:
            g (float, optional): The transverse field strength. Defaults to 0.5.
            trotter_order (int, optional): The order of the Trotter approximation. Defaults to 1.
            style (str, optional): The style of the drawing (e.g., "mpl", "text"). Defaults to "mpl".

        Returns:
            matplotlib.figure.Figure or str: The visualization of the circuit.
        """
        qc = self.circuit_hamiltonian(g=g, trotter_order=trotter_order, early_stop=None)
        return qc.draw(output=style)

    def sparse_hamiltonian(self):
        """
        Constructs the exact sparse Hamiltonian matrices for the 1D Transverse Field Ising Model.

        Returns:
            tuple: A tuple (Hzz, Hx) containing:
                - Hzz (scipy.sparse.csr_matrix): The interaction term of the Hamiltonian.
                - Hx (scipy.sparse.csr_matrix): The transverse field term of the Hamiltonian.
        """
        if self._Hzz is not None: return self._Hzz, self._Hx

        # Initialize Hamiltonian operators
        Id = sparse.eye(2, dtype=complex)
        sigmax = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
        sigmaz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
        
        d = 2**self.n_qubits # Hilbert space dimension
        transverse_field_term = sparse.csr_matrix((d, d), dtype=complex)
        interaction_term = sparse.csr_matrix((d, d), dtype=complex)
        
        for i in range(self.n_qubits):
            # Transverse field
            transverse_ops = [Id]*self.n_qubits
            transverse_ops[i] = sigmax
            transverse_field_term += reduce(sparse.kron, transverse_ops)

            # Interaction
            if i != (self.n_qubits-1):
                interaction_ops = [Id]*self.n_qubits
                interaction_ops[i] = sigmaz
                interaction_ops[(i+1)] = sigmaz
                interaction_term += reduce(sparse.kron, interaction_ops)
        
        self._Hzz = interaction_term
        self._Hx = transverse_field_term

        # Checkpoints
        matrix_checkpoints(verbosity=0, A=interaction_term.toarray(), debug=True) # check hermitian
        matrix_checkpoints(verbosity=0, A=transverse_field_term.toarray(), debug=True) # check hermitian
        matrix_checkpoints(verbosity=1, A=interaction_term.toarray(), debug=True) # check diagonal

        return interaction_term, transverse_field_term
    
    ### Energy computation
    def exact_GS_energy(self, g, stride=10):
        """
        Computes the exact Ground State energy using the Jordan-Wigner 
        transformation to map the TFIM into free fermions.

        Args:
            g (float): The target transverse field strength.
            stride (int, optional): The number of steps to skip between energy calculations. Defaults to 10.

        Returns:
            tuple: A tuple (times, energies) containing:
                - times (list): The list of time points.
                - energies (list): The list of exact ground state energies at each time point.
        """
        # Lists to store results
        times = []
        energies = []
        
        message_checkpoints(0, f"\nEXACT ENERGY EVOLUTION (FREE FERMIONS) - Starting evolution for g={g} and N={self.n_qubits}...", True)
        
        for step in range(0, self.n_steps+1, stride):
            t = step*self.dt
            gk = (t/self.T) * g
            
            # Build the NxN matrix for free fermions
            # M_ij = g * delta_ij + delta_{i, j+1}
            M = np.zeros((self.n_qubits, self.n_qubits))
            
            # Transverse field on the principal diagonal
            np.fill_diagonal(M, gk)
            
            # Interaction term on the upper diagonal 
            np.fill_diagonal(M[:, 1:], 1.0) 
            
            # The singular values represent the fermions energies
            sv = np.linalg.svd(M, compute_uv=False)
            
            # Ground state energy is the negative sum of the excitation energies
            Egs = -np.sum(sv)
            
            times.append(t)
            energies.append(Egs)

            # Print results
            print(f"t={t:.2f}, E={Egs:.4f}")
            
        return times, energies

    def _runMPSsimulation(self, qc, ansatz="MPS", max_bond_dimension=128, cut_ratio=1e-9, observables=None):
        """
        Internal helper function to run a Matrix Product State (MPS) simulation using QMatchaTea.

        Args:
            qc (QuantumCircuit): The quantum circuit to simulate.
            ansatz (str, optional): The tensor network ansatz to use. Defaults to "MPS".
            max_bond_dimension (int, optional): The maximum allowed bond dimension for the MPS. Defaults to 128.
            cut_ratio (float, optional): The singular value truncation threshold. Defaults to 1e-9.
            observables (TNObservables, optional): The observables to measure during simulation. Defaults to None.

        Returns:
            dict: The simulation results containing measured observables.
            
        Raises:
            ValueError: If `observables` are not provided.
        """
        # Define the observables
        if observables is None:
            raise ValueError("ERROR: provide observables to run the simulation.")

        # Define backend and convergence parameters to run the TN simulation
        backend = QCBackend(precision="Z", device="cpu", ansatz=ansatz, tensor_module="numpy")
        conv_params = QCConvergenceParameters(max_bond_dimension=max_bond_dimension, cut_ratio=cut_ratio)

        return run_simulation(qc, observables=observables, convergence_parameters=conv_params, backend=backend)

    def MPS_GS_energy(self, g, m, cut_ratio=1e-9, stride=10):
        """
        Computes the ground state energy evolution using the MPS ansatz simulation.

        Args:
            g (float): The target transverse field strength.
            m (int): The maximum bond dimension for the MPS simulation.
            cut_ratio (float, optional): The singular value truncation threshold. Defaults to 1e-9.
            stride (int, optional): The number of steps to skip between energy calculations. Defaults to 10.

        Returns:
            tuple: A tuple (times, energies) containing:
                - times (list): The list of time points.
                - energies (list): The list of simulated ground state energies.
        """
        # Lists to store results
        times = []
        energies = []
        
        # Setup energy observables
        obs = TNObservables()
        obs += TNObsLocal("Mean_X", "X")
        obs += TNObsCorr("Corr_ZZ", ["Z", "Z"])
        
        message_checkpoints(0, f"\nSIMULATED ENERGY EVOLUTION - Starting evolution for g={g} and N={self.n_qubits}...", True)

        for step in range(0, self.n_steps+1, stride):
            t = step * self.dt
            
            # Generate the Ising circuit
            qc = self.circuit_hamiltonian(g, early_stop=step)
            
            try:
                # Run MPS simulation
                sim = self._runMPSsimulation(qc=qc, 
                                             ansatz="MPS", 
                                             max_bond_dimension=m,
                                             observables=obs,
                                             cut_ratio=cut_ratio)
                
                # Extract data
                # <X>
                x = sim.observables.get("Mean_X")
                if x is None: raise ValueError("Mean_X missing")
                sum_x = np.sum(np.real(np.array(x).flatten()))
                
                # <ZZ>
                zz = sim.observables.get("Corr_ZZ")
                if zz is None: raise ValueError("Corr_ZZ missing")
                
                sum_zz = 0.0
                if isinstance(zz, dict):
                    for v in zz.values():
                        sum_zz += np.real(v)
                else:
                    sum_zz = np.sum(np.real(np.diagonal(zz, offset=1)))
                
                # Compute total energy
                # E(t) = -g(t)*<X> - <ZZ>
                g_curr = (t/self.T) * g
                total_energy = (-g_curr * sum_x) - sum_zz
                
                times.append(t)
                energies.append(total_energy)
                
                print(f"t={t:5.2f}, E={total_energy:8.4f}")
                
            except Exception as e:
                print(f"ERROR at step {step}: {e}")
                times.append(t)
                energies.append(np.nan)
                break
        
        message_checkpoints(0, "\nCalculation DONE.", True)
        return times, energies
    
    ### Magnetization computation
    def _build_mag_operators(self):
        """
        Constructs the total z-magnetization operator for the system.

        Returns:
            scipy.sparse.csr_matrix: The sparse matrix representing the sum of sigma-z operators on all qubits.
        """
        # Initialize operators
        Id = sparse.identity(2, format="csr")
        sigmaz = sparse.csr_matrix([[1., 0.], [0., -1.]])
        SumZ = sparse.csr_matrix((2**self.n_qubits, 2**self.n_qubits))

        # Compute magnetization
        for i in range(self.n_qubits):
            ops = [Id]*self.n_qubits
            ops[i] = sigmaz
            SumZ += reduce(sparse.kron, ops)

        return SumZ
    
    def compute_phase_transition_data(self, g_max, bond_dim, stride=10):
        """
        Simulates the system evolution across a phase transition to collect order parameter and heatmap data.

        Args:
            g_max (float): The maximum transverse field strength at the end of the quench.
            bond_dim (int): The maximum bond dimension for the MPS simulation.
            stride (int, optional): The step interval for data collection. Defaults to 10.

        Returns:
            dict: A dictionary containing:
                - "g_axis" (list): The list of transverse field values.
                - "mps_order_param" (list): The order parameter computed via MPS.
                - "exact_order_param" (list): The exact order parameter computed via sparse diagonalization.
                - "heatmap_data" (list): Local Z-magnetization profiles for heatmap visualization.
                - "N" (int): The number of qubits.
        """
        # Setup the observables
        obs = TNObservables()
        obs += TNObsLocal("Local_Z", "Z") 
        obs += TNObsCorr("Corr_ZZ", ["Z", "Z"])
        
        # Compute exact hamiltonians
        Hzz, Hx = self.sparse_hamiltonian()
        Sum_Z_comp = self._build_mag_operators()
        
        results = {
            "g_axis": [],
            "mps_order_param": [],
            "exact_order_param": [],
            "heatmap_data": [],
            "N": self.n_qubits
        }
        
        message_checkpoints(0, f"Phase transition simulation (N={self.n_qubits})...", True)
        
        for step in range(0, self.n_steps+1, stride):
            # Compute g at the current time
            t = step*self.dt
            g = (t / self.T) * g_max
            results["g_axis"].append(g)
            
            # MPS simulation
            qc = self.circuit_hamiltonian(g_max, early_stop=step)
            sim = self._runMPSsimulation(qc, max_bond_dimension=bond_dim, observables=obs)
            
            # Heatmap Data
            local_z = np.real(np.array(sim.observables.get("Local_Z")).flatten())
            results["heatmap_data"].append(np.abs(local_z))
            
            # Order Parameter MPS
            res_zz = sim.observables.get("Corr_ZZ")
            zz_sum = 0.0
            if isinstance(res_zz, dict):
                for i in range(self.n_qubits):
                    zz_sum += 1.0
                    for j in range(i+1, self.n_qubits):
                        val = res_zz.get(f"{i}_{j}") or res_zz.get((i, j))
                        if val: zz_sum += 2 * np.real(val)
            else:
                zz_mat = np.array(res_zz)
                np.fill_diagonal(zz_mat, 1.0)
                zz_sum = np.sum(np.real(zz_mat))
            results["mps_order_param"].append(np.sqrt(zz_sum) / self.n_qubits)
            
            # Exact Hamiltonian simulation
            H = -Hzz - g*Hx
            _, eigvecs = eigsh(H, k=1, which='SA')
            GS = eigvecs[:, 0]

            # Check normalization
            vector_checkpoints(verbosity=0, v=GS, debug=True)
            
            v = Sum_Z_comp @ GS
            op_exact = np.sqrt(np.real(v.conj().T @ v)) / self.n_qubits
            results["exact_order_param"].append(op_exact)
            
            message_checkpoints(0, f"g={g:.2f}, MPS Order={results['mps_order_param'][-1]:.3f}", True)

        message_checkpoints(0, "\nComputation completed.", True)
        return results

    ### Bond dimension computation
    def benchmark_bond_dimensions(self, g, m_list):
        """
        Benchmarks the simulation performance (energy accuracy, time, memory) for different bond dimensions.

        Args:
            g (float): The target transverse field strength for the benchmark.
            m_list (list): A list of bond dimensions to test.

        Returns:
            list: A list of dictionaries, where each dictionary contains:
                - "bond_dim": The tested bond dimension.
                - "energy": The computed energy.
                - "time_s": The execution time in seconds.
                - "memory_mb": The peak memory usage in MB.
        """
        # Generate the Ising circuit
        qc = self.circuit_hamiltonian(g, early_stop=self.n_steps)
        results = []
        
        # Define the Observables
        obs = TNObservables()
        obs += TNObsLocal("Mean_X", "X")
        obs += TNObsCorr("Corr_ZZ", ["Z", "Z"])

        message_checkpoints(0, f"BENCHMARK: Testing bond dims {m_list} for g={g}...", True)

        for m in m_list:
            
            gc.collect() # cleanup the memory

            # Start measurement
            tracemalloc.start()
            start_t = perf_counter()
            
            try:
                # Run Simulation
                sim = self._runMPSsimulation(qc, max_bond_dimension=m, observables=obs)
                
                # Stop measurement
                stop_t = perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                elapsed_time = stop_t - start_t
                peak_mb = peak / (1024*1024) # Convert bytes to MB

                # Compute final energy
                mean_x = np.sum(np.real(sim.observables["Mean_X"]))
                
                res_zz = sim.observables["Corr_ZZ"]
                mean_zz = 0.0
                if isinstance(res_zz, dict):
                    for v in res_zz.values():
                        mean_zz += np.real(v)
                else:
                    mean_zz = np.sum(np.real(np.diagonal(res_zz, offset=1)))
                
                E_final = - g*mean_x -mean_zz
                
                # Save all metrics
                entry = {
                    "bond_dim": m,
                    "energy": E_final,
                    "time_s": elapsed_time,
                    "memory_mb": peak_mb
                }
                results.append(entry)

                message_checkpoints(0, f"m={m}: E={E_final:.4f}, Time={elapsed_time:.2f}s, Memory={peak_mb:.2f}MB", True)
                
            except Exception as e:
                tracemalloc.stop() # Stop memory tracking even if it fails
                print(f"m={m} Failed: {e}")
                results.append({
                    "bond_dim": m, 
                    "energy": np.nan, 
                    "time_s": np.nan, 
                    "memory_mb": np.nan
                })
        
        return results
    
    def compute_full_correlation_matrix(self, g_vals, g_max_quench=3.0, m=128):
        """
        Computes the full spin-spin (ZZ) correlation matrix for specific transverse field values.

        Args:
            g_vals (list): A list of transverse field values at which to compute the correlations.
            g_max_quench (float, optional): The maximum field strength of the quench protocol. Defaults to 3.0.
            m (int, optional): The maximum bond dimension for the simulation. Defaults to 128.

        Returns:
            dict: A dictionary mapping each `g` value to its corresponding NxN correlation matrix (numpy array).
        """        
        matrices = {}
        
        message_checkpoints(0, f"Starting FULL ZZ correlation matrix analysis (N={self.n_qubits})...", True)
        
        for g in g_vals:
            message_checkpoints(0, f"Extracting state at g={g:.2f}...", True)
            
            step = int(round((g / g_max_quench) * self.n_steps))
            
            obs = TNObservables()
            obs += TNObsCorr("Corr_ZZ", ["Z", "Z"])
            
            qc = self.circuit_hamiltonian(g_max_quench, early_stop=step)
            res = self._runMPSsimulation(qc, max_bond_dimension=m, observables=obs)
            
            zz_data = res.observables.get("Corr_ZZ", {})
            
            # Ricostruiamo la matrice N x N completa
            mat = np.eye(self.n_qubits) # Diagonale a 1
            if isinstance(zz_data, dict):
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        val = np.real(zz_data.get(f"{i}_{j}") or zz_data.get((i, j), 0.0))
                        mat[i, j] = val
                        mat[j, i] = val # È simmetrica
            else:
                mat = np.real(zz_data)

            # Check diagonal
            matrix_checkpoints(verbosity=0, A=mat, debug=True)
            matrix_checkpoints(verbosity=3, A=mat, debug=True)
                
            matrices[g] = mat
            message_checkpoints(0, "DONE.", True)
            
            del qc, res, obs, zz_data
            gc.collect()
            
        return matrices
    
    def entropy(self, gmax, m=128, stride=10):
        """
        Computes the Von Neumann entanglement entropy across the time evolution.

        Args:
            gmax (float): The maximum transverse field strength at the end of the protocol.
            m (int, optional): The maximum bond dimension for the simulation. Defaults to 128.
            stride (int, optional): The step interval for entropy calculation. Defaults to 10.

        Returns:
            tuple: A tuple (g_axis_ent, entropy_mid, entropy_profiles) containing:
                - g_axis_ent (list): The list of transverse field values corresponding to the measurements.
                - entropy_mid (list): The half-chain entanglement entropy values.
                - entropy_profiles (list): A list of entropy profiles across all bonds for each time step.
        """
        # Setup the observables
        obs = TNObservables()
        obs += TNObsBondEntropy()

        # Initialize lists
        g_axis_ent = []
        entropy_mid = []
        entropy_profiles = []

        message_checkpoints(0, f"Begin Von-Neumann entropy calculation (N={self.n_qubits})...\n", True)

        for step in range(0, self.n_steps + 1, stride):
            # Compute time and field
            t = step * self.dt
            g = (t / self.T) * gmax
            g_axis_ent.append(g)

            # Create the circuit and run the simulation
            qc = self.circuit_hamiltonian(gmax, early_stop=step)
            sim = self._runMPSsimulation(qc, max_bond_dimension=m, observables=obs)

            # Extract the entropy
            ee_res = sim.observables.get("bond_entropy", {})
            
            S_profile = []
            if isinstance(ee_res, dict) and len(ee_res) > 0:
                sorted_keys = sorted(ee_res.keys(), key=lambda x: x[1])
                S_profile = [np.real(ee_res[k]) for k in sorted_keys]

            # Save the half chain entropy
            if len(S_profile) > 0:
                mid_idx = len(S_profile) // 2
                entropy_mid.append(S_profile[mid_idx])
            else:
                entropy_mid.append(0.0)

            # Save the space profile
            entropy_profiles.append(S_profile)

            message_checkpoints(0, f"g={g:.2f}, Half-chain Entropy = {entropy_mid[-1]:.3f}", True)

            # Cleanup
            del qc, sim, ee_res
            gc.collect()

        message_checkpoints(0, "DONE.", True)
        
        return g_axis_ent, entropy_mid, entropy_profiles