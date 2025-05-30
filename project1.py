import os
import json
import numpy as np
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# --- Load environment and initialize IBM Quantum service ---
load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")
service = QiskitRuntimeService(channel="ibm_quantum", token=token)
backend = service.backend(name='ibm_brisbane')

# --- Load real wind speed data ---
def get_real_weather_data_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    # Extract the "val" field from each entry
    values = np.array([entry.get("val", 0) for entry in data])
    print(f"Loaded {len(values)} wind speed data points from {file_path}")
    return values

# Optionally, allow user to specify file path
import sys
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = r"C:\Users\sanje\cwq\windSpeed.json"

wind_speeds = get_real_weather_data_from_file(file_path)
timesteps = len(wind_speeds)
frames = timesteps
t_eval = np.linspace(0, timesteps - 1, timesteps)

# --- Quantum PDE Solver Class ---
class QuantumPDESolver:
    def __init__(self, num_qubits=4, reps=2):
        self.num_qubits = num_qubits
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)
        self.optimizer = COBYLA(maxiter=100)
        self.estimator = StatevectorEstimator()

    def create_quantum_hamiltonian(self, grid_size):
        num_qubits = min(self.num_qubits, grid_size)
        terms = []
        for i in range(num_qubits):
            pauli_str = ['I'] * num_qubits
            pauli_str[i] = 'Z'
            terms.append((''.join(pauli_str), 1.0))
        if num_qubits > 1:
            for i in range(num_qubits - 1):
                pauli_str = ['I'] * num_qubits
                pauli_str[i] = 'X'
                pauli_str[i + 1] = 'X'
                terms.append((''.join(pauli_str), 0.5))
        return SparsePauliOp.from_list(terms)

    def solve(self, initial_params=None):
        H = self.create_quantum_hamiltonian(self.num_qubits)

        def quantum_cost(params):
            try:
                param_dict = dict(zip(self.ansatz.parameters, params))
                circuit = self.ansatz.assign_parameters(param_dict)
                job = self.estimator.run([circuit], [H])
                result = job.result()
                expectation = result.values[0]
                print(f"Params: {np.round(params, 3)}, Cost: {expectation:.6f}")
                return np.real(expectation)
            except Exception as e:
                print(f"Quantum circuit evaluation error: {e}")
                return float('inf')

        num_params = len(self.ansatz.parameters)
        if initial_params is None:
            initial_params = np.random.random(num_params) * 2 * np.pi
        else:
            initial_params = np.array(initial_params)
            if initial_params.size != num_params:
                initial_params = np.random.random(num_params) * 2 * np.pi

        result = self.optimizer.minimize(
            fun=quantum_cost,
            x0=initial_params,
            bounds=[(0, 2 * np.pi)] * num_params
        )

        # FIX: Don't use .status, use .success or just print message
        if hasattr(result, 'success'):
            print(f"Optimization success: {result.success}, message: {result.message}")
        else:
            print(f"Optimization message: {getattr(result, 'message', 'No message')}")
        return result.x  # optimized parameters

# --- Load prior optimized params or start fresh ---
PARAMS_FILE = "optimized_params.npy"
if os.path.exists(PARAMS_FILE):
    print("Loading previous optimized parameters...")
    prior_params = np.load(PARAMS_FILE)
else:
    prior_params = None

# --- Initialize Quantum PDE Solver with prior parameters ---
quantum_solver = QuantumPDESolver(num_qubits=4, reps=2)
optimized_params = quantum_solver.solve(initial_params=prior_params)

# --- Save updated optimized parameters for next run ---
np.save(PARAMS_FILE, optimized_params)
print(f"Saved optimized parameters for next run: {optimized_params}")

# --- Setup and solve Navier-Stokes with parameters affecting viscosity or other factors ---
avg_param = np.mean(optimized_params) / (2*np.pi)
NU_BASE = 0.1  # Increased viscosity for stability
NU = NU_BASE * (0.5 + avg_param)

print(f"Using viscosity NU = {NU:.5f} based on quantum solver parameters.")

shape = (3, 3, 3)  # Smaller grid for stability
u0 = np.ones(shape) * wind_speeds[0]  # Use hour 0 wind speed everywhere
v0 = np.zeros_like(u0)
w0 = np.zeros_like(u0)
y0 = np.stack([u0, v0, w0]).flatten()

timesteps = len(wind_speeds)
t_eval = np.linspace(0, timesteps - 1, timesteps)

def navier_stokes_3D(t, y, shape, nu, wind_speeds=None):
    # Unpack the velocity fields
    n = np.prod(shape)
    u = y[:n].reshape(shape)
    v = y[n:2*n].reshape(shape)
    w = y[2*n:].reshape(shape)

    # Compute derivatives (simple finite difference, periodic BCs)
    dudx = np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)
    dudy = np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)
    dudz = np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)

    dvdx = np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)
    dvdy = np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)
    dvdz = np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)

    dwdx = np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)
    dwdy = np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)
    dwdz = np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)

    # Laplacian (diffusion term)
    lap_u = (
        np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0) +
        np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) +
        np.roll(u, -1, axis=2) + np.roll(u, 1, axis=2) - 6 * u
    )
    lap_v = (
        np.roll(v, -1, axis=0) + np.roll(v, 1, axis=0) +
        np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) +
        np.roll(v, -1, axis=2) + np.roll(v, 1, axis=2) - 6 * v
    )
    lap_w = (
        np.roll(w, -1, axis=0) + np.roll(w, 1, axis=0) +
        np.roll(w, -1, axis=1) + np.roll(w, 1, axis=1) +
        np.roll(w, -1, axis=2) + np.roll(w, 1, axis=2) - 6 * w
    )

    # Nonlinear advection term (u·∇)u, (v·∇)v, (w·∇)w
    du_dt = - (u * dudx + v * dudy + w * dudz) + nu * lap_u
    dv_dt = - (u * dvdx + v * dvdy + w * dvdz) + nu * lap_v
    dw_dt = - (u * dwdx + v * dwdy + w * dwdz) + nu * lap_w

    return np.concatenate([du_dt.flatten(), dv_dt.flatten(), dw_dt.flatten()])

sol = solve_ivp(
    lambda t, y: navier_stokes_3D(t, y, shape, NU, wind_speeds),
    [0, timesteps - 1],
    y0,
    t_eval=t_eval,
    method='Radau',
    rtol=1e-3,
    atol=1e-5
)

if not sol.success:
    print(f"ODE integration failed: {sol.message}")
    exit(1)

n = u0.size
frames = len(t_eval)
u_sim = np.zeros((shape[0], shape[1], shape[2], frames))
v_sim = np.zeros_like(u_sim)
w_sim = np.zeros_like(u_sim)

for i in range(frames):
    u_sim[..., i] = sol.y[:n, i].reshape(shape)
    v_sim[..., i] = sol.y[n:2*n, i].reshape(shape)
    w_sim[..., i] = sol.y[2*n:, i].reshape(shape)

real_velocities = wind_speeds[:frames]
simulated_means = np.array([
    np.mean(np.sqrt(u_sim[..., i]**2 + v_sim[..., i]**2 + w_sim[..., i]**2)) for i in range(frames)
]).reshape(-1, 1)

# Nudge simulated values toward real values and store successful parameter sets
nudged_simulated = simulated_means.flatten().copy()
nudged_params = []
nudging_strength = 0.25 # Reduced for more realism

for i in range(frames):
    real_val = real_velocities[i]
    sim_val = nudged_simulated[i]
    # Add a small random noise to prevent exact equality
    epsilon = np.random.uniform(-0.001, 0.001)
    nudged_val = sim_val + nudging_strength * (real_val - sim_val) + epsilon
    # If nudged value is closer to real value, store the parameters
    if abs(nudged_val - real_val) < abs(sim_val - real_val):
        nudged_simulated[i] = nudged_val
        nudged_params.append({
            "frame": int(i),
            "nudged_value": float(nudged_val),
            "real_value": float(real_val),
            "sim_value": float(sim_val),
            "NU": float(NU),
            "optimized_params": [float(x) for x in optimized_params]
        })

# Optionally, save the nudged_params for future learning
with open("nudged_params_log.json", "w") as f:
    json.dump(nudged_params, f, indent=2)

print(f"Stored {len(nudged_params)} nudged parameter records.")

# Use nudged_simulated for regression, but keep original simulated_means for plotting
nudged_simulated_reshaped = nudged_simulated.reshape(-1, 1)

# Scale data for better regression performance
scaler_sim = StandardScaler()
scaler_real = StandardScaler()

sim_scaled = scaler_sim.fit_transform(nudged_simulated_reshaped)
real_scaled = scaler_real.fit_transform(real_velocities.reshape(-1, 1))

# Use a higher-degree polynomial regression
polyreg = make_pipeline(PolynomialFeatures(4), LinearRegression())
polyreg.fit(sim_scaled, real_scaled)
adjusted_scaled = polyreg.predict(sim_scaled)
adjusted_velocities = scaler_real.inverse_transform(adjusted_scaled).flatten()

# Optionally clip to real data range
adjusted_velocities = np.clip(adjusted_velocities, real_velocities.min(), real_velocities.max())

# Ensure real_velocities and adjusted_velocities are 1D arrays
real_velocities_1d = np.array(real_velocities).flatten()
adjusted_velocities_1d = np.array(adjusted_velocities).flatten()

fig, ax = plt.subplots(2, 1, figsize=(12, 9))

# Plot only real and final adjusted simulated values
ax[0].plot(range(frames), real_velocities_1d, label='Real Wind Speed', marker='o')
ax[0].plot(range(frames), adjusted_velocities_1d, label='Adjusted Simulated Speed', marker='s')
ax[0].set_xlabel("Time (hours)")
ax[0].set_ylabel("Wind Speed (units)")
ax[0].set_title("Wind Speed Comparison")
ax[0].legend()
ax[0].grid(True)

# Show the velocity magnitude slice at the final timestep
z_slice = shape[2] // 2
velocity_magnitude = np.sqrt(
    u_sim[:, :, z_slice, -1] ** 2 +
    v_sim[:, :, z_slice, -1] ** 2 +
    w_sim[:, :, z_slice, -1] ** 2
)
im = ax[1].imshow(
    velocity_magnitude,
    origin='lower',
    cmap='viridis',
    extent=[0, shape[0], 0, shape[1]],
    aspect='auto'
)
fig.colorbar(im, ax=ax[1], label='Velocity Magnitude')
ax[1].set_title(f"Simulated Velocity Magnitude (Slice z={z_slice}) at Final Timestep")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")

plt.tight_layout()
plt.show()
