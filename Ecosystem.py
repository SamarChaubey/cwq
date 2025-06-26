from qiskit_ibm_provider import IBMProvider
import numpy as np
import os
from dotenv import load_dotenv
from qiskit_algorithms.optimizers import SPSA  
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

# Load environment variables from .env file
load_dotenv()
token = os.getenv("IBM_QUANTUM_TOKEN")
provider = IBMProvider(token=token)
backend = provider.get_backend('ibm_brisbane')

# Set random seed for reproducibility
np.random.seed(42)

# === Quantum Ansatz and Observable ===
def build_ansatz(params):
    # Example: 2-qubit ansatz, can be expanded
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc

observable = SparsePauliOp.from_list([("ZZ", 1.0)])  # Example observable

estimator = Estimator(options={"backend": backend})

# === STEP 1: Classical Ecosystem Simulator ===
def simulate_lotka_volterra(t_span, y0, params, t_eval=None):
    alpha, beta, gamma, delta = params

    def lotka_volterra(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]

    result = solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval, method='RK45')
    return result.t, result.y

# === STEP 2: Collapse Cost Function ===
def collapse_cost(x_vals, y_vals, epsilon=0.1):
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if x < epsilon or y < epsilon:
            return (len(x_vals) - i) ** 2  # Penalize early collapse
    return 0  # No collapse

# === STEP 3: Hybrid Quantum-Classical Cost Evaluation Wrapper ===
def evaluate_cost(params):
    # Classical part
    alpha, beta, gamma, delta = np.abs(params[:4])  # Ensure positivity
    t_span = (0, 25)
    y0 = [1.0, 1.0]  # Initial prey and predator populations
    t_eval = np.linspace(*t_span, 200)
    t, (x_vals, y_vals) = simulate_lotka_volterra(t_span, y0, [alpha, beta, gamma, delta], t_eval)
    collapse = collapse_cost(x_vals, y_vals)

    # Quantum part (use next 2 params for ansatz)
    qc = build_ansatz(params[4:6])
    q_result = estimator.run([qc], [observable]).result()
    quantum_val = np.real(q_result.values[0])

    # Combine: weight collapse and quantum value (tune weights as needed)
    cost = collapse / 100.0 + 0.1 * abs(quantum_val)
    return cost

# === STEP 4: Quantum Optimization via SPSA ===
def hybrid_quantum_optimize(cost_fn, num_params=6):
    optimizer = SPSA(maxiter=20)  # Reduce for QPU, increase for simulator

    def wrapped_cost(theta):
        return cost_fn(theta)

    initial_point = np.random.uniform(0.1, 1.0, num_params)
    result = optimizer.minimize(fun=wrapped_cost, x0=initial_point)
    return result.x, result.fun  # best_params, best_cost

# === STEP 5: Run Optimization ===
best_params, best_cost = hybrid_quantum_optimize(evaluate_cost)
alpha, beta, gamma, delta = np.abs(best_params[:4])

print(f"\n🔧 Best Parameters:")
print(f"  α (prey birth rate): {alpha:.4f}")
print(f"  β (predation rate):  {beta:.4f}")
print(f"  γ (predator death):  {gamma:.4f}")
print(f"  δ (predator growth): {delta:.4f}")
print(f"  ➤ Optimized collapse cost: {best_cost:.4f}")

# === STEP 6: Final Simulation with Optimized Parameters ===
t_span = (0, 25)
y0 = [1.0, 1.0]
t_eval = np.linspace(*t_span, 200)
t, (x_vals, y_vals) = simulate_lotka_volterra(t_span, y0, [alpha, beta, gamma, delta], t_eval)

# === STEP 6.5: Predict and Print Collapse Delay Time ===
epsilon = 0.1
collapse_time = None
for i, (x, y) in enumerate(zip(x_vals, y_vals)):
    if x < epsilon or y < epsilon:
        collapse_time = t[i]
        break
if collapse_time is None:
    collapse_time = t[-1]  # No collapse within simulation window

print(f"  ⏳ Collapse delayed until t = {collapse_time:.2f}")

# === STEP 7: Plot Results ===
plt.figure(figsize=(10, 5))
plt.plot(t, x_vals, label='Prey Population', color='blue')
plt.plot(t, y_vals, label='Predator Population', color='red')
plt.axhline(epsilon, color='gray', linestyle='--', label='Collapse Threshold (ε)')
plt.axvline(collapse_time, color='purple', linestyle=':', label=f'Collapse Time ({collapse_time:.2f})')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Ecosystem Dynamics (Quantum-Optimized Parameters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()