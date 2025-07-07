import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dotenv import load_dotenv

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# === Load token and optional instance, no saving to disk ===
load_dotenv()
TOKEN = os.getenv("IBM_QUANTUM_TOKEN")
INSTANCE = os.getenv("IBM_QUANTUM_INSTANCE")  # Optional: your CRN or instance name

if not TOKEN:
    raise ValueError("Missing IBM_QUANTUM_TOKEN in .env – aborting.")

service = QiskitRuntimeService(
    channel="ibm_quantum",  # Use new platform; equivalent to "ibm_cloud" :contentReference[oaicite:0]{index=0}
    token=TOKEN,
    instance=INSTANCE
)

# Quick check that authentication works
print("Available backends:", service.backends())

# === Estimator setup ===
backend = service.backend("ibm_brisbane")
estimator = Estimator(mode=backend, options={"default_shots": 1024})

# === Core Algorithm ===
np.random.seed(42)

def build_ansatz(params):
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc

observable = SparsePauliOp.from_list([("ZZ", 1.0)])

def simulate_lotka_volterra(t_span, y0, params, t_eval=None):
    alpha, beta, gamma, delta = params
    def lv(t, z):
        x, y = z
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]
    sol = solve_ivp(lv, t_span, y0, t_eval=t_eval, method="RK45")
    return sol.t, sol.y

def collapse_cost(x_vals, y_vals, eps=0.1):
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if x < eps or y < eps:
            return (len(x_vals) - i)**2
    return 0

def evaluate_cost(params):
    alpha, beta, gamma, delta = np.abs(params[:4])
    t, (x, y) = simulate_lotka_volterra((0, 25), [1,1],
                                        [alpha, beta, gamma, delta],
                                        np.linspace(0,25,200))
    collapse = collapse_cost(x, y)

    qc = build_ansatz(params[4:6])
    job = estimator.run([(qc, observable, [params[4:6]])])
    ev = job.result()[0].data.evs[0]
    quantum_term = abs(np.real(ev))

    return collapse / 100.0 + 0.1 * quantum_term

def hybrid_quantum_optimize(cost_fn, num_params=6):
    opt = SPSA(maxiter=20)
    x0 = np.random.uniform(0.1, 1.0, size=num_params)
    res = opt.minimize(fun=cost_fn, x0=x0)
    return res.x, res.fun

# === Run Optimization ===
best_theta, best_cost = hybrid_quantum_optimize(evaluate_cost)
alpha, beta, gamma, delta = np.abs(best_theta[:4])
print(f"\n🔧 Best params: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, δ={delta:.4f}")
print(f"Optimized cost: {best_cost:.4f}")

# === Final Simulation & Plot ===
t, (x, y) = simulate_lotka_volterra((0,25), [1,1],
                                     [alpha, beta, gamma, delta],
                                     np.linspace(0,25,200))
epsilon = 0.1
collapse_time = next((tv for tv, xv, yv in zip(t, x, y) if xv<epsilon or yv<epsilon), t[-1])
print(f"⏳ Collapse at t = {collapse_time:.2f}")

plt.figure(figsize=(10,5))
plt.plot(t, x, label="Prey", color="blue")
plt.plot(t, y, label="Predator", color="red")
plt.axhline(epsilon, linestyle="--", color="gray", label="Threshold ε")
plt.axvline(collapse_time, linestyle=":", color="purple", label=f"Collapse @ t={collapse_time:.2f}")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Predator–Prey Dynamics (Quantum‑Optimized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
