import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dotenv import load_dotenv

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# === ACCOUNT SETUP ===
load_dotenv()
IBM_QUANTUM_TOKEN="iVop7ecx6YavrViH2SNzxQnCpiE5Z5QawRU1bvnWLlO5"
token = IBM_QUANTUM_TOKEN

# Save credentials locally (one-time)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token=token,
    overwrite=True,
    set_as_default=True
)

# Load service and choose backend
service = QiskitRuntimeService()
backend = service.backend(name="ibm_brisbane")  # or use least_busy()

# Instantiate Estimator primitive
estimator = Estimator(mode=backend, options={"default_shots": 1024})

# === QUANTUM & CLASSICAL FUNCTIONS ===
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
            return (len(x_vals) - i) ** 2
    return 0

def evaluate_cost(params):
    alpha, beta, gamma, delta = np.abs(params[:4])
    t_span = (0, 25)
    t_eval = np.linspace(*t_span, 200)
    t_vals, (x_vals, y_vals) = simulate_lotka_volterra(t_span, [1, 1],
                                                       [alpha, beta, gamma, delta],
                                                       t_eval)
    collapse = collapse_cost(x_vals, y_vals)

    qc = build_ansatz(params[4:6])
    job = estimator.run([(qc, observable, [params[4:6]])])
    qres = job.result()[0]
    qval = np.real(qres.data.evs[0])

    return collapse / 100.0 + 0.1 * abs(qval)

def hybrid_quantum_optimize(cost_fn, num_params=6):
    opt = SPSA(maxiter=20)
    x0 = np.random.uniform(0.1, 1.0, size=num_params)
    res = opt.minimize(fun=cost_fn, x0=x0)
    return res.x, res.fun

# === RUN HYBRID OPTIMIZATION ===
best_theta, best_cost = hybrid_quantum_optimize(evaluate_cost)
alpha, beta, gamma, delta = np.abs(best_theta[:4])

print(f"\n🔧 Best Parameters:"
      f"\n  α: {alpha:.4f}, β: {beta:.4f}, γ: {gamma:.4f}, δ: {delta:.4f}"
      f"\n  Optimized cost: {best_cost:.4f}")

# === FINAL SIMULATION & PLOT ===
t, (x_vals, y_vals) = simulate_lotka_volterra((0, 25), [1, 1],
                                             [alpha, beta, gamma, delta],
                                             np.linspace(0, 25, 200))
epsilon = 0.1
collapse_time = next((tv for tv, xv, yv in zip(t, x_vals, y_vals)
                      if xv < epsilon or yv < epsilon), t[-1])
print(f"⏳ Collapse delayed until t = {collapse_time:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(t, x_vals, label="Prey", color="blue")
plt.plot(t, y_vals, label="Predator", color="red")
plt.axhline(epsilon, color="gray", linestyle="--", label="Threshold ε")
plt.axvline(collapse_time, color="purple", linestyle=":", label=f"Collapse @ t={collapse_time:.2f}")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Predator–Prey Dynamics with Quantum‑Optimized Parameters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
