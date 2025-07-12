import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA  # ✅ Qiskit 2.x compatible
from qiskit.primitives import BaseSamplerV1        # ✅ Qiskit 2.x compatible

np.random.seed(42)

# === Quantum ansatz circuit ===
def build_ansatz(params):
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc

# === Observable (Pauli ZZ) ===
observable = SparsePauliOp.from_list([("ZZ", 1.0)])

# === Classical predator–prey dynamics ===
def simulate_lotka_volterra(t_span, y0, params, t_eval=None):
    alpha, beta, gamma, delta = params
    def lv(t, z):
        x, y = z
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]
    sol = solve_ivp(lv, t_span, y0, t_eval=t_eval, method="RK45")
    return sol.t, sol.y

# === Collapse penalty ===
def collapse_cost(x_vals, y_vals, eps=0.1):
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if x < eps or y < eps:
            return (len(x_vals) - i) ** 2
    return 0

# === Combined cost: classical + quantum ===
def evaluate_cost(params):
    alpha, beta, gamma, delta = np.abs(params[:4])
    t, (x, y) = simulate_lotka_volterra((0, 25), [1, 1],
                                        [alpha, beta, gamma, delta],
                                        np.linspace(0, 25, 200))
    collapse = collapse_cost(x, y)

    qc = build_ansatz(params[4:6])
    estimator = BaseSamplerV1()
    job = estimator.run(circuits=[qc], observables=[observable])
    result = job.result()
    ev = result.values[0]
    quantum_term = abs(np.real(ev))

    return collapse / 100.0 + 0.1 * quantum_term

# === SPSA optimization ===
def hybrid_quantum_optimize(cost_fn, num_params=6):
    opt = SPSA(maxiter=20)
    x0 = np.random.uniform(0.1, 1.0, size=num_params)
    res = opt.minimize(fun=cost_fn, x0=x0)
    return res.x, res.fun

# === Run optimization ===
best_theta, best_cost = hybrid_quantum_optimize(evaluate_cost)
alpha, beta, gamma, delta = np.abs(best_theta[:4])
print(f"\n🔧 Best params: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, δ={delta:.4f}")
print(f"Optimized cost: {best_cost:.4f}")

# === Final simulation and collapse time ===
t, (x, y) = simulate_lotka_volterra((0, 25), [1, 1],
                                     [alpha, beta, gamma, delta],
                                     np.linspace(0, 25, 200))
epsilon = 0.1
collapse_time = next((tv for tv, xv, yv in zip(t, x, y) if xv < epsilon or yv < epsilon), t[-1])
print(f"⏳ Collapse at t = {collapse_time:.2f}")

# === Plot ===
plt.figure(figsize=(10, 5))
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
