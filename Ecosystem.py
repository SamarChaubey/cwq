import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import SPSA


# 📌 Ensure reproducibility
np.random.seed(42)

def build_ansatz(params):
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    return qc

observable = SparsePauliOp.from_list([("ZZ", 1.0)])

def simulate_lv(params, x0, y0, steps=200):
    α, β, γ, δ = params
    def lv(t, z):
        x, y = z
        return [α * x - β * x * y, δ * x * y - γ * y]
    t = np.linspace(0, 25, steps)
    sol = solve_ivp(lv, (0, 25), (x0, y0), t_eval=t)
    return sol.t, sol.y

def collapse_cost(xs, ys, eps=0.1):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x < eps or y < eps:
            return (len(xs) - i) ** 2
    return 0


# Quantum primitive (Aer Estimator)
estimator = AerEstimator()

def evaluate_cost(params, x0, y0):
    # Classical collapse cost
    t, (x, y) = simulate_lv(np.abs(params[:4]), x0, y0)
    collapse = collapse_cost(x, y)

    # Quantum expectation ⟨ZZ⟩
    qc = build_ansatz(params[4:6])
    result = estimator.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[[]]
    ).result()
    ev = result.values[0]
    return collapse / 100 + 0.1 * abs(ev)

def optimize_lv(x0, y0, α, β, γ, δ):
    optimizer = SPSA(maxiter=30)
    x_init = np.array([α, β, γ, δ, 0.1, 0.1])
    return optimizer.minimize(fun=lambda p: evaluate_cost(p, x0, y0), x0=x_init)

if __name__ == "__main__":
    x0, y0, α, β, γ, δ = (float(input(prompt) or default)
                         for prompt, default in [
                             ("Initial prey x0: ", 1.0),
                             ("Initial predator y0: ", 1.0),
                             ("α (prey growth): ", 1.0),
                             ("β (predation): ", 0.5),
                             ("γ (predator death): ", 1.0),
                             ("δ (efficiency): ", 0.5),
                         ])
    print("Running hybrid quantum-classical optimization...")
    res = optimize_lv(x0, y0, α, β, γ, δ)

    θ = res.x
    α_opt, β_opt, γ_opt, δ_opt = np.abs(θ[:4])
    print(f"✅ Optimized: α={α_opt:.4f}, β={β_opt:.4f}, γ={γ_opt:.4f}, δ={δ_opt:.4f}")
    print(f"Cost: {res.fun:.4f} (calls: {res.nfev})")

    # Final collapse prediction
    t, (x, y) = simulate_lv([α_opt, β_opt, γ_opt, δ_opt], x0, y0)
    eps = 0.1
    collapse_t = next((tv for tv, xv, yv in zip(t, x, y) if xv < eps or yv < eps), t[-1])
    print(f"⏳ Collapse predicted at t = {collapse_t:.2f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(t, x, label="Prey", color="blue")
    plt.plot(t, y, label="Predator", color="red")
    plt.axhline(eps, linestyle="--", color="gray")
    plt.axvline(collapse_t, linestyle=":", color="purple", label=f"Collapse @ t={collapse_t:.2f}")
    plt.xlabel("Time"); plt.ylabel("Population")
    plt.title("Predator–Prey Dynamics — Optimized via QNSPSA")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()
