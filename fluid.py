import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from qiskit.primitives import Estimator
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA

# Load initial fields from wind_data.csv (no time column, just one field)
wind_data = pd.read_csv("wind_data.csv")
x_vals = sorted(wind_data["x"].unique())
y_vals = sorted(wind_data["y"].unique())

u_field = wind_data.pivot(index='y', columns='x', values='u').values
v_field = wind_data.pivot(index='y', columns='x', values='v').values
p_field = wind_data.pivot(index='y', columns='x', values='p').values
k_field = wind_data.pivot(index='y', columns='x', values='k').values
epsilon_field = wind_data.pivot(index='y', columns='x', values='epsilon').values

nu = 1.5e-5  # Kinematic viscosity of air (m^2/s)
dx = x_vals[1] - x_vals[0]

dy = y_vals[1] - y_vals[0]
X, Y = np.meshgrid(x_vals, y_vals)

# Quantum optimizer for C_mu using Qiskit primitives
class QuantumViscosityOptimizer:
    def __init__(self, estimator):
        self.estimator = estimator
        self.theta = Parameter('θ')
        self.ansatz = QuantumCircuit(1)
        self.ansatz.rx(self.theta, 0)
        self.observable = SparsePauliOp.from_list([("Z", 1.0)])
        self.optimizer = SPSA(maxiter=30)

    def quantum_cost(self, theta_value):
        result = self.estimator.run(
            circuits=[self.ansatz],
            observables=[self.observable],
            parameter_values=[[theta_value]]
        ).result()
        expectation = result.values[0]
        C_mu = 0.05 + 0.1 * ((theta_value / np.pi) % 1)
        cost = (C_mu - 0.09)**2 + 0.01 * C_mu + 0.1 * abs(expectation)
        return cost

    def run_vqe(self):
        def objective(params):
            return self.quantum_cost(params[0])
        initial_point = [np.pi / 2]
        result = self.optimizer.optimize(num_vars=1, objective_function=objective, initial_point=initial_point)
        optimal_theta = result[0][0]
        x_opt = (optimal_theta / np.pi) % 1
        C_mu_opt = 0.05 + 0.1 * x_opt
        return C_mu_opt

def compute_turbulent_viscosity(k, epsilon, quantum_optimizer=None):
    if quantum_optimizer:
        C_mu = quantum_optimizer.run_vqe()
        print(f"Quantum-optimized C_mu: {C_mu:.5f}")
    else:
        C_mu = 0.09
    return C_mu * k**2 / (epsilon + 1e-10)

def compute_rhs(u, v, p, nu_t, nu, dx, dy):
    Ny, Nx = u.shape
    rhs_u = np.zeros_like(u)
    rhs_v = np.zeros_like(v)
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            mu_eff = nu + nu_t[j, i]
            d2u_dx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2
            d2u_dy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2
            d2v_dx2 = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / dx**2
            d2v_dy2 = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / dy**2
            rhs_u[j, i] = mu_eff * (d2u_dx2 + d2u_dy2)
            rhs_v[j, i] = mu_eff * (d2v_dx2 + d2v_dy2)
    return rhs_u, rhs_v

def visualize_flow(X, Y, u, v, title="Velocity Field"):
    plt.figure(figsize=(7,7))
    speed = np.sqrt(u**2 + v**2)
    plt.quiver(X, Y, u, v, speed, cmap='viridis')
    plt.colorbar(label='Velocity magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Set up quantum optimizer (optional, can be toggled)
use_quantum = False  # Set to True to use quantum optimizer for C_mu
quantum_optimizer = QuantumViscosityOptimizer(Estimator()) if use_quantum else None

# Initial visualization
nu_t = compute_turbulent_viscosity(k_field, epsilon_field, quantum_optimizer)
rhs_u, rhs_v = compute_rhs(u_field, v_field, p_field, nu_t, nu, dx, dy)
visualize_flow(X, Y, u_field, v_field, title="Initial Velocity Field")

dt = 1.0
num_steps = 12

u = u_field.copy()
v = v_field.copy()
p = p_field.copy()
k = k_field.copy()
epsilon = epsilon_field.copy()

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for step in range(num_steps):
    nu_t = compute_turbulent_viscosity(k, epsilon, quantum_optimizer)
    rhs_u, rhs_v = compute_rhs(u, v, p, nu_t, nu, dx, dy)

    u += dt * rhs_u
    v += dt * rhs_v    

    ax.clear()
    magnitude = np.sqrt(u**2 + v**2)
    q = ax.quiver(X, Y, u, v, magnitude, angles='xy', scale_units='xy', scale=50, cmap='viridis')
    cb = fig.colorbar(q, ax=ax, label='Velocity Magnitude')
    ax.set_title(f"Velocity Field at step {step+1}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_aspect('equal')
    display.clear_output(wait=True)
    display.display(fig)
    plt.pause(0.5)
    cb.remove()  # Remove colorbar before next frame to avoid stacking


plt.ioff()
plt.show()

# Calculate initial and final velocity magnitude
initial_magnitude = np.sqrt(u_field**2 + v_field**2)
final_magnitude = np.sqrt(u**2 + v**2)
change = final_magnitude - initial_magnitude

plt.figure(figsize=(8, 6))
im = plt.imshow(change, origin='lower', cmap='bwr', aspect='auto')
plt.colorbar(im, label='Change in Velocity Magnitude')
plt.title("Change in Velocity Magnitude (Final - Initial)")
plt.xlabel('X grid index')
plt.ylabel('Y grid index')
plt.grid(False)
plt.show()
