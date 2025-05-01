import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Dataset
request_times = np.array([9.55,	21.67, 34.74, 7.41,	12.4,	
                          20.66, 11.44,	13.75, 35.79, 14.56]) 
pass_rates = np.array([76.32, 85.76, 88.47, 12.41, 39.37, 79.12,
                       31.56, 28.46, 44.49, 41.55])

# Binary outcome and standardized predictors
pass_binary = np.array([1 if rate >= 80 else 0 for rate in pass_rates])
request_times_std = (request_times - request_times.mean()) / request_times.std()

# Bayesian Logistic Regression model
with pm.Model() as logistic_model:
    α = pm.Normal("α", mu=0, sigma=5)
    β = pm.Normal("β", mu=0, sigma=5)
    p = pm.math.sigmoid(α + β * request_times_std)
    pm.Bernoulli("obs", p, observed=pass_binary)
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=42)

# ========================
# 1. Print Key Statistics
# ========================
print("\n=== Bayesian Logistic Regression Results ===")
print(az.summary(trace, hdi_prob=0.95, var_names=["α", "β"]))

# Calculate probability that β < 0
β_samples = trace.posterior["β"].values.flatten()
prob_β_neg = (β_samples < 0).mean()
print(f"\nP(β < 0): {prob_β_neg:.3f}")

# ========================
# 2. Save Diagnostic Plots
# ========================
# Plot 1: Posterior distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
az.plot_posterior(trace, var_names=["α"], ax=ax[0])
az.plot_posterior(trace, var_names=["β"], ax=ax[1], ref_val=0)
fig.suptitle("Posterior Distributions of Parameters")
plt.tight_layout()
plt.savefig("logistic_posteriors.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Logistic curve
x_vals = np.linspace(-2, 2, 100)  # Coverage for standardized values
p_mean = 1 / (1 + np.exp(-(trace.posterior["α"].mean().item() + 
                          trace.posterior["β"].mean().item() * x_vals)))

plt.figure(figsize=(10, 6))
plt.scatter(request_times_std, pass_binary, alpha=0.7, 
           label=f"Observed (Mean RT: {request_times.mean():.1f}s)")
plt.plot(x_vals, p_mean, 'r-', label='Posterior Mean Probability')
plt.xlabel("Standardized Request Time (0 = mean)")
plt.ylabel("P(Pass Rate ≥ 80%)")
plt.title("Logistic Regression: Request Time vs. Pass Probability")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("logistic_curve.png", dpi=300, bbox_inches='tight')
plt.close()
