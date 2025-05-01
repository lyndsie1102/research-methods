import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Dataset
code_scores = np.array([6.49, 7.32, 6.74, 2.01, 5.01, 6.15, 1.40, 2.38, 3.50, 5.22])
request_times = np.array([9.55, 34.74, 21.67, 7.41, 12.40, 20.66, 11.44, 13.75, 35.79, 14.56])

# Bayesian Linear Regression model
with pm.Model() as linear_model:
    α = pm.Normal("α", mu=0, sigma=10)  # Intercept
    β = pm.Normal("β", mu=0, sigma=1)   # Slope
    σ = pm.HalfNormal("σ", sigma=1)     # Error term
    μ = α + β * code_scores
    pm.Normal("obs", mu=μ, sigma=σ, observed=request_times)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

# Print summary statistics
print("Posterior Summary Statistics:")
print(az.summary(trace))

# Create prediction matrix
x_vals = np.linspace(min(code_scores), max(code_scores), 100)
posterior_pred = (
    trace.posterior['α'].values.reshape(-1, 1) + 
    trace.posterior['β'].values.reshape(-1, 1) * x_vals
)  # Shape: (8000, 100)

# Calculate HDI for each x value
hdi_vals = np.array([az.hdi(posterior_pred[:, i], hdi_prob=0.95) 
                  for i in range(len(x_vals))])


# Print key parameters
print("\nKey Parameters:")
print(f"Posterior mean intercept (α): {trace.posterior['α'].mean().item():.2f}")
print(f"Posterior mean slope (β): {trace.posterior['β'].mean().item():.2f}")
print(f"Posterior mean error term (σ): {trace.posterior['σ'].mean().item():.2f}")

# Plotting
plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(code_scores, request_times, c='blue', alpha=0.7, label='Observed Data')

# Regression line
posterior_mean = trace.posterior['α'].mean().item() + trace.posterior['β'].mean().item() * x_vals
plt.plot(x_vals, posterior_mean, 'r-', label='Posterior Mean')

# HDI bands
plt.fill_between(x_vals, hdi_vals[:, 0], hdi_vals[:, 1],
                color='gray', alpha=0.3, label='95% HDI')

# Formatting
plt.title("Bayesian Linear Regression: Code Quality vs. Generation Speed")
plt.xlabel("Code Score (Higher = Better Quality)")
plt.ylabel("Request Time (s) (Lower = Faster Generation)")
plt.grid(True, alpha=0.3)
plt.legend()

# Save plot
plt.savefig('quality_vs_speed_regression.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPlot saved as 'quality_vs_speed_regression.png'")