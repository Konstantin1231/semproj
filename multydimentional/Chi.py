import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Set the degrees of freedom
df = 200

# Generate a range of x-values for the plot
x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)

# Plot the probability density function (PDF)
plt.plot(x, chi2.pdf(x, df), 'r-', lw=5, alpha=0.6, label='chi2 pdf')

# Add a title and labels
plt.title("Chi-Squared Distribution (df = 200)")
plt.xlabel("x")
plt.ylabel("Probability Density")

# Show the plot
plt.show()