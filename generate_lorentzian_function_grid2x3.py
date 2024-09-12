import numpy as np
import matplotlib.pyplot as plt

# Define the Lorentzian function
def lorentzian(x, gamma=1, T=36):
    return 1 / (1 + ((x - T/2)/gamma)**2)

# Generate data points for x
x = np.linspace(0, 36, 1000)

# Create a 3x2 plot grid
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Custom function to plot with professional appearance and filled area
def plot_lorentzian(ax, x, data, power, ylabel=False, xlabel=False, yticks=False, xticks=False):
    ax.plot(x, data, label=f'Lorentzian$^{power}$', linewidth=2)
    ax.fill_between(x, data, color='skyblue', alpha=0.4)  # Fill the area under the curve
    ax.legend(loc='upper right', fontsize=14)
    
    if xlabel:
        ax.set_xlabel('Time (a.u.)', fontsize=15)
    if ylabel:
        ax.set_ylabel('Rabi frequency (a.u.)', fontsize=15)
    
    # Customize ticks and their font size
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.tick_params(axis='x', which='major', labelsize=15)  # Set font size for x ticks
    ax.tick_params(axis='y', which='major', labelsize=15)  # Set font size for y ticks
    
    if not yticks:
        ax.set_yticklabels([])
    if not xticks:
        ax.set_xticklabels([])

# Plot Lorentzian^2
plot_lorentzian(axs[0, 0], x, lorentzian(x)**2, 2, ylabel=True, xlabel=False, yticks=True, xticks=False)

# Plot Lorentzian^(3/2)
plot_lorentzian(axs[0, 1], x, lorentzian(x)**(3/2), r'{3/2}', ylabel=False, xlabel=False, yticks=False, xticks=False)

# Plot Lorentzian
plot_lorentzian(axs[1, 0], x, lorentzian(x), 1, ylabel=True, xlabel=False, yticks=True, xticks=False)

# Plot Lorentzian^(3/4)
plot_lorentzian(axs[1, 1], x, lorentzian(x)**(3/4), r'{3/4}', ylabel=False, xlabel=False, yticks=False, xticks=False)

# Plot Lorentzian^(2/3)
plot_lorentzian(axs[2, 0], x, lorentzian(x)**(2/3), r'{2/3}', ylabel=True, xlabel=True, yticks=True, xticks=True)

# Plot Lorentzian^(3/5)
plot_lorentzian(axs[2, 1], x, lorentzian(x)**(3/5), r'{3/5}', ylabel=False, xlabel=True, yticks=False, xticks=True)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("C://Users/Ivo/Documents/PhD Documents/lorentzians_2x3.pdf",format="PDF")
