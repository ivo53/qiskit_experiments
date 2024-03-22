import matplotlib.pyplot as plt
import numpy as np
import matplotlib; matplotlib.use('Agg')

# Define the Lorentzian function
def lorentzian(x, x0, sigma):
    return 1 / (((x - x0) / sigma) ** 2 + 1)

# Define the truncated Lorentzian function
def truncated_lorentzian(x, x0, sigma, duration):
    return np.where(np.abs(x - x0) <= duration/2, lorentzian(x, x0, sigma), 0)

# Generate x values
x = np.linspace(-320, 320, 1000)

# Define the parameters for each subplot
parameters = [
    (0, 21.33, 42.666666666),
    (0, 21.33, 85.3333333333),
    (0, 21.33, 101.566982095),
    (0, 21.33, 149.84040893),
    (0, 21.33, 185.979688258),
    (0, 21.33, 242.612936664),
    (0, 21.33, 298.666666667),
    (0, 21.33, 424.527973164),
    (0, 21.33, 601.8874018)
]

# Create the grid of plots
fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex='col', sharey='row')

# Iterate over each subplot
for ax, params in zip(axes.flat, parameters):
    x0, sigma, duration = params
    y = truncated_lorentzian(x, x0, sigma, duration)
    ax.plot(x, y, color='black', linewidth=1.5)
    ax.axvline(-duration/2, color='r', linestyle='--', linewidth=1.5)
    ax.axvline(duration/2, color='r', linestyle='--', linewidth=1.5)
    # ax.set_title(f"duration={np.round(duration, 2)} ns")
    param_box_text = f'T={int(np.round(duration, 0))} ns'
    ax.text(
        0.58,
        0.95,
        param_box_text,
        fontsize=18,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

for ax in axes[-1]:
    ax.set_xlabel('Time [ns]', fontsize=18)
for ax in axes[:, 0]:
    ax.set_ylabel('Rabi Freq. [au]', fontsize=18)

# Adjust the spacing between subplots
plt.tight_layout()

plt.savefig(f"C:/Users/Ivo/Downloads/lorentzian_grid_{np.random.randint(1000)}.pdf", format="PDF")
# Display the grid of plots
# plt.show()
