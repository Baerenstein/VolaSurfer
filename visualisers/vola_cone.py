import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define parameters for a circular base
radius = 5  # Increased radius for wider surface
height = 25  # Increased height to make it more cone-like
n_points = 50  # Number of points for the circular base

# Create points for full circle
theta = np.linspace(0, 2*np.pi, n_points)
z = np.linspace(0, height, n_points)
theta, z = np.meshgrid(theta, z)

# Calculate coordinates for a horizontal cone pointing right
r = np.maximum(0.1, (height - z) * (radius / height))  # Prevent radius from going to exactly zero
x = z * 3  # Stretch the cone horizontally by scaling z
y = -r * np.cos(theta)  # Flip the cone by negating y
z = r * np.sin(theta)

# Create the volatility surface
def create_vol_surface(x, y, z):
    # Adjusted formula to create more pronounced cone effect
    distance_from_axis = np.sqrt(y**2 + z**2) / radius
    time_factor = x/np.max(x)  # Normalize x to [0,1]
    return 0.15 + 0.2 * (1 - time_factor) + 0.1 * distance_from_axis

vol_surface = create_vol_surface(x, y, z)

# Create the 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the first surface (horizontal cone)
surf1 = ax.plot_surface(x, y, z,
                        facecolors=cm.viridis(vol_surface/vol_surface.max()),
                        alpha=0.6,
                        edgecolor='black',
                        linewidth=0.5)

# Add gridlines on the surface for the first cone
for i in range(0, n_points, 5):
    ax.plot(x[i, :], y[i, :], z[i, :], 'k-', alpha=0.2)
    ax.plot(x[:, i], y[:, i], z[:, i], 'k-', alpha=0.2)

# Define parameters for the second plot (cone of implied volatility)
strike_range = np.linspace(80, 120, 50)  # Focusing on a narrower strike range around ATM
maturity_range = np.linspace(0.1, 2, 50)  # Time to maturity in years
ATM_strike = 100

# Create meshgrid for the second plot
strikes, maturities = np.meshgrid(strike_range, maturity_range)

# Generate the surface for the second plot
iv_surface = create_vol_surface(strikes, maturities, ATM_strike)

# Plot the second surface (cone of implied volatility)
surf2 = ax.plot_surface(strikes, maturities, iv_surface,
                        cmap=cm.plasma,
                        alpha=0.5,
                        edgecolor='none')

# Customize the plot
ax.set_title("Combined Volatility Surfaces", fontsize=14, pad=20)
ax.set_xlabel("Time to Maturity", labelpad=10)
ax.set_ylabel("Strike Price (Width)", labelpad=10)
ax.set_zlabel("Strike Price (Height)", labelpad=10)

# Add colorbar for both surfaces
norm1 = plt.Normalize(vol_surface.min(), vol_surface.max())
sm1 = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm1)
fig.colorbar(sm1, ax=ax, shrink=0.5, aspect=5, label="Volatility Level (Cone)")

norm2 = plt.Normalize(iv_surface.min(), iv_surface.max())
sm2 = plt.cm.ScalarMappable(cmap=cm.plasma, norm=norm2)
fig.colorbar(sm2, ax=ax, shrink=0.5, aspect=5, label="Volatility Level (Implied)")

# Set the viewing angle
ax.view_init(elev=15, azim=-60)

# Adjust aspect ratio
ax.set_box_aspect([1.5, 1, 1])

# Remove background grid
ax.grid(False)

plt.show()