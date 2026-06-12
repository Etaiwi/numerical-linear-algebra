import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Part I - Preliminaries

# Section (a)

def build_grid_points(lam: float, alpha: float) -> np.ndarray:
    """Build particle coordinates in the square [-W/2, W/2]^2.
    Returns:
        points: An array of shape (N, 2) containing the coordinates of the particles,
        while N is the amount of points in the grid.
    """
    W = alpha * lam # region size
    delta = lam / 10 # spacing between particles

    n_points = int(round(W / delta)) + 1 # number of points along each axis
    coords = np.linspace(-W / 2, W / 2, n_points) # coordinates along each axis

    x, y = np.meshgrid(coords, coords) # create a grid of points

    points = np.column_stack((x.ravel(), y.ravel())) # combine x and y into a single array of shape (N, 2)
    return points


def plot_grid(points: np.ndarray, title: str = "Particle Grid",
              savefig: bool = False, filename: str = "grid.png", show: bool = True) -> None:
    """Plot the particle grid."""
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved figure to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def select_square_points(points: np.ndarray, center: np.ndarray, w: float) -> np.ndarray:
    """Select particles inside a square of side length w centered at center."""
    x = points[:, 0]
    y = points[:, 1]

    xc, yc = center
    tol = 1e-10 # small tolerance to include points on the boundary

    # Create a boolean mask to select points within the square
    mask = (
        (np.abs(x - xc) <= w / 2 + tol) &
        (np.abs(y - yc) <= w / 2 + tol)
    )

    return points[mask]


def kernel_a(r_m: np.ndarray, r_n: np.ndarray) -> float:
    """Non-oscillatory kernel from Part I(a)."""
    dist = np.linalg.norm(r_m - r_n)

    # Handle the singularity at zero distance
    if dist < 1e-10:
        return 0.5
    
    # Otherwise, return the kernel value
    return float(1.0 / np.sqrt(dist))


def corner_centers(W: float, w: float):
    """Return corner centers for source/observer squares given W and w."""
    c = W / 2 - w / 2

    bottom_left = np.array([-c, -c])
    top_right = np.array([c, c])
    top_left = np.array([-c, c])
    bottom_right = np.array([c, -c])

    return bottom_left, top_right, top_left, bottom_right


def constructing_A_os(source_points: np.ndarray, observer_points: np.ndarray, kernel):
    """Construct the matrix A_os."""
    A_os = np.array([[kernel(r_o, r_s) for r_s in source_points] for r_o in observer_points])
    return A_os


def plot_matrix(A: np.ndarray, title: str = "A_os",
                savefig: bool = False, filename: str = "Matrix.png", show: bool = True) -> None:
    """Plot a matrix with colorbar for interaction strength."""
    plt.figure(figsize=(6, 5))
    plt.imshow(A, aspect="auto")
    plt.colorbar(label="Interaction strength")
    plt.xlabel("Source particle index")
    plt.ylabel("Observer particle index")
    plt.title(title)

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_singular_values(s: np.ndarray, semilog: bool = True, title: str = "Singular Values of the Matrix",
                         savefig: bool = False, filename: str = "singular_values.png", show: bool = True) -> None:
    """Plot singular values, optionally using semilog scale."""
    fig, ax = plt.subplots(figsize=(6, 5))

    idx = np.arange(1, len(s) + 1)

    if semilog:
        ax.semilogy(idx, s, marker="o", linestyle="None")
    else:
        ax.plot(idx, s, marker="o", linestyle="None")

    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    ax.grid(True)

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

lam = 1.0
alpha = 4
W = alpha * lam
w = W / 8

bottom_left, top_right, top_left, bottom_right = corner_centers(W=W, w=w)

r_cs = bottom_left
r_co = top_right

# Build the full grid and select source/observer points
points = build_grid_points(lam=lam, alpha=alpha)
source_points = select_square_points(points=points, center=r_cs, w=w)
observer_points = select_square_points(points=points, center=r_co, w=w)

# Print the shape of the points arrays
print("Shape of points:", points.shape)
print("Shape of source_points:", source_points.shape)
print("Shape of observer_points:", observer_points.shape)

# Plot the grid
plot_grid(points=points, title="Full Grid", savefig=True, filename="Part I/Section a/full_grid.png", show=False)
plot_grid(points=source_points, title="Source Points",
          savefig=True, filename="Part I/Section a/source_points.png", show=False)
plot_grid(points=observer_points, title="Observer Points",
          savefig=True, filename="Part I/Section a/observer_points.png", show=False)

# Creating A_os
A_os = constructing_A_os(source_points=source_points, observer_points=observer_points, kernel=kernel_a)

# Print shape of A_os
print("Shape of A_os:", A_os.shape)

# Plot A_os
plot_matrix(A_os, title="Interaction Block A_os", savefig=True, filename="Part I/Section a/A_os.png", show=False)

# Computing the SVD of A_os
U, s, Vh = np.linalg.svd(A_os, full_matrices=False)

# Plot A_os SVD curve
plot_singular_values(s=s, semilog=True, title="Singular Values of A_os",
                     savefig=True, filename="Part I/Section a/singular_values_A_os.png", show=False)

# Section (b)
