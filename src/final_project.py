import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Part I - Preliminaries

run_section_a = False
run_section_b = False
run_section_c = False

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
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_source_observer_grid(points, source_points, observer_points, title="Source and Observer Regions",
                              savefig=False, filename="source_observer_grid.png", show=True):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.25, label="Full grid")
    ax.scatter(source_points[:, 0], source_points[:, 1], s=25, label="Source points")
    ax.scatter(observer_points[:, 0], observer_points[:, 1], s=25, label="Observer points")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

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

def plot_matrix(A: np.ndarray, title: str = "$A^os$",
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
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

if run_section_a:
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

    plot_source_observer_grid(points=points, source_points=source_points, observer_points=observer_points, 
                            title="Full Grid with Source and Observer Points", savefig=True, filename="Part I/Section a/source_observer_grid.png", show=False)

    # Creating A_os
    A_os = constructing_A_os(source_points=source_points, observer_points=observer_points, kernel=kernel_a)

    # Print shape of A_os
    print("Shape of A_os:", A_os.shape)

    # Plot A_os
    plot_matrix(A_os, title="Interaction Block $A^{os}$", savefig=True, filename="Part I/Section a/A_os.png", show=False)

    # Computing the SVD of A_os
    _, s, _ = np.linalg.svd(A_os, full_matrices=False)

    # Plot A_os SVD curve
    plot_singular_values(s=s, semilog=True, title="Singular values of $A^{os}$",
                        savefig=True, filename="Part I/Section a/singular_values_A_os.png", show=False)

# Section (b)
def svd_numerical_rank(s: np.ndarray, tau: float) -> int:
    """Compute numerical rank using the relative spectral norm.
    Rank r is chosen using the 2-norm, which means:
        sigma_{r+1} / sigma_1 <= tau

    Equivalently, r is chosen so that:
        sigman_i / sigma_1 > tau
    """
    if len(s) == 0:
        return 0
    
    return int(np.sum(s / s[0] > tau))

def run_single_case(lam: float, alpha: float, kernel, tau_values):
    """Run one section (b) case and return data needed for plots."""
    W = alpha * lam # W and w change each case
    w = W / 8

    # Same as in section (a), choosing bottom left and top right
    bottom_left, top_right, top_left, bottom_right = corner_centers(W=W, w=w)
    r_cs = bottom_left
    r_co = top_right

    # Build the full grid and select source/observer points for a single case
    points = build_grid_points(lam=lam, alpha=alpha)
    source_points = select_square_points(points=points, center=r_cs, w=w)
    observer_points = select_square_points(points=points, center=r_co, w=w)

    # Creating A_os
    A_os = constructing_A_os(source_points=source_points, observer_points=observer_points, kernel=kernel)

    # Computing the SVD of A_os and time it
    start = time.perf_counter()
    _, s, _ = np.linalg.svd(A_os, full_matrices=False)
    end = time.perf_counter()
    svd_time = end - start

    ranks = {tau: svd_numerical_rank(s=s, tau=tau) for tau in tau_values}

    return {
        "alpha": alpha,
        "W": W,
        "w": w,
        "A_shape": A_os.shape,
        "singular values": s,
        "ranks": ranks,
        "svd_time": svd_time
    }

def plot_singular_values_sequence(results, normalized: bool = False,
                                  savefig: bool = False, filename: str = "singular_values_sequence.png", show: bool = True) -> None:
    """Plot singular value curves."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for result in results:
        # Getting the singular values and the corresponding W
        s = result["singular values"]
        W = result["W"]

        # Normalizing to show the relative strength compared to sigma_1
        values = s / s[0] if normalized else s

        idx = np.arange(1, len(values) + 1)

        # Adding to the fig
        ax.scatter(idx, values, s=12, label=f"W={W:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Relative singular value" if normalized else "Singular value")
    ax.set_title("Relative singular values curves for different W" if normalized else "Singular values curves for different W")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend()

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Save figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_ranks_vs_W(results, tau_values,
                    savefig: bool = False, filename: str = "ranks_vs_W.png", show: bool = True) -> None:
    """Plot numerical rank versus W for several tau values."""
    fig, ax = plt.subplots(figsize=(6, 5))

    W_values = np.array([result["W"] for result in results])

    for tau in tau_values:
        ranks = np.array([result["ranks"][tau] for result in results])
        ax.scatter(W_values, ranks, label = f"$\\tau={tau:g}$")

    ax.set_xlabel("W")
    ax.set_ylabel("Numerical rank")
    ax.set_title("Numerical Rank vs W")
    ax.grid(True)
    ax.set_xscale("log", base=2)
    ax.legend()

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_svd_time_vs_W(results, savefig: bool = False, filename: str = "svd_time_vs_W.png", show: bool = True) -> None:
    """Plot the svd computation time for different W."""
    fig, ax = plt.subplots(figsize=(6,5))

    W_values = np.array([result["W"] for result in results])
    svd_times = np.array([result["svd_time"] for result in results])

    ax.scatter(W_values, svd_times)

    ax.set_xlabel("W")
    ax.set_ylabel("SVD computation time [sec]")
    ax.set_title("SVD computation time vs W")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True)

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

if run_section_b:
    lam = 1.0
    tau_values = [1e-2, 1e-5, 1e-8]
    alpha_values = [4, 8, 16, 32, 64] # doubling W by doubling alpha
    results = []

    for alpha in alpha_values:
        result = run_single_case(lam=lam, alpha=alpha, kernel=kernel_a, tau_values=tau_values)
        results.append(result)

        print(
            f"alpha={alpha}, W={result['W']}, "
            f"A_os shape={result['A_shape']}, "
            f"time={result['svd_time']:.4f} sec, "
            f"ranks={result['ranks']}"
        )

    # Plotting the singular values curves for every W
    plot_singular_values_sequence(results=results, normalized=False,
        savefig=True, filename="Part I/Section b/singular_values_sequence.png", show=False)

    plot_singular_values_sequence(results=results, normalized=True,
        savefig=True, filename="Part I/Section b/relative_singular_values_sequence.png", show=False)

    # Plotting the ranks vs W for the different tau values
    plot_ranks_vs_W(results, tau_values=tau_values,
        savefig=True, filename="Part I/Section b/ranks_vs_W.png", show=False)

    # Plotting the SVD computation time vs W
    plot_svd_time_vs_W(results, savefig=True, filename="Part I/Section b/svd_time_vs_W.png", show=False)

# Section (c)
def kernel_c(r_m: np.ndarray, r_n: np.ndarray, lam: float) -> complex:
    """Oscillatory kernel from Part I(c)."""
    dist = np.linalg.norm(r_m - r_n)

    if dist < 1e-10:
        return 0.5

    k = 2 * np.pi / lam
    return np.exp(-1j * k * dist) / np.sqrt(dist)

if run_section_c:
    lam = 1.0
    tau_values = [1e-2, 1e-5, 1e-8]
    alpha_values = [4, 8, 16, 32, 64] # doubling W by doubling alpha
    kernel_c_lam = lambda r_m, r_n: kernel_c(r_m, r_n, lam) # lambda function to insert lam into kernel_c
    results = []

    for alpha in alpha_values:
        result = run_single_case(lam=lam, alpha=alpha, kernel=kernel_c_lam, tau_values=tau_values)
        results.append(result)

        print(
            f"alpha={alpha}, W={result['W']}, "
            f"A_os shape={result['A_shape']}, "
            f"time={result['svd_time']:.4f} sec, "
            f"ranks={result['ranks']}"
        )

    # Plotting the singular values curves for every W
    plot_singular_values_sequence(results=results, normalized=False,
        savefig=True, filename="Part I/Section c/singular_values_sequence.png", show=False)

    plot_singular_values_sequence(results=results, normalized=True,
        savefig=True, filename="Part I/Section c/relative_singular_values_sequence.png", show=False)

    # Plotting the ranks vs W for the different tau values
    plot_ranks_vs_W(results, tau_values=tau_values,
        savefig=True, filename="Part I/Section c/ranks_vs_W.png", show=False)

    # Plotting the SVD computation time vs W
    plot_svd_time_vs_W(results, savefig=True, filename="Part I/Section c/svd_time_vs_W.png", show=False)

# Part II - Fast Rank Estimation

run_section_f = False
run_section_h = False

# Section (f)
def fast_rank_estimation(A_os: np.ndarray, tau: float, tau_R: float) -> int:
    """Estimate the numerical rank of A_os using the proposed randomized algorithm."""
    l = 1
    n = 10
    N = A_os.shape[0]
    prev_rank = None

    while True:
        n = min(n, N) # make sure n is not bigger than A_os

        row_idx = np.random.choice(N, size=n, replace=False) # row index vector
        col_idx = np.random.choice(N, size=n, replace=False) # col index vector

        A_l_os = A_os[row_idx][:, col_idx] # submatrix A_l_os

        _, s, _ = np.linalg.svd(A_l_os, full_matrices=False)
        rank = svd_numerical_rank(s=s, tau=tau)

        if l != 1 and prev_rank is not None:
            eps_R = (rank - prev_rank) / rank
            if eps_R < tau_R:
                return rank

        if n == N:
            return rank

        prev_rank = rank
        n = 2 * n
        l += 1

def run_fast_rank_case(lam: float, alpha: float, kernel, tau_values, tau_R: float):
    """Run one fast rank-estimation case and return data needed for plots."""
    W = alpha * lam
    w = W / 8

    bottom_left, top_right, top_left, bottom_right = corner_centers(W=W, w=w)
    r_cs = bottom_left
    r_co = top_right

    points = build_grid_points(lam=lam, alpha=alpha)
    source_points = select_square_points(points=points, center=r_cs, w=w)
    observer_points = select_square_points(points=points, center=r_co, w=w)

    A_os = constructing_A_os(
        source_points=source_points,
        observer_points=observer_points,
        kernel=kernel
    )

    start = time.perf_counter()
    ranks = {tau: fast_rank_estimation(A_os=A_os, tau=tau, tau_R=tau_R) for tau in tau_values}
    fast_time = time.perf_counter() - start

    return {
        "alpha": alpha,
        "W": W,
        "w": w,
        "A_shape": A_os.shape,
        "ranks": ranks,
        "fast_time": fast_time
    }

def plot_rank_comparison_vs_W(svd_results, fast_results, tau_values,
                              savefig: bool = False, filename: str = "rank_comparison_vs_W.png",
                              show: bool = True) -> None:
    """Plot full-SVD ranks and fast-estimated ranks versus W."""
    fig, ax = plt.subplots(figsize=(6, 5))

    W_values = np.array([result["W"] for result in svd_results])

    for tau in tau_values:
        svd_ranks = np.array([result["ranks"][tau] for result in svd_results])
        fast_ranks = np.array([result["ranks"][tau] for result in fast_results])

        ax.scatter(W_values, svd_ranks, marker="o", label=f"SVD, $\\tau={tau:g}$")
        ax.scatter(W_values, fast_ranks, marker="x", label=f"Fast, $\\tau={tau:g}$")

    ax.set_xlabel("W")
    ax.set_ylabel("Numerical rank")
    ax.set_title("Full SVD Rank vs Fast Estimated Rank")
    ax.set_xscale("log", base=2)
    ax.grid(True)
    ax.legend()

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def fast_rank_estimation_from_points(source_points: np.ndarray, observer_points: np.ndarray,
                                     kernel, tau: float, tau_R: float) -> int:
    """Estimate the numerical rank using sampled source and observer points."""
    l = 1
    n = 10
    N = observer_points.shape[0]
    prev_rank = None

    while True:
        n = min(n, N)

        row_idx = np.random.choice(N, size=n, replace=False)
        col_idx = np.random.choice(N, size=n, replace=False)

        sampled_observer_points = observer_points[row_idx]
        sampled_source_points = source_points[col_idx]

        A_l_os = constructing_A_os(
            source_points=sampled_source_points,
            observer_points=sampled_observer_points,
            kernel=kernel
        )

        _, s, _ = np.linalg.svd(A_l_os, full_matrices=False)
        rank = svd_numerical_rank(s=s, tau=tau)

        if l != 1 and prev_rank is not None:
            eps_R = (rank - prev_rank) / rank
            if eps_R < tau_R:
                return rank

        if n == N:
            return rank

        prev_rank = rank
        n = 2 * n
        l += 1

def run_fast_rank_case_from_points(lam: float, alpha: float, kernel, tau_values, tau_R: float):
    """Run one fast rank-estimation case without constructing the full A_os."""
    W = alpha * lam
    w = W / 8

    bottom_left, top_right, top_left, bottom_right = corner_centers(W=W, w=w)
    r_cs = bottom_left
    r_co = top_right

    points = build_grid_points(lam=lam, alpha=alpha)
    source_points = select_square_points(points=points, center=r_cs, w=w)
    observer_points = select_square_points(points=points, center=r_co, w=w)

    start = time.perf_counter()
    ranks = {
        tau: fast_rank_estimation_from_points(
            source_points=source_points,
            observer_points=observer_points,
            kernel=kernel,
            tau=tau,
            tau_R=tau_R
        )
        for tau in tau_values
    }
    fast_time = time.perf_counter() - start

    return {
        "alpha": alpha,
        "W": W,
        "w": w,
        "A_shape": (observer_points.shape[0], source_points.shape[0]),
        "ranks": ranks,
        "fast_time": fast_time
    }

def estimate_rank_scaling(results, tau_values, W_min=None):
    """Estimate rank scaling R(W) ~ C W^p using a log-log fit."""
    scaling_results = {}

    for tau in tau_values:
        W_values = np.array([result["W"] for result in results])
        ranks = np.array([result["ranks"][tau] for result in results])

        mask = ranks > 0
        if W_min is not None:
            mask = mask & (W_values >= W_min)

        W_fit = W_values[mask]
        ranks_fit = ranks[mask]

        p, log_C = np.polyfit(np.log(W_fit), np.log(ranks_fit), 1)
        C = np.exp(log_C)

        scaling_results[tau] = {
            "p": p,
            "C": C
        }

        print(f"tau={tau:g}: R(W) ~ {C:.4g} * W^{p:.4f}")

    return scaling_results

def save_scaling_results(scaling_results, filename: str = "rank_scaling_results.txt") -> None:
    """Save asymptotic rank scaling results to a text file."""
    out_path = Path("results") / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("Asymptotic rank scaling results\n")
        f.write("Model: R(W) ~ C * W^p\n\n")

        for tau, result in scaling_results.items():
            f.write(
                f"tau={tau:g}: "
                f"C={result['C']:.6g}, "
                f"p={result['p']:.6f}, "
                f"R(W) ~ {result['C']:.6g} * W^{result['p']:.6f}\n"
            )

    print(f"Saved scaling results to {out_path}")

if run_section_f:
    np.random.seed(0)

    # Comparing between I(c) to II(f)
    lam = 1.0
    tau_values = [1e-2, 1e-5, 1e-8]
    tau_R = 0.05
    alpha_values = [4, 8, 16, 32, 64]  # Values where full SVD is still feasible

    kernel_c_lam = lambda r_m, r_n: kernel_c(r_m, r_n, lam)

    svd_results = []
    fast_results = []

    for alpha in alpha_values:
        svd_result = run_single_case(
            lam=lam,
            alpha=alpha,
            kernel=kernel_c_lam,
            tau_values=tau_values
        )

        fast_result = run_fast_rank_case(
            lam=lam,
            alpha=alpha,
            kernel=kernel_c_lam,
            tau_values=tau_values,
            tau_R=tau_R
        )

        svd_results.append(svd_result)
        fast_results.append(fast_result)

        print(
            f"alpha={alpha}, W={fast_result['W']}, "
            f"A_os shape={fast_result['A_shape']}, "
            f"SVD ranks={svd_result['ranks']}, "
            f"Fast ranks={fast_result['ranks']}, "
            f"SVD time={svd_result['svd_time']:.4f} sec, "
            f"Fast time={fast_result['fast_time']:.4f} sec"
        )

    plot_rank_comparison_vs_W(
        svd_results=svd_results,
        fast_results=fast_results,
        tau_values=tau_values,
        savefig=True,
        filename="Part II/Section f/rank_comparison_vs_W.png",
        show=False
    )

    ## Going for higher W
    large_alpha_values = [128, 256, 512, 1024]
    large_fast_results = []

    for alpha in large_alpha_values:
        fast_result = run_fast_rank_case_from_points(
            lam=lam,
            alpha=alpha,
            kernel=kernel_c_lam,
            tau_values=tau_values,
            tau_R=tau_R
        )

        large_fast_results.append(fast_result)

        print(
            f"alpha={alpha}, W={fast_result['W']}, "
            f"Equivalent A_os shape={fast_result['A_shape']}, "
            f"Fast ranks={fast_result['ranks']}, "
            f"Fast time={fast_result['fast_time']:.4f} sec"
        )

    all_fast_results = fast_results + large_fast_results
    plot_ranks_vs_W(
        results=all_fast_results,
        tau_values=tau_values,
        savefig=True,
        filename="Part II/Section f/fast_ranks_extended_vs_W.png",
        show=False
    )

    # Find asymptotic scaling
    scaling_results = estimate_rank_scaling(
    results=all_fast_results,
    tau_values=tau_values,
    W_min=64
    )

    save_scaling_results(
    scaling_results=scaling_results,
    filename="Part II/Section f/rank_scaling_results.txt"
    )

# Section (h)
def run_time_case(lam: float, alpha: float, kernel, tau_values, tau_R: float, num_runs: int = 3):
    """Measure total fast rank-estimation time for all tau values for one W."""
    W = alpha * lam
    w = W / 8

    bottom_left, top_right, top_left, bottom_right = corner_centers(W=W, w=w)
    r_cs = bottom_left
    r_co = top_right

    points = build_grid_points(lam=lam, alpha=alpha)
    source_points = select_square_points(points=points, center=r_cs, w=w)
    observer_points = select_square_points(points=points, center=r_co, w=w)

    times = []
    ranks_list = []

    for run in range(num_runs):
        np.random.seed(run)

        start = time.perf_counter()
        ranks = {
            tau: fast_rank_estimation_from_points(
                source_points=source_points,
                observer_points=observer_points,
                kernel=kernel,
                tau=tau,
                tau_R=tau_R
            )
            for tau in tau_values
        }
        fast_time = time.perf_counter() - start

        times.append(fast_time)
        ranks_list.append(ranks)

    return {
        "alpha": alpha,
        "W": W,
        "A_shape": (observer_points.shape[0], source_points.shape[0]),
        "median_time": float(np.median(times)),
        "times": times,
        "ranks_list": ranks_list
    }

def plot_fast_time_vs_W(results, savefig: bool = False, filename: str = "fast_time_vs_W.png", show: bool = True) -> None:
    """Plot median fast rank-estimation time versus W."""
    fig, ax = plt.subplots(figsize=(6, 5))

    W_values = np.array([result["W"] for result in results])
    times = np.array([result["median_time"] for result in results])

    ax.scatter(W_values, times)

    ax.set_xlabel("W")
    ax.set_ylabel("Median fast rank-estimation time [sec]")
    ax.set_title("Fast Rank-Estimation Time vs W")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True)

    if savefig:
        out_path = Path("images") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def estimate_time_scaling(results, W_min=None):
    """Estimate T(W) ~ C W^p."""
    W_values = np.array([result["W"] for result in results])
    times = np.array([result["median_time"] for result in results])

    mask = times > 0
    if W_min is not None:
        mask = mask & (W_values >= W_min)

    p, log_C = np.polyfit(np.log(W_values[mask]), np.log(times[mask]), 1)
    C = np.exp(log_C)

    print(f"T(W) ~ {C:.4g} * W^{p:.4f}")

    return {
        "C": C,
        "p": p
    }

def save_time_scaling_results(result, filename: str = "time_scaling_results.txt") -> None:
    """Save time scaling result to a text file."""
    out_path = Path("results") / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("Asymptotic time scaling results\n")
        f.write("Model: T(W) ~ C * W^p\n")
        f.write("Timing value: median over repeated runs\n")
        f.write("Each timing measures total time for all tau values\n\n")
        f.write(
            f"C={result['C']:.6g}, "
            f"p={result['p']:.6f}, "
            f"T(W) ~ {result['C']:.6g} * W^{result['p']:.6f}\n"
        )

    print(f"Saved time scaling results to {out_path}")

if run_section_h:
    np.random.seed(0)

    lam = 1.0
    tau_values = [1e-2, 1e-5, 1e-8]
    tau_R = 0.05
    num_runs = 5

    alpha_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    kernel_c_lam = lambda r_m, r_n: kernel_c(r_m, r_n, lam)

    time_results = []

    for alpha in alpha_values:
        result = run_time_case(
            lam=lam,
            alpha=alpha,
            kernel=kernel_c_lam,
            tau_values=tau_values,
            tau_R=tau_R,
            num_runs=num_runs
        )

        time_results.append(result)

        print(
            f"alpha={alpha}, W={result['W']}, "
            f"Equivalent A_os shape={result['A_shape']}, "
            f"median time={result['median_time']:.4f} sec, "
            f"times={np.round(result['times'], 4)}"
        )

    plot_fast_time_vs_W(
        results=time_results,
        savefig=True,
        filename="Part II/Section h/fast_time_vs_W.png",
        show=False
    )

    time_scaling_result = estimate_time_scaling(
        results=time_results,
        W_min=64
    )

    save_time_scaling_results(
        result=time_scaling_result,
        filename="Part II/Section h/time_scaling_results.txt"
    )

# Part III - Fast LR Estimation

run_section_j = False

def informed_blocked_randomized_lr(A: np.ndarray, tau: float, tau_R: float, gamma: int, N_b: int, seed=None):
    """Informed Blocked Randomized LR approximation."""
    if seed is not None:
        np.random.seed(seed)

    start_time = time.perf_counter()

    # Step 1+2: Initialization and estimate rank
    A_res = A.copy()
    N = A.shape[0]
    A_norm = np.linalg.norm(A, ord="fro")

    # Estimate rank using Part II
    R_l = fast_rank_estimation(A_os=A, tau=tau, tau_R=tau_R)

    # Block size
    B0 = int(np.ceil(R_l / N_b))
    B0 = max(B0, 1)

    U_blocks = []
    B_blocks = []
    block_norms = []
    residual_errors = []

    max_blocks = int(np.ceil(N / B0))

    for _ in range(max_blocks):
        current_rank = sum(block.shape[1] for block in U_blocks)
        block_size = min(B0, N - current_rank)

        if block_size <= 0:
            break

        # Step 3: Generate Complex Gaussian matrix
        G = (
            np.random.randn(N, block_size + gamma)
            + 1j * np.random.randn(N, block_size + gamma)
        ) / np.sqrt(2)

        # Step 4: Sample residual
        M = A_res @ G

        # Step 5: SVD of sampled matrix
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        U_block = U[:, :block_size]

        # Step 6: Compute coefficient block
        B = U_block.conj().T @ A_res

        # Step 7: Add the current block
        U_blocks.append(U_block)
        B_blocks.append(B)

        # Step 8: Compute block norm and residual error
        block_norm = np.linalg.norm(B, ord="fro")
        block_norms.append(block_norm / A_norm)

        # Step 9: Update residual
        A_res = A_res - U_block @ B
        residual_error = np.linalg.norm(A_res, ord="fro") / A_norm
        residual_errors.append(residual_error)

        # Stopping criterion of step 8
        if block_norm <= tau * A_norm:
            break
    
    U_hat = np.hstack(U_blocks)
    B_hat = np.vstack(B_blocks)

    return {
        "U_hat": U_hat,
        "B_hat": B_hat,
        "R_l": R_l,
        "B0": B0,
        "rank_lr": U_hat.shape[1],
        "num_blocks": len(U_blocks),
        "block_norms": block_norms,
        "residual_errors": residual_errors,
        "lr_time": time.perf_counter() - start_time,
        "final_error": residual_errors[-1]
    }

def best_svd_rank_error(s: np.ndarray, r: int) -> float:
    """Relative Frobenius error of the best rank-r SVD approximation."""
    if r >= len(s):
        return 0.0

    return float(np.linalg.norm(s[r:]) / np.linalg.norm(s))

def build_A_os_for_alpha(alpha: int, lam: float, kernel):
    """Build A_os using the same geometry as Section (c)."""
    W = alpha * lam
    w = W / 8

    bottom_left, top_right, _, _ = corner_centers(W=W, w=w)
    r_cs = bottom_left
    r_co = top_right

    points = build_grid_points(lam=lam, alpha=alpha)
    source_points = select_square_points(points=points, center=r_cs, w=w)
    observer_points = select_square_points(points=points, center=r_co, w=w)

    A_os = constructing_A_os(
        source_points=source_points,
        observer_points=observer_points,
        kernel=kernel
    )

    return W, A_os

if run_section_j:
    np.random.seed(0) # For reproducibility

    # Parameters for the experiments
    lam = 1.0
    tau_values = [1e-2, 1e-5, 1e-8]
    tau_R = 0.05
    seed = 0

    alpha_values = [4, 8, 16, 32, 64]  # Values where full SVD is still feasible
    kernel_c_lam = lambda r_m, r_n: kernel_c(r_m, r_n, lam)

    # Parameter calibration
    calibration_alpha = 64
    calibration_tau = 1e-5
    parameter_grid = [(2, 2), (5, 1), (5, 2), (10, 2), (5, 4)] # (gamma, N_b) pairs

    # Selected parameters for final comparison
    selected_gamma = 5
    selected_N_b = 2

    calibration_results = []
    final_results = []

    for alpha in alpha_values:
        W, A_os = build_A_os_for_alpha(
            alpha=alpha,
            lam=lam,
            kernel=kernel_c_lam
        )

        print(f"\nW={W:g}, A_os shape={A_os.shape}")

        # Full SVD once per matrix
        start = time.perf_counter()
        s = np.linalg.svd(A_os, compute_uv=False)
        svd_time = time.perf_counter() - start

        svd_ranks = {
            tau: svd_numerical_rank(s=s, tau=tau)
            for tau in tau_values
        }

        print(f"SVD time={svd_time:.4f} sec, SVD ranks={svd_ranks}")

        # Calibration only at W=64 and tau=1e-5
        if alpha == calibration_alpha:
            print("\nParameter calibration at W=64, tau=1e-5")

            for gamma, N_b in parameter_grid:
                lr_result = informed_blocked_randomized_lr(
                    A=A_os,
                    tau=calibration_tau,
                    tau_R=tau_R,
                    gamma=gamma,
                    N_b=N_b,
                    seed=seed
                )

                r = lr_result["rank_lr"]
                E_lr = lr_result["final_error"]
                E_svd = best_svd_rank_error(s=s, r=r)
                rel_diff = (E_lr - E_svd) / E_svd if E_svd > 0 else np.nan

                row = {
                    "W": W,
                    "tau": calibration_tau,
                    "gamma": gamma,
                    "N_b": N_b,
                    "svd_rank": svd_ranks[calibration_tau],
                    "R_l": lr_result["R_l"],
                    "B0": lr_result["B0"],
                    "rank_lr": r,
                    "num_blocks": lr_result["num_blocks"],
                    "svd_time": svd_time,
                    "lr_time": lr_result["lr_time"],
                    "E_lr": E_lr,
                    "E_svd": E_svd,
                    "relative_difference": rel_diff
                }

                calibration_results.append(row)

                print(
                    f"gamma={gamma}, N_b={N_b}, "
                    f"R_l={row['R_l']}, B0={row['B0']}, "
                    f"rank={r}, blocks={row['num_blocks']}, "
                    f"LR time={row['lr_time']:.4f}, "
                    f"E_LR={E_lr:.3e}, E_SVD={E_svd:.3e}, "
                    f"rel_diff={rel_diff:.3e}"
                )

        # Final comparison using selected gamma and N_b
        for tau in tau_values:
            lr_result = informed_blocked_randomized_lr(
                A=A_os,
                tau=tau,
                tau_R=tau_R,
                gamma=selected_gamma,
                N_b=selected_N_b,
                seed=seed
            )

            r = lr_result["rank_lr"]
            E_lr = lr_result["final_error"]
            E_svd = best_svd_rank_error(s=s, r=r)
            rel_diff = (E_lr - E_svd) / E_svd if E_svd > 0 else np.nan

            row = {
                "W": W,
                "tau": tau,
                "gamma": selected_gamma,
                "N_b": selected_N_b,
                "svd_rank": svd_ranks[tau],
                "R_l": lr_result["R_l"],
                "B0": lr_result["B0"],
                "rank_lr": r,
                "num_blocks": lr_result["num_blocks"],
                "svd_time": svd_time,
                "lr_time": lr_result["lr_time"],
                "E_lr": E_lr,
                "E_svd": E_svd,
                "relative_difference": rel_diff
            }

            final_results.append(row)

            print(
                f"tau={tau:g}, "
                f"SVD rank={row['svd_rank']}, "
                f"R_l={row['R_l']}, B0={row['B0']}, "
                f"rank={r}, blocks={row['num_blocks']}, "
                f"LR time={row['lr_time']:.4f}, "
                f"E_LR={E_lr:.3e}, E_SVD={E_svd:.3e}, "
                f"rel_diff={rel_diff:.3e}"
            )

    # Save results
    header = [
        "W", "tau", "gamma", "N_b", "svd_rank", "R_l", "B0",
        "rank_lr", "num_blocks", "svd_time", "lr_time",
        "E_lr", "E_svd", "relative_difference"
    ]

    files_to_save = [
        ("Part III/Section j/parameter_calibration_W64.csv", calibration_results),
        ("Part III/Section j/final_comparison.csv", final_results)
    ]

    for filename, rows in files_to_save:
        out_path = Path("results") / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            f.write(",".join(header) + "\n")

            for row in rows:
                f.write(",".join(str(row[col]) for col in header) + "\n")

        print(f"Saved results to {out_path}")

    # Plot LR time vs SVD time
    fig, ax = plt.subplots(figsize=(6, 5))

    W_unique = sorted(set(row["W"] for row in final_results))
    svd_times = []

    for W in W_unique:
        for row in final_results:
            if row["W"] == W:
                svd_times.append(row["svd_time"])
                break

    ax.scatter(W_unique, svd_times, marker="o", label="Full SVD")

    for tau in tau_values:
        W_plot = [row["W"] for row in final_results if row["tau"] == tau]
        T_plot = [row["lr_time"] for row in final_results if row["tau"] == tau]
        ax.scatter(W_plot, T_plot, marker="x", label=f"Fast LR, tau={tau:g}")

    ax.set_xlabel("W")
    ax.set_ylabel("Computation time [sec]")
    ax.set_title("Fast LR Time vs Full SVD Time")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    out_path = Path("images") / "Part III/Section j/time_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")

    # Plot relative difference
    fig, ax = plt.subplots(figsize=(6, 5))

    for tau in tau_values:
        W_plot = [row["W"] for row in final_results if row["tau"] == tau]
        diff_plot = [row["relative_difference"] for row in final_results if row["tau"] == tau]
        ax.scatter(W_plot, diff_plot, label=f"tau={tau:g}")

    ax.set_xlabel("W")
    ax.set_ylabel(r"$(E_{LR}-E_{SVD,r})/E_{SVD,r}$")
    ax.set_title("Relative Difference Between Fast LR and Best SVD")
    ax.set_xscale("log", base=2)
    ax.grid(True)
    ax.legend()

    out_path = Path("images") / "Part III/Section j/relative_difference.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")