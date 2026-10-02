"""
Server Health Anomaly Detection Using Gaussian Density Estimation
------------------------------------------------------------------
An unsupervised anomaly detection system built from scratch using NumPy
to monitor server latency and throughput metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(random_state=42):
    """
    Generates realistic 2D server performance metrics:
    - Feature 1: Latency (ms) - Response time of server requests.
    - Feature 2: Throughput (mbps) - Network/processing throughput.

    Returns:
        X_train (ndarray): 300 normal operational server metrics.
        X_val (ndarray): 66 validation samples (60 normal, 6 anomalies).
        y_val (ndarray): Binary labels for validation (1 = anomaly, 0 = normal).
        X_test (ndarray): 50 unseen server samples including anomalies.
    """
    rng = np.random.default_rng(random_state)

    # Typical normal server operations: ~15 ms latency, ~15 mbps throughput
    mean_normal = [15.0, 15.0]
    cov_normal = [[1.8, 0.4],
                  [0.4, 1.5]]

    # 1. Training set: 300 normal operational observations (unlabeled / purely normal)
    X_train = rng.multivariate_normal(mean_normal, cov_normal, size=300)

    # 2. Validation set: 60 normal points + 6 injected anomalous points
    val_normal = rng.multivariate_normal(mean_normal, cov_normal, size=60)
    # Anomalies represent server degradation (e.g., latency spikes, throughput drop/stall)
    val_anomalies = np.array([
        [22.5, 14.8],  # severe latency spike
        [15.2, 8.2],   # throughput collapse
        [24.0, 7.5],   # high latency and stalled throughput
        [8.5, 15.1],   # unexpected anomalous drop in response time
        [20.5, 20.0],  # abnormal resource surge
        [12.1, 21.5]   # excessive throughput anomaly
    ])
    X_val = np.vstack([val_normal, val_anomalies])
    y_val = np.hstack([np.zeros(60, dtype=int), np.ones(6, dtype=int)])

    # 3. Test set: 46 normal points + 4 distinct real-world anomaly spikes
    test_normal = rng.multivariate_normal(mean_normal, cov_normal, size=46)
    test_anomalies = np.array([
        [23.8, 14.5],  # severe latency spike
        [15.0, 6.8],   # network choke / throughput drop
        [22.0, 8.0],   # compound degradation
        [19.5, 19.8]   # traffic burst / memory leak symptom
    ])
    X_test = np.vstack([test_normal, test_anomalies])

    return X_train, X_val, y_val, X_test


def estimate_gaussian(X):
    """
    Calculates the mean vector (mu) and variance vector (var) for each feature.

    Args:
        X (ndarray): Dataset of shape (m, n) where m is samples, n is features.

    Returns:
        mu (ndarray): Mean of each feature, shape (n,).
        var (ndarray): Variance of each feature, shape (n,).
    """
    m = X.shape[0]
    mu = (1.0 / m) * np.sum(X, axis=0)
    var = (1.0 / m) * np.sum((X - mu) ** 2, axis=0)
    return mu, var


def compute_gaussian_probabilities(X, mu, var, eps=1e-12):
    """
    Computes the probability density p(x) for each sample assuming feature independence
    (product of univariate Gaussians):
        p(x) = prod_{j=1}^n [ 1 / sqrt(2 * pi * var_j) * exp(-(x_j - mu_j)^2 / (2 * var_j)) ]

    Args:
        X (ndarray): Feature matrix of shape (m, n).
        mu (ndarray): Feature means of shape (n,).
        var (ndarray): Feature variances of shape (n,).
        eps (float): Small epsilon added to variance for numerical stability.

    Returns:
        p (ndarray): Probability density for each sample, shape (m,).
    """
    safe_var = var + eps
    prefactor = 1.0 / np.sqrt(2.0 * np.pi * safe_var)
    exponent = np.exp(-((X - mu) ** 2) / (2.0 * safe_var))
    
    # Feature-wise probability densities: shape (m, n)
    feature_probs = prefactor * exponent
    
    # Joint probability under feature independence: product across columns
    return np.prod(feature_probs, axis=1)


def select_threshold(y_val, p_val):
    """
    Finds the optimal threshold (epsilon) to separate anomalies from normal points
    by maximizing the F1-score on labeled validation data.

    Args:
        y_val (ndarray): Ground truth binary labels (1 = anomaly, 0 = normal).
        p_val (ndarray): Computed probabilities for validation samples.

    Returns:
        best_epsilon (float): The threshold that achieved the highest F1-score.
        best_f1 (float): The highest F1-score achieved.
    """
    best_epsilon = 0.0
    best_f1 = 0.0

    p_min = np.min(p_val)
    p_max = np.max(p_val)
    step_size = (p_max - p_min) / 1000.0

    for epsilon in np.arange(p_min, p_max, step_size):
        # A sample is flagged as an anomaly if its probability density is below epsilon
        predictions = (p_val < epsilon).astype(int)

        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))

        # Precision & Recall with division-by-zero protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Harmonic mean (F1-score)
        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


def plot_anomaly_detection(X_train, X_test, mu, var, epsilon, train_anomalies, test_anomalies):
    """
    Renders 2D visualization of the server metrics:
    - Normal training distribution with Gaussian density contour lines.
    - Anomaly decision boundary (contour line at p(x) = epsilon).
    - Detected anomalies on both training and production test streams.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    # Generate grid coordinates for contour evaluation
    x1_range = np.linspace(6, 26, 200)
    x2_range = np.linspace(4, 24, 200)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[X1_grid.ravel(), X2_grid.ravel()]

    # Compute probability densities over the entire grid
    Z = compute_gaussian_probabilities(grid_points, mu, var).reshape(X1_grid.shape)

    # Logarithmic contour levels to showcase exponential decay of Gaussian density
    contour_levels = np.geomspace(max(epsilon * 0.1, 1e-8), np.max(Z), num=8)

    # ------------------ Subplot 1: Training Set & Learned Model ------------------
    ax1 = axes[0]
    ax1.scatter(X_train[:, 0], X_train[:, 1], color="#2b5c8f", alpha=0.6, edgecolors="none", s=35, label="Normal Training Samples")
    if np.any(train_anomalies):
        ax1.scatter(X_train[train_anomalies, 0], X_train[train_anomalies, 1],
                    facecolors="none", edgecolors="#e63946", s=130, linewidths=2.0, label="Flagged Anomalies")

    # Probability density contours
    contours_1 = ax1.contour(X1_grid, X2_grid, Z, levels=contour_levels, cmap="Blues", alpha=0.7, linewidths=1.0)
    ax1.clabel(contours_1, inline=True, fontsize=8, fmt="%.1e")

    # Decision boundary (p(x) = epsilon)
    ax1.contour(X1_grid, X2_grid, Z, levels=[epsilon], colors="#e63946", linestyles="dashed", linewidths=2.0)

    ax1.set_title("Training Data & Gaussian Density Contours", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Latency (ms)", fontsize=11)
    ax1.set_ylabel("Throughput (mbps)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    # ------------------ Subplot 2: Unseen Production Test Stream ------------------
    ax2 = axes[1]
    test_normal_mask = ~test_anomalies
    ax2.scatter(X_test[test_normal_mask, 0], X_test[test_normal_mask, 1],
                color="#2a9d8f", alpha=0.75, edgecolors="white", s=45, label="Normal Requests")
    ax2.scatter(X_test[test_anomalies, 0], X_test[test_anomalies, 1],
                color="#e76f51", marker="x", s=90, linewidths=2.5, label=f"Detected Anomalies (n={np.sum(test_anomalies)})")

    # Reference contours and decision boundary on test set
    ax2.contour(X1_grid, X2_grid, Z, levels=contour_levels, cmap="Greens", alpha=0.5, linewidths=0.9)
    decision_line = ax2.contour(X1_grid, X2_grid, Z, levels=[epsilon], colors="#e63946", linestyles="dashed", linewidths=2.0)

    ax2.set_title("Production Test Stream & Detected Anomalies", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Latency (ms)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Global formatting
    fig.suptitle(f"Server Health Anomaly Detection (Threshold $\\epsilon$ = {epsilon:.2e})", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save and show
    output_filename = "server_anomalies_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[+] Visual plot successfully saved to '{output_filename}'")
    plt.show()


def main():
    print("=" * 70)
    print("   SERVER HEALTH ANOMALY DETECTION USING GAUSSIAN DENSITY ESTIMATION")
    print("=" * 70)

    # Step 1: Generate synthetic server metrics
    print("\n[Step 1] Generating synthetic telemetry data...")
    X_train, X_val, y_val, X_test = generate_synthetic_data(random_state=42)
    print(f"  -> X_train: {X_train.shape[0]} normal operational samples")
    print(f"  -> X_val:   {X_val.shape[0]} samples ({np.sum(y_val == 0)} normal, {np.sum(y_val == 1)} injected anomalies)")
    print(f"  -> X_test:  {X_test.shape[0]} unseen production requests")

    # Step 2: Estimate Gaussian parameters from training set
    print("\n[Step 2] Estimating Gaussian parameters from normal operational data...")
    mu, var = estimate_gaussian(X_train)
    print(f"  -> Feature 1 (Latency ms):     mean = {mu[0]:.3f}, variance = {var[0]:.3f} (std = {np.sqrt(var[0]):.3f})")
    print(f"  -> Feature 2 (Throughput mbps): mean = {mu[1]:.3f}, variance = {var[1]:.3f} (std = {np.sqrt(var[1]):.3f})")

    # Step 3: Compute probability densities
    print("\n[Step 3] Computing Gaussian probability densities p(x)...")
    p_train = compute_gaussian_probabilities(X_train, mu, var)
    p_val = compute_gaussian_probabilities(X_val, mu, var)
    p_test = compute_gaussian_probabilities(X_test, mu, var)
    print(f"  -> Train density range: min = {np.min(p_train):.3e}, max = {np.max(p_train):.3e}")
    print(f"  -> Val density range:   min = {np.min(p_val):.3e}, max = {np.max(p_val):.3e}")
    print(f"  -> Test density range:  min = {np.min(p_test):.3e}, max = {np.max(p_test):.3e}")

    # Step 4: Tune decision threshold epsilon on validation set via F1-score
    print("\n[Step 4] Tuning decision threshold (epsilon) via F1-Score on validation set...")
    best_epsilon, best_f1 = select_threshold(y_val, p_val)
    print(f"  -> Optimal Epsilon: {best_epsilon:.6e}")
    print(f"  -> Optimal F1-Score: {best_f1:.4f}")

    # Evaluate validation metrics at best_epsilon
    val_preds = (p_val < best_epsilon).astype(int)
    val_tp = np.sum((val_preds == 1) & (y_val == 1))
    val_fp = np.sum((val_preds == 1) & (y_val == 0))
    val_fn = np.sum((val_preds == 0) & (y_val == 1))
    val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
    val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
    print(f"  -> Validation Metrics: Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, TP = {val_tp}/{np.sum(y_val)}, FP = {val_fp}")

    # Step 5: Detect anomalies across datasets
    print("\n[Step 5] Detecting anomalies using selected epsilon...")
    train_anomalies = (p_train < best_epsilon)
    test_anomalies = (p_test < best_epsilon)

    print(f"  -> Training Set:   {np.sum(train_anomalies)} / {len(X_train)} flagged as low-density anomalies ({np.mean(train_anomalies)*100:.2f}%)")
    print(f"  -> Production Test: {np.sum(test_anomalies)} / {len(X_test)} flagged as critical anomalies ({np.mean(test_anomalies)*100:.2f}%)")

    # Step 6: Visual Contour Plotting
    print("\n[Step 6] Generating visual contour plot with decision boundary...")
    plot_anomaly_detection(X_train, X_test, mu, var, best_epsilon, train_anomalies, test_anomalies)

    print("\n[Done] Execution completed successfully.\n")


if __name__ == "__main__":
    main()
