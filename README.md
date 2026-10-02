# Server Health Anomaly Detection Using Gaussian Density Estimation

A standalone, from-scratch unsupervised anomaly detection system built with NumPy to monitor server performance health metrics (request latency and CPU/network throughput).

---

## 1. Project Overview

Modern cloud infrastructure produces millions of telemetry data points per minute. Detecting hardware degradation, memory leaks, connection bottlenecks, and unexpected server failures requires identifying when system behavior deviates from nominal patterns.

This project implements an unsupervised Gaussian Density Estimation model:
1. **Unsupervised Fit:** Learns the distribution parameters ($\mu, \sigma^2$) strictly from normal, unlabelled server operational telemetry ($X_{\text{train}}$).
2. **Threshold Tuning:** Evaluates candidate thresholds ($\varepsilon$) on a small labeled validation set ($X_{\text{val}}, y_{\text{val}}$) by maximizing the $F_1$-score.
3. **Inference & Alerting:** Flags incoming production requests ($X_{\text{test}}$) as anomalous if their joint probability density $p(x) < \varepsilon$.
4. **Visual Diagnosis:** Produces density contour plots demonstrating the elliptical decision boundary separating nominal server behavior from degradation.

---

## 2. Why Anomaly Detection vs. Supervised Classification?

In industry settings, server monitoring is almost always framed as **unsupervised anomaly detection** rather than standard supervised binary classification (e.g., Logistic Regression, Random Forest, or XGBoost). The primary reasons are:

| Consideration | Supervised Classification | Unsupervised Anomaly Detection |
|---|---|---|
| **Class Imbalance** | Requires large numbers of both positive ($y=1$) and negative ($y=0$) examples. Struggles when positive examples are < 0.1%. | Designed specifically for extreme asymmetry where $99.9\%+$ of traffic is normal ($y=0$) and anomalies ($y=1$) are exceedingly rare. |
| **Failure Modes** | Assumes future anomalies look identical to historical training anomalies. | Assumes anomalies are simply *any* low-probability deviation from nominal behavior. |
| **Novel / Zero-Day Outages** | Fails to detect new kinds of failure (e.g., novel cyber-attacks, untested race conditions, or uncharacterized hardware faults). | Naturally flags zero-day anomalies because any unseen anomaly pattern will yield an extremely low probability density $p(x) < \varepsilon$. |

**Summary Rule of Thumb:**
- Use **Supervised Classification** when you have plenty of positive examples and expect future positive cases to be similar to past cases (e.g., spam filtering, image categorization).
- Use **Anomaly Detection** when positive examples are extremely scarce, and future anomalies could manifest in completely new, unforeseen ways (e.g., server health monitoring, aircraft engine telemetry, fraud detection).

---

## 3. Why $F_1$-Score Instead of Accuracy for Selecting $\varepsilon$?

When choosing the threshold $\varepsilon$, optimizing for raw **Accuracy** is dangerously misleading due to severe class imbalance.

Suppose our validation set contains $1,000$ nominal server queries and $2$ anomalous failures ($99.8\%$ normal):
- A trivial dummy model that predicts every single point is "normal" ($y=0$) achieves **99.8% Accuracy**.
- However, this model catches **0 out of 2 anomalies**, leading to catastrophic silent server downtime.

By optimizing for the **$F_1$-score** (the harmonic mean of Precision and Recall), we force the threshold $\varepsilon$ to balance:
- **Precision:** $\frac{TP}{TP + FP}$ — Preventing alert fatigue by keeping false alarms low.
- **Recall:** $\frac{TP}{TP + FN}$ — Catching as many genuine system degradations as possible.

---

## 4. Mathematical Foundations

### 4.1. Estimating Mean and Variance
Given an unlabeled training dataset $X \in \mathbb{R}^{m \times n}$ with $m$ samples and $n$ features, the parameters for each feature $j \in \{1, \dots, n\}$ are computed as:

$$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$$

$$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} \left(x_j^{(i)} - \mu_j\right)^2$$

Where:
- $\mu \in \mathbb{R}^n$ is the feature mean vector.
- $\sigma^2 \in \mathbb{R}^n$ is the feature variance vector.

### 4.2. Gaussian Probability Density Function
Assuming independence between features, the joint probability density $p(x)$ of an observation $x = [x_1, x_2, \dots, x_n]^T$ is the product of univariate Gaussians:

$$p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$

An observation $x$ is flagged as an anomaly if:

$$p(x) < \varepsilon$$

### 4.3. Evaluation Metrics ($F_1$-Score)
To tune the decision boundary $\varepsilon$, predictions are compared against labeled ground truth validation points:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:
- **$TP$ (True Positives):** Model predicted anomaly ($p(x) < \varepsilon$) and it was a true anomaly ($y = 1$).
- **$FP$ (False Positives):** Model predicted anomaly ($p(x) < \varepsilon$) but it was normal ($y = 0$).
- **$FN$ (False Negatives):** Model predicted normal ($p(x) \ge \varepsilon$) but it was actually an anomaly ($y = 1$).

---

## 5. Project Structure & Execution

### 5.1. File Structure
```text
server-health-anomaly-detector/
├── server_anomaly_detector.py   # Standalone Python script (NumPy + Matplotlib)
├── server_anomalies_plot.png    # Output contour and anomaly visualization
├── requirements.txt             # Minimal dependencies (numpy, matplotlib)
└── README.md                    # Project explanation, math, and interview guide
```

### 5.2. Installation & Running
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the detector:
```bash
python server_anomaly_detector.py
```

### 5.3. Expected Terminal Output
```text
======================================================================
   SERVER HEALTH ANOMALY DETECTION USING GAUSSIAN DENSITY ESTIMATION
======================================================================

[Step 1] Generating synthetic telemetry data...
  -> X_train: 300 normal operational samples
  -> X_val:   66 samples (60 normal, 6 injected anomalies)
  -> X_test:  50 unseen production requests

[Step 2] Estimating Gaussian parameters from normal operational data...
  -> Feature 1 (Latency ms):      mean = 15.045, variance = 1.957 (std = 1.399)
  -> Feature 2 (Throughput mbps): mean = 14.991, variance = 1.206 (std = 1.098)

[Step 3] Computing Gaussian probability densities p(x)...
  -> Train density range: min = 1.109e-04, max = 1.033e-01
  -> Val density range:   min = 1.025e-20, max = 9.753e-02
  -> Test density range:  min = 7.008e-16, max = 1.035e-01

[Step 4] Tuning decision threshold (epsilon) via F1-Score on validation set...
  -> Optimal Epsilon: 9.753246e-05
  -> Optimal F1-Score: 1.0000
  -> Validation Metrics: Precision = 1.0000, Recall = 1.0000, TP = 6/6, FP = 0

[Step 5] Detecting anomalies using selected epsilon...
  -> Training Set:    0 / 300 flagged as low-density anomalies (0.00%)
  -> Production Test: 4 / 50 flagged as critical anomalies (8.00%)

[Step 6] Generating visual contour plot with decision boundary...

[+] Visual plot successfully saved to 'server_anomalies_plot.png'

[Done] Execution completed successfully.
```

---

## 6. Visual Interpretation

The generated visualization (`server_anomalies_plot.png`) consists of two synchronized panels:

1. **Left Panel (Model Fitting & Decision Boundary):**
   - Shows normal training points ($X_{\text{train}}$) clustered around nominal operating conditions ($\sim 15\text{ ms}$, $\sim 15\text{ mbps}$).
   - Concentric blue density contours visualize the fitted Gaussian probability density surface.
   - The **dashed red contour** represents the exact decision boundary where $p(x) = \varepsilon$. Any observation falling outside this contour is flagged as an anomaly.

2. **Right Panel (Production Evaluation):**
   - Evaluates incoming unseen server requests ($X_{\text{test}}$).
   - Nominal requests (green circles) sit comfortably within the high-density interior.
   - Anomaly spikes (red crosses), such as high-latency stalls ($23.8\text{ ms}$) or network throughput collapses ($6.8\text{ mbps}$), fall well outside the boundary into near-zero density space ($p(x) \ll \varepsilon$) and are flagged immediately.

---

## 7. Machine Learning Engineering Interview Q&A

### Question 1: "What is the assumption behind modeling the joint distribution as the product of individual Gaussians, and what would you do if features are strongly correlated?"
> **Answer:**
> Modeling $p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2)$ assumes that the features are mutually conditionally independent. Geometrically, this restricts the probability density contours to axis-aligned ellipses.
>
> If features exhibit strong covariance (for example, if latency naturally scales with throughput under heavy load), a point might fall along an unexpected off-axis combination without having an extreme value on either single feature.
>
> To capture this, I would upgrade from independent Gaussians to a full **Multivariate Gaussian Distribution** with a full covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$:
> $$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$
> The trade-off is computational cost: computing $\Sigma^{-1}$ and $\det(\Sigma)$ takes $\mathcal{O}(n^3)$ time and requires $m > n$ samples (preferably $m \ge 10n$) to prevent $\Sigma$ from becoming non-invertible/singular. The independent model is $\mathcal{O}(n)$ and works even when $m < n$.

---

### Question 2: "What happens if a feature is not normally distributed (e.g., highly skewed)? How do you address it?"
> **Answer:**
> Gaussian density estimation assumes that feature values follow a bell-shaped normal distribution. If a feature has heavy right-skew (e.g., web response latencies or server memory consumption), the estimated mean $\mu$ will be pulled toward the tail and variance $\sigma^2$ will be overestimated, distorting density estimates and leading to false negatives.
>
> In practice, we inspect histograms or Q-Q plots of each feature. If strong skewness is present, we apply monotonic transformations prior to fitting parameters:
> - **Log Transformation:** $x \leftarrow \log(x + c)$ (very common for strictly positive, right-skewed metrics).
> - **Power / Box-Cox Transformation:** $x \leftarrow \frac{x^\lambda - 1}{\lambda}$.
> - **Square Root:** $x \leftarrow \sqrt{x}$ (effective for count-based metrics).
>
> These transformations reshape the empirical distribution into a closer approximation of a Gaussian, enabling the density model to accurately delineate true outliers.

---

### Question 3: "In production, how would you handle concept drift where normal server behavior shifts over time?"
> **Answer:**
> Server workloads fluctuate due to software deployments, new API endpoints, seasonal traffic spikes (e.g., Black Friday), or hardware upgrades. A static $(\mu, \sigma^2, \varepsilon)$ model will quickly become stale and cause false positive alert storms or miss genuine incidents.
>
> To mitigate this, I would implement:
> 1. **Exponentially Weighted Moving Statistics:** Instead of static batch mean and variance, update $\mu_t$ and $\sigma_t^2$ continuously using streaming exponential decay:
>    $$\mu_t = \alpha x_t + (1 - \alpha) \mu_{t-1}$$
> 2. **Sliding Window Retraining:** Re-estimate $\mu$ and $\sigma^2$ over rolling windows (e.g., past 7 or 14 days) to capture baseline drift while remaining resilient to intraday volatility.
> 3. **Contextual Partitioning:** Separate models by traffic regime (e.g., distinct models for peak business hours vs. off-peak night maintenance, or weekday vs. weekend).
> 4. **Shadow Validation:** Continually collect verified alerts and operational feedback to re-validate that the threshold $\varepsilon$ maintains target Precision/Recall thresholds over time.
