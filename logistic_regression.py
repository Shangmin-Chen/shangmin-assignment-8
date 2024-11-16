import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Flask compatibility
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 + distance], 
                                       cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def logistic_loss(X, y, model):
    probs = model.predict_proba(X)
    loss = -np.mean(y * np.log(probs[:, 1]) + (1 - y) * np.log(probs[:, 0]))
    return loss

def do_experiments(start, end, step_num):
    # Set up experiment parameters
    shift_distances = np.linspace(start, end, step_num)  # Range of shift distances
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list = [], [], [], [], [], []
    
    # Run experiments for each shift distance
    for distance in shift_distances:
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)

        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Calculate logistic loss
        loss = logistic_loss(X, y, model)
        loss_list.append(loss)

        # Save dataset plot with decision boundary
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        decision_boundary = slope * x_vals + intercept
        plt.plot(x_vals, decision_boundary, 'k--', label='Decision Boundary')
        plt.title(f"Shift Distance = {distance:.2f}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.savefig(f"{result_dir}/dataset_shift_{distance:.2f}.png")
        plt.close()

    # Plot parameters vs. shift distance
    plt.figure(figsize=(18, 15))

    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o')
    plt.title("Shift Distance vs Beta1")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o')
    plt.title("Shift Distance vs Beta2")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o')
    plt.title("Shift Distance vs Slope (Beta1 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o')
    plt.title("Shift Distance vs Intercept")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
