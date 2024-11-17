import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2[:, 0] += distance  # Shift along x-axis
    X2[:, 1] += distance  # Shift along y-axis
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

def do_experiments(start, end, step_num):
    # Set up experiment parameters
    shift_distances = np.linspace(start, end, step_num, endpoint=True)  # Range of shift distances
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}  # Store sample datasets and models for visualization

    n_samples = 8
    n_cols = 2  # Fixed number of columns
    n_rows = (n_samples + n_cols - 1) // n_cols  # Calculate rows needed
    plt.figure(figsize=(20, n_rows * 10))  # Adjust figure height based on rows

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
    
        # Implement: record all necessary information for each distance
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        # Store the model, coefficients, and dataset
        sample_data[distance] = (X, y, model, beta0, beta1, beta2)
        
        # Implement: Plot the dataset
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.6)
        plt.title(f"Dataset for Shift Distance = {distance:.2f}", fontsize=18)
        plt.xlabel("x1", fontsize=14)
        plt.ylabel("x2", fontsize=14)
        # Add legend only for the first plot
        if i == 1:  # Include the legend only in the first plot
            plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)

        
        # Implement: Calculate and store logistic loss
        proba = model.predict_proba(X)  # Get predicted probabilities
        log_loss = -np.mean(y * np.log(proba[:, 1]) + (1 - y) * np.log(1 - proba[:, 1]))  # Logistic loss formula
        loss_list.append(log_loss)  # Append the calculated loss to the list
        
        # Calculate margin width between 70% confidence contours for each class
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        plt.contourf(xx, yy, Z, levels=[0.0, 0.7], alpha=0.2, colors=['blue'])
        plt.contourf(xx, yy, Z, levels=[0.3, 1.0], alpha=0.2, colors=['red'])
        
        # Plot fading red and blue contours for confidence levels
        class_1_contour = plt.contour(xx, yy, Z, levels=[0.7], colors=['red'], alpha=0)
        class_0_contour = plt.contour(xx, yy, Z, levels=[0.3], colors=['blue'], alpha=0)

        # Extract paths for margin width calculation
        try:
            vertices_1 = class_1_contour.collections[0].get_paths()[0].vertices
            vertices_0 = class_0_contour.collections[0].get_paths()[0].vertices
            distances = cdist(vertices_1, vertices_0)
            margin_width = np.min(distances)
            margin_widths.append(margin_width)
        except IndexError:
            margin_widths.append(0)  # If contours are not present, append 0

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=12)
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot 1: Parameters vs. Shift Distance
    plt.figure(figsize=(18, 15))

    # Implement: Plot beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o', linestyle='-', color='blue')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Implement: Plot beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o', linestyle='-', color='green')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Implement: Plot beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o', linestyle='-', color='red')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Implement: Plot beta1 / beta2 (Slope)
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o', linestyle='-', color='purple')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1 / Beta2")

    # Implement: Plot beta0 / beta2 (Intercept ratio)
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o', linestyle='-', color='orange')
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o', linestyle='-', color='brown')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Implement: Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', linestyle='-', color='cyan')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)