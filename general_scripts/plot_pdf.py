import numpy as np
import matplotlib.pyplot as plt

# Define the range of depths along the ray
near_thresh = 0.0
far_thresh = 1.0

# Number of samples along the ray
num_samples = 50

# Ray depths for visualization
ray_depths = np.linspace(near_thresh, far_thresh, num_samples)

# Calculate the PDF using a Gaussian distribution
mean_depth = (far_thresh + near_thresh) / 2.0
std_dev = 0.2  # Standard deviation for the Gaussian PDF
pdf_values = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((ray_depths - mean_depth) / std_dev) ** 2)

# Generate PDF-based sampled points along the ray
random_samples = np.random.rand(num_samples)
pdf_based_samples = near_thresh + random_samples * (far_thresh - near_thresh)

# Plotting the sampled points and PDF
plt.figure(figsize=(8, 6))

# Plotting the PDF
plt.plot(ray_depths, pdf_values, label='PDF (Gaussian)', color='red')
plt.title('Comparison of Sampling Strategies')
plt.xlabel('Depth along the Ray')
plt.ylabel('PDF Score')
plt.legend()
plt.grid()

# Plotting uniform samples
plt.scatter(ray_depths, np.zeros_like(ray_depths), label='Uniform Sampling', marker='o')

# Plotting PDF-based samples
plt.scatter(pdf_based_samples, np.ones_like(pdf_based_samples), label='PDF-based Sampling', marker='x')

plt.legend()
plt.yticks([0, 1], ['PDF', 'Samples'])
plt.grid(axis='x')
plt.show()
