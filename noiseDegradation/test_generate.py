import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Function to generate noise with specified properties
def generate_noise_with_properties(shape, mean, std_dev, desired_skewness, desired_kurtosis):
    np.random.seed(42)
    noise = np.random.normal(0, 1, size=shape)

    # Normalize to desired mean and standard deviation
    current_mean = np.mean(noise)
    current_std_dev = np.std(noise)
    normalized_noise = (noise - current_mean) * (std_dev / current_std_dev) + mean

    # Adjust skewness
    current_skewness = skew(normalized_noise.flatten())
    skewness_diff = desired_skewness - current_skewness
    normalized_noise = normalized_noise + skewness_diff * (normalized_noise - np.mean(normalized_noise))

    # Adjust kurtosis
    current_kurt = kurtosis(normalized_noise.flatten())
    kurt_diff = desired_kurtosis - current_kurt
    normalized_noise = normalized_noise + kurt_diff * ((normalized_noise - np.mean(normalized_noise))**2 - current_std_dev**2)

    return normalized_noise

# Load an example image (replace 'your_image_path' with the path to your image)
# Example using a grayscale image (you can modify it for RGB images)
original_image = plt.imread('D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/original/10.png')

# Specify the desired statistical properties
desired_mean = 3.061528275862069
desired_std_dev = 10.207202023813368
desired_skewness = 6.745085125565042
desired_kurtosis = 70.47755010697897

# Generate noise with specified properties
noise_properties = generate_noise_with_properties(original_image.shape, desired_mean, desired_std_dev, desired_skewness, desired_kurtosis)

# Add the noise to the original image
degraded_image = original_image + noise_properties

# Clip the values to be in the valid image range
degraded_image = np.clip(degraded_image, 0, 255)

# Display the original and degraded images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(degraded_image, cmap='gray')
plt.title('Degraded Image')

plt.show()


