
# Replace these paths with your actual directory paths
original_dir = "D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/original/"
reconstructed_dir = "D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/reconstructed"
simulated_dir = "D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/testOutput/"
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def calculate_metrics(original_path, reconstructed_path, simulated_path):
    original_images = [cv2.imread(os.path.join(original_path, img)) for img in os.listdir(original_path)]
    reconstructed_images = [cv2.imread(os.path.join(reconstructed_path, img)) for img in os.listdir(reconstructed_path)]
    simulated_images = [cv2.imread(os.path.join(simulated_path, img)) for img in os.listdir(simulated_path)]

    psnr_reconstructed = [psnr(original, reconstructed) for original, reconstructed in zip(original_images, reconstructed_images)]
    psnr_simulated = [psnr(original, simulated) for original, simulated in zip(original_images, simulated_images)]

    mse_reconstructed = [np.mean((original.astype(float) - reconstructed.astype(float))**2) for original, reconstructed in zip(original_images, reconstructed_images)]
    mse_simulated = [np.mean((original.astype(float) - simulated.astype(float))**2) for original, simulated in zip(original_images, simulated_images)]

    ncc_reconstructed = [np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1] for original, reconstructed in zip(original_images, reconstructed_images)]
    ncc_simulated = [np.corrcoef(original.flatten(), simulated.flatten())[0, 1] for original, simulated in zip(original_images, simulated_images)]

    return psnr_reconstructed, psnr_simulated, mse_reconstructed, mse_simulated, ncc_reconstructed, ncc_simulated

def plot_metrics(psnr_reconstructed, psnr_simulated, mse_reconstructed, mse_simulated, ncc_reconstructed, ncc_simulated):
    plt.figure(figsize=(20, 20))

    # Histograms for PSNR
    plt.subplot(3, 2, 1)
    plt.hist(psnr_simulated, bins=20, color='orange', alpha=0.7, label='Simulated')
    plt.hist(psnr_reconstructed, bins=20, color='blue', alpha=0.7, label='Reconstructed')
    plt.title('PSNR - Reconstructed vs. Simulated')
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')
    plt.legend()

    # Histograms for MSE
    plt.subplot(3, 2, 2)
    plt.hist(mse_simulated, bins=20, color='red', alpha=0.7, label='Simulated')
    plt.hist(mse_reconstructed, bins=20, color='green', alpha=0.7, label='Reconstructed')
    plt.title('MSE - Reconstructed vs. Simulated')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.legend()

    # Histograms for NCC
    plt.subplot(3, 2, 3)
    plt.hist(ncc_simulated, bins=10, color='brown', alpha=0.7, label='Simulated')
    plt.hist(ncc_reconstructed, bins=10, color='purple', alpha=0.7, label='Reconstructed')
    plt.title('NCC - Reconstructed vs. Simulated')
    plt.xlabel('NCC')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

psnr_reconstructed, psnr_simulated, mse_reconstructed, mse_simulated, ncc_reconstructed, ncc_simulated = calculate_metrics(original_dir, reconstructed_dir, simulated_dir)
plot_metrics(psnr_reconstructed, psnr_simulated, mse_reconstructed, mse_simulated, ncc_reconstructed, ncc_simulated)
