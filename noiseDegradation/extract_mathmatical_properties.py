from scipy.stats import skew, kurtosis
import os 
import cv2
import numpy as np


def calculate_residual(original_path, reconstructed_path):
    # Read the original and reconstructed images
    original = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    reconstructed = cv2.imread(reconstructed_path, cv2.IMREAD_UNCHANGED)
    
    # Convert images to float for accurate subtraction
    original_float = original.astype(np.float32)
    reconstructed_float = reconstructed.astype(np.float32)
    
    # Calculate the residual
    residual = original_float - reconstructed_float
    
    # Clip values to the range [0, 255] and convert to uint8
    residual_clipped = np.clip(residual, 0, 255).astype(np.uint8)
    
    return residual_clipped

def calculate_statistics(original_image_folder,reconstructed_image_folder):
    #original_image_folder : Something like drive/MyDrive/BDRP_CS/test_data/originalNeRF_400pxs_1epoch-20240130T095901Z-001/originalNeRF_400pxs_1epoch/reconstructed/
    #reconstructed_image_folder: Something like drive/MyDrive/BDRP_CS/test_data/originalNeRF_400pxs_1epoch-20240130T095901Z-001/originalNeRF_400pxs_1epoch/original/

    #Reconstruct_path and Original Path
    r_paths = sorted([reconstructed_image_folder + i for i in os.listdir(reconstructed_image_folder)])
    o_paths = sorted([original_image_folder + i for i in os.listdir(original_image_folder)])

    r_images = [cv2.imread(i) for i in r_paths]
    o_images = [cv2.imread(i) for i in o_paths]

    resi_images = [calculate_residual(i,j) for i,j in zip(o_paths,r_paths)]

    #average of all means
    mean = np.average([np.mean(resi_image) for resi_image in resi_images])
    #average of all stds
    std = np.average([np.std(resi_image) for resi_image in resi_images])
    #average of all skewness

    skewness = np.average([skew(resi_image.flatten()) for resi_image in resi_images])
    #average of all kurtosis
    kurtosisvalue = np.average([kurtosis(resi_image.flatten()) for resi_image in resi_images])

    return mean, std, skewness, kurtosisvalue

# create array for mean, std, skewness, kurtosisvalue
mean_array = []
std_array = []
skewness_array = []
kurtosis_array = []

# define the original and reconstructed image folder
original_image_folder = "D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/original/"
reconstructed_image_folder = "D:/CentraleSupelec/BDRP/denoising repo/MIRNetv2/Real_Denoising/Datasets/originalNeRF/reconstructed/"

# calculate the statistics
# loop over all the images in the folder
# get number of images in the folder
# num_images = len(os.listdir(original_image_folder))
# for i in range(num_images):
#     # get the image name
#     image_name = os.listdir(original_image_folder)[i]
#     # get the image path
#     image_path = original_image_folder + "/" + image_name
#     # get the reconstructed image path
#     reconstructed_image_path = reconstructed_image_folder + "/" + image_name
#     # calculate the statistics
#     mean, std, skewness, kurtosisvalue = calculate_statistics(image_path, reconstructed_image_path)
#     # append to the array
#     mean_array.append(mean)
#     std_array.append(std)
#     skewness_array.append(skewness)
#     kurtosis_array.append(kurtosisvalue)

mean, std, skewness, kurtosis = calculate_statistics(original_image_folder,reconstructed_image_folder)

# print the statistics
print("Mean: ", mean)
print("Standard Deviation: ", std)
print("Skewness: ", skewness)
print("Kurtosis: ", kurtosis)