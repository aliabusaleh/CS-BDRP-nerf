## For evaluating with metrics and plotting the results

import matplotlib.pyplot as plt
import numpy as np

# Image processing
import cv2

# Metrics
from skimage.metrics import structural_similarity as ssim
import lpips

def rgb2gray(rgb):
  "used for ssim"
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_metrics_diff(ground_truth, reconstructed, filtered, name='', output_recons=True,use_lpips=True):
    """get change in PSNR, SSIM and LPIPS if possible
    after applying a filter to the reconstructed image"""
    ## PSNR
    # print(name)
    recons_PSNR = cv2.PSNR(ground_truth, reconstructed)
    filter_PSNR = cv2.PSNR(ground_truth, filtered)
    # if output_recons:
    #   print('recons PSNR:', recons_PSNR)

    diff = (filter_PSNR - recons_PSNR) / recons_PSNR
    psnr_diff = round(diff * 100, 3)
    # print('PSNR:', filter_PSNR,', % change: ', diff)

    ## SSIM
    # converting to grayscale
    rgb_predicted1 = rgb2gray(reconstructed)
    target_img1 = rgb2gray(ground_truth)
    recons_SSIM = ssim(target_img1, rgb_predicted1,
                       data_range=rgb_predicted1.max() - rgb_predicted1.min())
    # if output_recons:
    #   print('recons SSIM:', recons_SSIM)
    
    rgb_predicted1 = rgb2gray(filtered)
    filter_SSIM = ssim(target_img1, rgb_predicted1,
                       data_range=rgb_predicted1.max() - rgb_predicted1.min())
    diff = (filter_SSIM - recons_SSIM) / recons_SSIM
    ssim_diff = round(diff * 100, 3)
    # print('SSIM: ', filter_SSIM, ', % change: ', diff)
    # print()

    ## LPIPS
    if use_lpips:
      loss_fn = lpips.LPIPS(net='alex') # best forward scores
      loss_fn.cuda()

      img0 = lpips.im2tensor(ground_truth)
      img1 = lpips.im2tensor(reconstructed)
      img2 = lpips.im2tensor(filtered)

      img0 = img0.cuda()
      img1 = img1.cuda()
      img2 = img2.cuda()

      recons_LPIPS = loss_fn.forward(img0, img1)
      recons_LPIPS = round(float(recons_LPIPS), 3)
      filter_LPIPS = loss_fn.forward(img0, img2)
      filter_LPIPS = round(float(filter_LPIPS), 3)

      # if output_recons:
        # print('recons LPIPS:', recons_LPIPS)

      diff = (recons_LPIPS - filter_LPIPS)/recons_LPIPS
      lpips_diff = round(diff * 100, 3)
      # print('LPIPS:', filter_LPIPS, ', % change: ', diff)
      return psnr_diff, ssim_diff, lpips_diff
    else:
      return psnr_diff, ssim_diff
    
def plot_images(reconstructed, filtered, gray=False):
    """plot in the same plot the two images"""
    fig,ax = plt.subplots(1,2)
    
    if gray==True:
        ax[0].imshow(reconstructed, cmap=plt.get_cmap('gray'))
    else:
        ax[0].imshow(reconstructed)
    ax[0].set_title("Reconstructed")

    if gray==True:
        ax[1].imshow(filtered, cmap=plt.get_cmap('gray'))
    else:
        ax[1].imshow(filtered)
    ax[1].set_title("Filtered")

def read_images(gt_image, reconst_image, denoised=None):
    """read the images"""
    ground_truth = cv2.imread(gt_image)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

    reconstructed = cv2.imread(reconst_image)
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)

    if denoised != None:
        denoised = cv2.imread(denoised)
        denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    return ground_truth, reconstructed, denoised

def get_imgPairs(reconst_images, gt_images):
    """pairing up ground truth and reconstructed images together
       both should have the same name"""
    img_pairs = []
    for reconst_img in reconst_images:
        reconst_name = '/'+reconst_img.split("/")[-1]
        gt_name = [fn for fn in gt_images if reconst_name in fn]
        if len(gt_name) == 0:
            pass
        else: 
            img_pairs.append([reconst_img, gt_name[0]])

    return img_pairs

def get_imgTriples(reconst_images, gt_images, denoised_images):
    """pairing up ground truth and reconstructed images together
       both should have the same name"""
    img_pairs = []
    for reconst_img in reconst_images:
        reconst_name = '/'+reconst_img.split("/")[-1]
        gt_name = [fn for fn in gt_images if reconst_name in fn]
        denoised_name = [fn for fn in denoised_images if reconst_name in fn]
        if len(gt_name) == 0 or len(denoised_name) == 0:
            print(f"Couldn't find for {reconst_name}")
            pass
        else: 
            img_pairs.append([reconst_img, gt_name[0], denoised_name[0]])

    return img_pairs

def plot_metricsDiff(df, methods, methods_cols, idxs=[2,4], notShow_idxs=[], h_plus=5,xlabel='Filters'):
    """
    Plot metricsDiff, 
    pd.df should have 2 rows "psnr_mean" and "ssim_mean" on 2nd and 4th indeces
        method_cols - the column names of df for each method to consider
        methods - the labels of the methods, the same order as method_cols
    idxs - row idxs of psnr mean and ssim mean
    notShow_idxs - the idxs of the SSIM bars labels not to show, 
        because they overlap with PSNR bars
    """
    psnrs = [df.iloc[idxs[0]][methods_cols[j]] for j in range(len(methods_cols))]
    ssims = [df.iloc[idxs[1]][methods_cols[j]] for j in range(len(methods_cols))]

    # sort based on values of psnrs
    sorted_indices = np.argsort(psnrs)[::-1]
    psnrs = np.array(psnrs)[sorted_indices]
    ssims = np.array(ssims)[sorted_indices]
    methods = np.array(methods)[sorted_indices]
    methods_cols = np.array(methods_cols)[sorted_indices]

    x = np.arange(len(psnrs)) 
    width = 0.35

    _, ax = plt.subplots()
    # plot data in grouped manner of bar type 
    bars1 = ax.bar(x-0.2, psnrs, width, label='PSNR') 
    bars2 = ax.bar(x+0.2, ssims, width, label='SSIM') 

    ax.set_xticks(x)
    ax.set_xticks(x, methods, rotation=45, ha="left")
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel(xlabel)

    ax.set_ylabel('% Change in the Metrics')
    ax.legend()

    # Annotate the bars with values at the top or bottom
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height+h_plus if height > 0 else height-h_plus, f'{height:.2f}',
                ha='center', va='bottom' if height < 0 else 'top')
    for i, bar in enumerate(bars2):
        if i in notShow_idxs:
            continue
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height+h_plus if height > 0 else height-h_plus, f'{height:.2f}',
                ha='center', va='bottom' if height < 0 else 'top')

    # Horizontal line to signify no change
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.show()