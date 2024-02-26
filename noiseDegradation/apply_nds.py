import cv2
import torchvision
import torch
import concurrent.futures

from vimeo_7f import aug_moving
from nds_degradation import *

import os
import argparse

def run_for_dir(img_directory, output_directory):
    """Apply NeRF-Style Degradation Simulator (NDS)
        adapted from https://github.com/redrock303/NeRFLiX_CVPR2023
        - img_directory - directory where the original images are stored
        - output_directory - directory where to store the noisy images"""

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # Use ThreadPoolExecutor for I/O-bound tasks
        futures = []
        for i, filename in enumerate(os.listdir(img_directory)):
            if filename.endswith(".png"):
                img_path = os.path.join(img_directory, filename)
                output_path = os.path.join(output_directory, filename)

                if not os.path.exists(output_path):
                    future = executor.submit(apply_nds, img_path, output_path)
                    futures.append(future)
                # else:
                #     print("Already")
        
        concurrent.futures.wait(futures)
            
        # if i%100 == 0:
        #     print(f"Processed {i} images")
        
        # if i > 5:
        #     break

def apply_nds(img_path, output_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    patch_width=196
    patch_height=196

    H,W, c = img.shape
    coord_map_dict ={'{}-{}'.format(patch_height,patch_width):defineCoorMap(patch_height,patch_width)}
    key = '{}-{}'.format(H,W)
    if key in coord_map_dict:
        coord_map = coord_map_dict[key]
    else :
        coord_map_dict[key] = defineCoorMap(H,W)
        coord_map = coord_map_dict[key]
    mask = defineHighlightArea(H,W,coord_map.copy())
    tmp = color_jet(img)
    # img = img * (1-mask) + mask * tmp

    img,jpeg_quality,noise_level = process(img,coord_map.copy())
    img = reposition(img,ratio=0.3)
    img = img.transpose(2,0,1)
    lr_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).float()
    torchvision.utils.save_image(lr_tensor,output_path)
    # print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Apply NeRF-Style Degradation Simulator (NDS) to images in a directory")
    parser.add_argument('--img_dir', help='directory where the original images are stored')
    parser.add_argument('--out_dir', help='directory where to store the noisy images')

    args = parser.parse_args()
    arguments = vars(args)
    run_for_dir(arguments['img_dir'], arguments['out_dir'])