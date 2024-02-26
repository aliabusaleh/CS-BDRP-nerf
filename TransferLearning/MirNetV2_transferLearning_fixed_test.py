print("hello")
#exit()
import cProfile
import torch
import os

#import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
#from glob import glob
#from tqdm import tqdm
#import argparse
import numpy as np

import cv2
#import torch
from torch.utils.data import Dataset
#import torchvision.transforms.functional as F
from torchvision.transforms import Resize


#import torch
from torch import nn
#import os


# import data loader
from torch.utils.data import DataLoader



class DenoisingDataset(Dataset):
    def __init__(self, original_dir, reconstructed_dir, img_multiple_of=4):
        # List the images in the given directories
        self.original_paths = natsorted([os.path.join(original_dir, file) for file in os.listdir(original_dir)])
        self.reconstructed_paths = natsorted([os.path.join(reconstructed_dir, file) for file in os.listdir(reconstructed_dir)])
        self.img_multiple_of = img_multiple_of

    def __getitem__(self, index):
        original_path = self.original_paths[index]
        reconstructed_path = self.reconstructed_paths[index]

        # Load and preprocess the pair of original and reconstructed images
        original_image = self.load_and_preprocess_image(original_path)
        reconstructed_image = self.load_and_preprocess_image(reconstructed_path)

        return original_image, reconstructed_image

    def __len__(self):
        return len(self.original_paths)

    def load_and_preprocess_image(self, filepath):
        # print("inside load_and_preprocess_image")
        # Read image using OpenCV and convert to RGB
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        
        # Convert to torch tensor and normalize
        input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

        # Pad the input if not multiple of 4
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H, W = ((h + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of, \
               ((w + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - h if h % self.img_multiple_of != 0 else 0
        padw = W - w if w % self.img_multiple_of != 0 else 0
        # input_tensor = F.pad(input_tensor, (0, 0, 0, 0, padw, padh), 'reflect')

        return input_tensor



def get_weights_and_parameters(task, parameters):
    weights = os.path.join('Real_Denoising', 'pretrained_models', 'real_denoising.pth')
    return weights, parameters


# Define your DenoisingModel class
class DenoisingModel(nn.Module):
    def __init__(self, pretrained_model):
        super(DenoisingModel, self).__init__()
        self.features = pretrained_model
        # Get the number of output channels from the last layer of the MIRNet_v2 model
        in_channels = 3
        out_channels = 3
        kernel_size = 3
        padding = 1
        self.denoising_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding),
            nn.ReLU(),
            # Add more layers if needed
        )

    def forward(self, x):
        x = self.features(x)
        x = self.denoising_layer(x)
        return x




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    task = 'real_denoising'




    # Get model weights and parameters
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'n_feat': 80,
        'chan_factor': 1.5,
        'n_RRG': 4,
        'n_MRB': 2,
        'height': 3,
        'width': 2,
        'bias': False,
        'scale': 1,
        'task': task
    }

    weights, parameters = get_weights_and_parameters(task, parameters)
    #print("load architecure")
    # Load architecture
    load_arch = run_path('./basicsr/models/archs/mirnet_v2_arch.py')
    model = load_arch['MIRNet_v2'](**parameters)
    model.cuda()
    #print("load weights")
    # Load pretrained weights
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])

    # Freeze the last convolutional layer
    for param in model.parameters():
        # skip the last conv layer (conv_out)
        #if param.shape[0] == parameters['out_channels']:
        #    param.requires_grad = True
        #else:
           # print("not last conv layer")
        param.requires_grad = False
    for param in model.conv_out.parameters():
    #print("last conv layer")
            param.requires_grad = True
    model.eval()


    # original_dir = "./Real_Denoising/Datasets/originalNeRF/original/"
    # reconstructed_dir = "./Real_Denoising/Datasets/originalNeRF/reconstructed/"
    # # Instantiate the dataset and model
    # #print("load dataset")
    # dataset = DenoisingDataset(original_dir, reconstructed_dir, img_multiple_of=4)
    # model = DenoisingModel(model).cuda()



    # # Define loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # # Training loop
    # num_epochs = 5000
    # batch_size = 32
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # for epoch in range(num_epochs):
    #     print("Processing epoch: ", epoch)
    #     for original_image, reconstructed_image in dataloader:
    #         optimizer.zero_grad()

    #         #print(reconstructed_image.shape)
    #         reconstructed_image = reconstructed_image.squeeze(1)
    #         #print(reconstructed_image.shape)
        
    #         # Forward pass
    #         output_image = model(reconstructed_image.cuda())
    #         #print("Output size:", output_image.shape)
    #         #print("Target size:", original_image.squeeze(1).shape)
    #         # Resize target tensor to match the output size
    #         # Resize target tensor to match the output size
    #         resize_transform = Resize((398, 398))
    #         target_resized = resize_transform(original_image.squeeze(1))
    #         # Compute the loss
    #         loss = criterion(output_image, target_resized)

    #         # Backward and optimize
    #         loss.backward()
    #         optimizer.step()
    #         if (epoch % 10 ==0 & epoch != 0):
    #                 print("Finished epoch: ", epoch)
    #                 # save the model
    #                 torch.save(model.state_dict(), f'transferd_denoising_model_{epoch}.pth')
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    # # Save the model
    # torch.save(model.state_dict(), 'transferd_denoising_model.pth')



# if __name__ == "__main__":
#     cProfile.run("main()", sort="cumulative")


# test the mode
# load the model
    # model = DenoisingModel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/transferd_denoising_model_ruins2_180.pth'))
    model.eval()

    img_multiple_of = 4
    # get images names from ./Real_Denoising/Datasets/ruins2/unseen/files_to_move.txt
    images = open('./Real_Denoising/Datasets/originalNeRF/original_names.txt', 'r').readlines()
    print(len(images))
    for i in range(len(images)):
        # print(f"Processing image: {images[i].strip()}")
        # load the test image
        print("loading image", images[i].strip())
        img = cv2.cvtColor(cv2.imread(f'./Real_Denoising/Datasets/originalNeRF/reconstructed/{images[i].strip()}'), cv2.COLOR_BGR2RGB)
        # resize the image
        # img  = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 4
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        # print(f"saving image {images[i].strip()}")
        try:
            print(f"saving image {images[i].strip()}")
            # save original image 
            cv2.imwrite(f'./Real_Denoising/Datasets/originalNeRF/RuinsModel/TestOutput/{images[i].strip()}_nds.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            print(f"saved original image {images[i].strip()}")
            # save nds image
            cv2.imwrite(f'./Real_Denoising/Datasets/originalNeRF/RuinsModel/TestOutput/{images[i].strip()}_restored.png', cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

            # save the original image
            original = cv2.cvtColor(cv2.imread(f'./Real_Denoising/Datasets/originalNeRF/original/{images[i].strip()}'), cv2.COLOR_BGR2RGB)
            # original = cv2.resize(original, (0,0), fx=0.1, fy=0.1)
            cv2.imwrite(f'./Real_Denoising/Datasets/originalNeRF/RuinsModel/TestOutput/{images[i].strip()}_original.png', cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            
            # break
        except Exception as e:
            print(f"Error writing image: {e}")

    # # loop over the test images in the test folder
    # for i in range(2, 147):
    #     print(f"Processing image: {i}")
    #     # load the test image
    #     img = cv2.cvtColor(cv2.imread(f'./Real_Denoising/Datasets/ruins2/nds_outputLong2/ffd1a3ded5fbf0824aece0b6f9917823.jpg'), cv2.COLOR_BGR2RGB)
    #     # resize the image
    #     img  = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    #     input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

    #     # Pad the input if not_multiple_of 4
    #     h,w = input_.shape[2], input_.shape[3]
    #     H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    #     padh = H-h if h%img_multiple_of!=0 else 0
    #     padw = W-w if w%img_multiple_of!=0 else 0
    #     input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

    #     restored = model(input_)
    #     restored = torch.clamp(restored, 0, 1)

    #     # Unpad the output
    #     restored = restored[:,:,:h,:w]

    #     restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    #     restored = img_as_ubyte(restored[0])

    #     filename = os.path.split(f'./Real_Denoising/Datasets/originalNeRF/reconstructed/{i}.png')[-1]
    #     print(f"saving image ffd1a3ded5fbf0824aece0b6f9917823.jpg")
        
    #     try:
    #         # save original image 
    #         cv2.imwrite(f'./Real_Denoising/Datasets/originalNeRF/nds_data/ffd1a3ded5fbf0824aece0b6f9917823_original.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #         # save the denoised image
    #         cv2.imwrite(f'./Real_Denoising/Datasets/originalNeRF/nds_data/ffd1a3ded5fbf0824aece0b6f9917823_denoised.jpg', cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    #         break
    #     except Exception as e:
    #         print(f"Error writing image: {e}")

#     # load the test image
#     img = cv2.cvtColor(cv2.imread('./86.png'), cv2.COLOR_BGR2RGB)
#     input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()

#     # Pad the input if not_multiple_of 4
#     h,w = input_.shape[2], input_.shape[3]
#     H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
#     padh = H-h if h%img_multiple_of!=0 else 0
#     padw = W-w if w%img_multiple_of!=0 else 0
#     input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

#     restored = model(input_)
#     restored = torch.clamp(restored, 0, 1)

#     # Unpad the output
#     restored = restored[:,:,:h,:w]

#     restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
#     restored = img_as_ubyte(restored[0])

# #    filename = os.path.split(filepath)[-1]
#     cv2.imwrite("./86_denoised_image.png",cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
main()
