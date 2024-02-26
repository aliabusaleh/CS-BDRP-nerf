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
from glob import glob
from tqdm import tqdm
import argparse
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
        # compress the image
        img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)

        # Convert to torch tensor and normalize
        input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()

        # Pad the input if not multiple of 8
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H, W = ((h + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of, \
               ((w + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - h if h % self.img_multiple_of != 0 else 0
        padw = W - w if w % self.img_multiple_of != 0 else 0
        input_tensor = F.pad(input_tensor, (0,padw,0,padh), 'reflect')

        return input_tensor



def get_weights_and_parameters(task, parameters):
    weights = os.path.join('Real_Denoising', 'pretrained_models', 'real_denoising.pth')
    return weights, parameters


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

    # unFreeze the last convolutional layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv_out.parameters():
            param.requires_grad = True

    model.eval()


    original_dir = "./Real_Denoising/Datasets/ruins2/images/"
    reconstructed_dir = "./Real_Denoising/Datasets/ruins2/nds_outputLong2/"
    # Instantiate the dataset and model
    #print("load dataset")
    dataset = DenoisingDataset(original_dir, reconstructed_dir, img_multiple_of=8)

    # split the dataset into training and validation sets
    # add the random seed for reproducibility
    torch.manual_seed(233)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    batch_size = 32
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    img_multiple_of = 8
    # loss dictionary
    loss_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print("Processing epoch: ", epoch)
        for original_image, reconstructed_image in dataloader:
            optimizer.zero_grad()

            #print(reconstructed_image.shape)
            reconstructed_image = reconstructed_image.squeeze(1)
            #print(reconstructed_image.shape)
        
            # Forward pass
            output_image = model(reconstructed_image.cuda())
            original_image = original_image.squeeze(1)
            # Compute the loss
            loss = criterion(output_image, original_image)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            if (epoch % 10 ==0):
                    print("Finished epoch: ", epoch)
                    # save the model
                    torch.save(model.state_dict(), f'./pretrained_model/transferd_denoising_model_ruins2_{epoch}.pth')
                    # save the loss
                    loss_dict['train'].append(loss.item())
                    # validate the model
                    val_loss = 0
                    with torch.no_grad():
                        for original_image, reconstructed_image in val_dataloader:
                            reconstructed_image = reconstructed_image.squeeze(1)
                            output_image = model(reconstructed_image.cuda())
                            original_image = original_image.squeeze(1)
                            val_loss += criterion(output_image, original_image)
                    val_loss /= len(val_dataset)
                    loss_dict['val'].append(val_loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # save the loss dictionary
    np.save('loss_dict.npy', loss_dict)
    # Save the model
    torch.save(model.state_dict(), './pretrained_model/transferd_denoising_ruins2_model.pth')



# if __name__ == "__main__":
#     cProfile.run("main()", sort="cumulative")


# # test the mode
# # load the model
# model = DenoisingModel(model)
# model.load_state_dict(torch.load('denoising_model.pth'))
# model.eval()

# # load the test image
# test_image = cv2.cvtColor(cv2.imread('./2.png'), cv2.COLOR_BGR2RGB)
# test_image = torch.from_numpy(test_image).float().div(255.).permute(2, 0, 1).unsqueeze(0)

# # denoise the test image
# denoised_image = model(test_image)

# # save the denoised image
# denoised_image = denoised_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
# denoised_image = img_as_ubyte(denoised_image)
# cv2.imwrite('denoised_image.png', cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))

main()

