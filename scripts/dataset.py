#!/usr/bin/env python3
import os
from re import sub
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
from PIL import Image
import math
import monai
import nibabel as nib

## Define a dataset class for the doppler images. Needs __init__, __getitem__, and __len__.
class BrainDataset(Dataset):
    def __init__(self,
                 split = 'train',
                 subvolume_size = [110,110,100],
                 transform = True,
                 target_transform = None,
                 file_list_path = '',
                 split_col = 'split',
                 image_path_col = 't1_brain_path',
                 label_path_col = 'structure_label_path'
    ):
        self.split = split
        #self.subvolume_size = [subvolume_size, subvolume_size, subvolume_size]
        self.subvolume_size = subvolume_size
        self.transform = transform # Should add normalization.
        self.target_transform = target_transform
        self.file_list_path = file_list_path
        self.split_col = split_col
        self.image_path_col = image_path_col
        self.label_path_col = label_path_col

        data_df = pd.read_csv(file_list_path)
        data_df = data_df.loc[data_df[split_col] == split, :] # only the portion of the dataframe with the requested split (i.e. separate dataframes for training and validation sets)
        data_df.reset_index(drop=True, inplace=True)
        self.data_df = data_df


    def threshold_at_zero(self, x):
        return x>0

    def __getitem__(self, index):
        image_path = self.data_df.loc[index, self.image_path_col] # read path to image from CSV.
        img = np.array(nib.load(image_path).get_fdata(), dtype=np.float32) # load NIFTI file into numpy array.

        #### HACK START
        # For now, we will take a cubic region of size subvolume_size^3 from the centre of each image.
        xmin, ymin, zmin = self.getSubvolumeOrigin(index)
        img = img[xmin:min(xmin+self.subvolume_size[0],img.shape[0]), ymin:min(ymin+self.subvolume_size[1],img.shape[1]), zmin:min(zmin+self.subvolume_size[2],img.shape[2])]
        #### HACK END
        
        label_path = self.data_df.loc[index, self.label_path_col]
        #print(label_path)
        labels = np.array(nib.load(label_path).get_fdata(), dtype=np.float32)
        
        #### HACK START
        labels = labels[xmin:min(xmin+self.subvolume_size[0],labels.shape[0]), ymin:min(ymin+self.subvolume_size[1],labels.shape[1]), zmin:min(zmin+self.subvolume_size[2],labels.shape[2])]
        #### HACK END
        # labels[labels==1]=41
        # labels[labels==2]=47
        # labels[labels==3]=46
        # labels[labels==4]=0
        # labels[labels==5]=43
        # labels[labels==6]=46
        # labels[labels==7]=40
        # labels[labels==8]=47
        # labels[labels==9]=45
        # labels[labels==10]=0
        # labels[labels==11]=42
        # labels[labels==12]=44

        Normalizer = monai.transforms.NormalizeIntensity()
        img = Normalizer(img)
        
        img = np.expand_dims(img, axis=0) # add dimension (axis=0) to store each modality (T1, T2 etc.).
        labels = np.expand_dims(labels, axis=0) # new inserted first dimension will store label channels once labels are one-hot encoded.

        return img, labels

    def __len__(self):
        return len(self.data_df)
    
    def getSubvolumeOrigin(self, index): # HACK FUNCTION FOR TEMPORARY FIX OF MEMORY ISSUE.
        """Get the origin of the cubic subvolume at the centre of the image. This is just a temporary solution to the memory issue."""

        image_path = self.data_df.loc[index, self.image_path_col]
        img = np.array(nib.load(image_path).get_fdata(), dtype=np.float32)

        xlen = img.shape[0]
        ylen = img.shape[1]
        zlen = img.shape[2]
        xmin = max(int((xlen-self.subvolume_size[0])/2),0)
        ymin = max(int((ylen-self.subvolume_size[1])/2),0)
        #zmin = max(int((zlen-self.subvolume_size[2])/2),0)
        zmin = zlen-self.subvolume_size[2]-int(self.subvolume_size[2]/7) #forv02
        #zmin = max(int((zlen-self.subvolume_size[2])/2)+int((zlen-self.subvolume_size[2])/4),0) #hack for v02 images
        return (xmin, ymin, zmin)


#### You can ignore the rest of this; it is just tests of the dataset class.

# if __name__ == '__main__':
    # ## Run some tests of this class. 
    # train_dataset = BrainDataset(split='train')
    # val_dataset = BrainDataset(split='val')

    # out_dir = '../runs/dataset_test'

    # # Get the mean and standard deviation of the training dataset for normalization.
    # pixels = np.array([])
    # # for index in range(len(train_dataset)):
    # #     img, labels = train_dataset[index]
    # #     #print(img.shape, labels.shape)
    # #     pixels = np.append(pixels, img.flatten())
    # # mean = np.mean(pixels)
    # # std = np.std(pixels)
    # # print('Training set mean:', mean)
    # # print('Training set std:', std)
    # print(len(train_dataset))

    # for index in range(len(train_dataset)):
    #     # Save an image to see how the transform is working.
    #     file_path = train_dataset.data_df.loc[index, 't1_brain_path'] #kat q: don't see this column in FileList.csv
    #     subject = train_dataset.data_df.loc[index, 'subject']
    #     out_name = subject + '_cropforeground.png' 
    #     out_path = os.path.join(out_dir, out_name)

    #     img, target = train_dataset[index]
    #     tempimg = img[0,:,:,45]
    #     print("SHAPE:", tempimg.shape)
    #     print("OUTPATH:", out_path)
    #     # print("IMAGE:", img)
    #     tempimg = np.asarray(tempimg)
    #     tempimg = Image.fromarray(tempimg)
    #     tempimg = tempimg.convert('L')
    #     tempimg.save(out_path)

    # data_df = pd.read_csv('/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data/file_lists/downsample/FileList_Size=10_Split=0.9-0.1-0.csv')
    # image_path = data_df.loc[0, 't1_brain_path'] # read path to image from CSV.
    # img = np.array(nib.load(image_path).get_fdata(), dtype=np.float32) # load NIFTI file into numpy array.
    # tempimg = img[45,:,:]
    # tempimg = np.asarray(tempimg)
    # tempimg = Image.fromarray(tempimg)
    # tempimg = tempimg.convert('L')
    #tempimg.save(os.path.join(out_dir, "testImage.png"))
    # for index, (img, target) in enumerate(train_dataset):
    #     print("INDEX:", index)
    #     print("SHAPE:", img.shape)
    #     print("TYPE:", type(img))
    #     # print("TARGET:", target)

    #     # Save an image to see how the transform is working.
    #     file_path = train_dataset.data_df.loc[index, 't1_brain_path'] #kat q: don't see this column in FileList.csv
    #     subject = train_dataset.data_df.loc[index, 'subject']
    #     out_name = subject + '.png' 
    #     out_path = os.path.join(out_dir, out_name)

    #     print("SUBJECT:", subject)
    #     print("FILEPATH:", file_path)
    #     # print("IMAGE:", img)
    #     print("MEAN:", np.mean(np.array(img)))
    #     print("STDEV:", np.std(np.array(img)))

    #     #np.squeeze(img, axis=(2,))
    #     #img[0,:,:,:].reshape(img.shape[1],img.shape[2],img.shape[3])
    #     tempimg = img[0,:,:,50]
    #     print("SHAPE:", tempimg.shape)
    #     print(out_path)
    #     tempimg = np.asarray(tempimg)
    #     tempimg = Image.fromarray(tempimg)
    #     tempimg = tempimg.convert('RGB')
    #     #tempimg = transforms.ToPILImage()(tempimg)
    #     tempimg.save(out_path)
    #     # img_nifti = nib.Nifti1Image(img[0,:,:,:]) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.
    #     # nib.save(img_nifti, out_path)


    #     #break
''''
image = np.array(
    [[[0, 0, 0, 0, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 3, 2, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 1, 1, 1],
      [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 3, 2, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 2, 1, 1],
      [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 3, 2, 0],
      [0, 1, 2, 1, 0],
      [0, 1, 1, 1, 1],
      [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image


def threshold_at_one(x):
    # threshold at 1
    print (x>1)
    return x > 1


cropper = monai.transforms.CropForeground(select_fn=threshold_at_one, margin=0)
print(cropper(image))
'''