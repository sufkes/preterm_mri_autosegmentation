import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.axes_grid1 import make_axes_locatable


from matplotlib import cm
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from dataset import BrainDataset
import monai
import nibabel as nib
import time
import json

from PIL import Image

def one_hot(label_batch, num_labels=12):
    """One-hot encode labels in a batch of label images.
    input shape : (batch_size, 1, H, W, D)
    output shape: (batch_size, num_labels+1, H, W, D)"""

    num_classes = num_labels + 1 # Treat background (label 0) as a separate class).
    label_batch = monai.networks.utils.one_hot(label_batch, num_classes, dim=1) # one-hot encode the labels    
    return label_batch

def calculate_confusion_matrix(index, yhat_temp, y):
    yhat_temp = yhat_temp.to('cpu')
    yhat_temp = np.array(yhat_temp)
    yhat_temp = yhat_temp[0, :, :, :, :]
    yhat_temp = np.argmax(yhat_temp, axis=0)

    y = y.to('cpu')
    y = np.array(y)
    y = y[0, :, :, :, :]
    y = np.argmax(y, axis=0)

    y = np.ravel(y)
    yhat_temp = np.ravel(yhat_temp)

    return confusion_matrix(y,yhat_temp)


def calculate_hausdorff_distance(index, yhat_temp, y):
    hausdorff_distance = monai.metrics.hausdorff_distance.compute_hausdorff_distance(yhat_temp, y.to('cpu'))
    hausdorff_distance = hausdorff_distance.cpu().numpy()[0]
    mean_hausdorff_distance = np.mean(hausdorff_distance)
    # print("HD", hausdorff_distance)
    # print("HD", mean_hausdorff_distance)
    hausdorff_distance = np.append(index, hausdorff_distance)
    
    return hausdorff_distance, mean_hausdorff_distance

def calculate_surface_distance(index, yhat_temp, y):
    surface_distance = monai.metrics.surface_distance.compute_average_surface_distance(yhat_temp, y.to('cpu'))
    surface_distance = surface_distance.cpu().numpy()[0]
    mean_surface_distance = np.mean(surface_distance)
    # print("HD", hausdorff_distance)
    # print("HD", mean_hausdorff_distance)
    surface_distance = np.append(index, surface_distance)
    
    return surface_distance, mean_surface_distance

def calculate_dice_score(index, yhat_temp, y):
    dice_score = monai.metrics.compute_meandice(yhat_temp,y.to('cpu'))
    dice_score = dice_score.cpu().numpy()[0]
    mean_dice_score = np.mean(dice_score)
    #print("DS", dice_score)
    dice_score = np.append(index, dice_score)
    #print("DS", mean_dice_score)
    return dice_score, mean_dice_score


def main(seed=489, out_dir='../runs/post_tests/V02_nomask_pretrainNoMask', run_dir='../runs/ubc/V02/nomask_pretrainNoMask'):
    all_files = os.listdir(run_dir)
    for file in all_files:
        if 'FileList' in file:
            file_name = file
        if '.txt' in file:
            config_file_name = file

    #Load Config File
    config_file = open(os.path.join(run_dir,config_file_name), 'r')
    config_file_content = config_file.read()
    config = json.loads(config_file_content)
    
    file_list_path = os.path.join(run_dir,file_name)
    model_path = os.path.join(run_dir,'checkpoints/best.pt')
    
    #### initialize other variables ####
    labels = ["Background","L Caudate","L Putamen","R Putamen","R Substantia Nigra","L Thalamus","R Globus Pallidus","R Caudate","L Globus Pallidus","L Substantia Nigra","R Thalamus"]            
    num_labels = config["num_labels"]
    subvolume_size = config["subvolume_size"]
    dropout_prob = config["dropout_prob"]


    ### Make datasets.
    train_dataset = BrainDataset(split='train', subvolume_size=subvolume_size, file_list_path=file_list_path)
    val_dataset = BrainDataset(split='val', subvolume_size=subvolume_size, file_list_path=file_list_path)
    test_dataset = BrainDataset(split='test', subvolume_size=subvolume_size, file_list_path=file_list_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # #### Use model to predict results.
    # Load the model weights which minimized validation loss.
    model = monai.networks.nets.HighResNet(spatial_dims=3,
                                           in_channels=1,
                                           out_channels=num_labels+1,
                                           dropout_prob=dropout_prob
                                       )
    model.to(device)
    best_state_dict = torch.load(model_path)
    model.load_state_dict(best_state_dict)

    dataset_dict = {}
    # dataset_dict['test'] = test_dataset
    dataset_dict['train'] = train_dataset
    # dataset_dict['val'] = val_dataset
    

    model.eval() # Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results (https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
    
    # metrics = pd.DataFrame()
    # metrics = metrics.append(pd.Series(dtype='object'), ignore_index=True)
    # i = 0
    with torch.no_grad(): # No need to store gradients when doing inference.
        for split, dataset in dataset_dict.items():
            split_dir = os.path.join(out_dir,split)
            predicted_dir = os.path.join(split_dir, 'predicted_labels')
            os.makedirs(predicted_dir,exist_ok=True)
            diff_dir = os.path.join(split_dir, 'diff_labels')
            os.makedirs(diff_dir,exist_ok=True)
            cm_dir = os.path.join(split_dir, 'confusion_matrix')
            os.makedirs(cm_dir,exist_ok=True)
            metrics = pd.DataFrame()
            metrics = metrics.append(pd.Series(dtype='object'), ignore_index=True)
            i = 0
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # set batch_size to 1 since we want to make predictions for one image at a time, saving the result each time.
            try:
                del all_dice_scores
                del all_hausdorff_distance
                del all_surface_distance
            except:
                print("")

            for index, (img, y) in enumerate(dataloader):
                
                print("Index:",index)
                label_path = dataset.data_df.loc[index, dataset.label_path_col] # path to original label file.
                # Predict labels using best model.
                img = img.to(device)
                
                yhat = model(img) # shape (B=1, num_labels+1, H, W, D)
                y = one_hot(y, num_labels)
                y = y.to(device) 
 
                yhat_temp = yhat.to('cpu')
                yhat_temp = np.array(yhat_temp)
                yhat_temp = yhat_temp[0, :, :, :, :]
                yhat_temp = np.argmax(yhat_temp, axis=0)
                yhat_temp = np.expand_dims(np.expand_dims(yhat_temp, axis=0),axis=0)
                yhat_temp = torch.tensor(yhat_temp)
                yhat_temp = one_hot(yhat_temp, num_labels)
                
                dice_score, mean_dice_score = calculate_dice_score(index, yhat_temp, y)
                hausdorff_distance, mean_hausdorff_distance = calculate_hausdorff_distance(index, yhat_temp, y)
                surface_distance, mean_surface_distance = calculate_surface_distance(index, yhat_temp, y)


                cmx = calculate_confusion_matrix(index, yhat_temp, y)

                

                try:
                    all_dice_scores= np.vstack([all_dice_scores, dice_score])
                    all_hausdorff_distance = np.vstack([all_hausdorff_distance, hausdorff_distance])
                    all_surface_distance = np.vstack([all_surface_distance, surface_distance])
                except:
                    all_dice_scores = dice_score
                    all_hausdorff_distance = hausdorff_distance
                    all_surface_distance = surface_distance
                
                metrics.loc[i, 'image_index'] = index
                metrics.loc[i, 'mean_dice_score'] = mean_dice_score
                metrics.loc[i, 'mean_hausdorff_distance'] = mean_hausdorff_distance
                metrics.loc[i, 'mean_surface_distance'] = mean_surface_distance

                ## We want to save the predicted labels as a NIFTI file.
                yhat = yhat.to('cpu')
                yhat = np.array(yhat)
                

                yhat = yhat[0, :, :, :, :] # shape (num_labels+1, H, W, D) # get first item in batch (batch size is 1 here)
                yhat = np.argmax(yhat, axis=0) # shape (H, W, D) # Convert from one-hot-encoding back to labels.
                labels_orig = nib.load(label_path) # Load the original labels NIFTI file to use as a template.
                label_pred_data = np.zeros(shape=labels_orig.get_fdata().shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 

                #### HACK: Fill with predicted values in subvolume cube in which prediction were made.
                xmin, ymin, zmin = dataset.getSubvolumeOrigin(index)
                label_pred_data[xmin:xmin+subvolume_size[0], ymin:ymin+subvolume_size[1], zmin:zmin+subvolume_size[2]] = yhat
                label_pred_nifti = nib.Nifti1Image(label_pred_data, labels_orig.affine, labels_orig.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

                # Save predicted labels image as a NIFTI file.
                out_name = os.path.basename(label_path)
                out_path = os.path.join(predicted_dir, out_name)
                nib.save(label_pred_nifti, out_path)

                diff_image = np.where(np.argmax(np.array(y.to('cpu'))[0,:,:,:,:],axis=0) == yhat, 1, 0)
                diff_data = np.zeros(shape=labels_orig.get_fdata().shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 
                    
                #### HACK: Fill with predicted values in subvolume cube in which prediction were made.
                diff_data[xmin:xmin+subvolume_size[0], ymin:ymin+subvolume_size[1], zmin:zmin+subvolume_size[2]] = diff_image
                diff_data_nifti = nib.Nifti1Image(diff_data, labels_orig.affine, labels_orig.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

                # Save predicted labels image as a NIFTI file.
                out_name = os.path.basename(label_path)
                out_name = 'diff_image_'+out_name
                out_path = os.path.join(diff_dir, out_name)
                nib.save(diff_data_nifti, out_path)

                #Save confusion matrix
                out_name = os.path.basename(label_path).split('.')[0]
                df_cm = pd.DataFrame(cmx, range(num_labels+1), range(num_labels+1))
                #plt.figure(figsize = (20,15))
                fig = plt.figure(figsize = (13,12))
                sn.set(font_scale=1.25)
                
                ax = sn.heatmap(df_cm, annot=True, annot_kws={"fontsize":12}, fmt='g', cmap = "Blues", vmin=0, vmax=1000, cbar_kws={"shrink": .82})
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                # im = ax.imshow(np.arange(100).reshape((10,10)))

                plt.yticks(rotation=0, fontsize=9) 
                plt.xticks(rotation=90, fontsize=9) 
                plt.xlabel("True Label", fontsize=12)
                plt.ylabel("Predicted Label", fontsize=12) 
                #plt.title("Confusion matrix for "+out_name, fontsize=16)
                plt.title("UBC V02 confusion matrix (pretrained with dHCP model)", fontsize=16) #pretrained with dHCP model
                plt.imshow(df_cm)
                #plt.colorbar(im, cax=cax)
                plt.savefig(os.path.join(cm_dir,out_name))

                i+=1

            metrics_name = 'metrics.csv'
            metrics_path = os.path.join(split_dir, metrics_name)
            metrics.to_csv(metrics_path, index=False)

            dice_score_name = 'dice_scores.csv'
            dice_score_path = os.path.join(split_dir, dice_score_name)
            np.savetxt(dice_score_path, all_dice_scores, delimiter=",")

            hausdorff_distance_name = 'hausdorff_distance.csv'
            hausdorff_distance_path = os.path.join(split_dir, hausdorff_distance_name)
            np.savetxt(hausdorff_distance_path, all_hausdorff_distance, delimiter=",")

            surface_distance_name = 'surface_distance.csv'
            surface_distance_path = os.path.join(split_dir, surface_distance_name)
            np.savetxt(surface_distance_path, all_surface_distance, delimiter=",")


if __name__ == '__main__':
    main()

