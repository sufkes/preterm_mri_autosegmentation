import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


from dataset import BrainDataset
import monai
import nibabel as nib
import time

from PIL import Image

def one_hot(label_batch, num_labels=12):
    """One-hot encode labels in a batch of label images.
    input shape : (batch_size, 1, H, W, D)
    output shape: (batch_size, num_labels+1, H, W, D)"""

    num_classes = num_labels + 1 # Treat background (label 0) as a separate class).
    label_batch = monai.networks.utils.one_hot(label_batch, num_classes, dim=1) # one-hot encode the labels    
    return label_batch

def calculate_hausdorff_distance(index, yhat_temp, y):
    hausdorff_distance = monai.metrics.hausdorff_distance.compute_hausdorff_distance(yhat_temp, y.to('cpu'))
    hausdorff_distance = hausdorff_distance.cpu().numpy()[0]
    mean_hausdorff_distance = np.mean(hausdorff_distance)
    # print("HD", hausdorff_distance)
    # print("HD", mean_hausdorff_distance)
    hausdorff_distance = np.append(index, hausdorff_distance)
    
    return hausdorff_distance, mean_hausdorff_distance

def calculate_dice_score(index, yhat_temp, y):
    dice_score = monai.metrics.compute_meandice(yhat_temp,y.to('cpu'))
    dice_score = dice_score.cpu().numpy()[0]
    mean_dice_score = np.mean(dice_score)
    #print("DS", dice_score)
    dice_score = np.append(index, dice_score)
    #print("DS", mean_dice_score)
    return dice_score, mean_dice_score
                

def main(seed=489, out_dir='../runs/tests/testing_label_9'):
    dhcp_parameters = {
        "num_labels":87,
        "subvolume_size":[98,125,101],
        "dropout_prob": 0.0}
    ubc_parameters = {
        "num_labels":12,
        "subvolume_size":[72,100,120],
        "dropout_prob": 0.0}
    transfer = {
        "num_labels":87,
        "subvolume_size":[72,100,120],
        "dropout_prob": 0.0}

    # file_list_name = 'dhcp/FileList_dhcp_Size=30_Split=0.9-0.1-0.0.csv'
    # model_name = 'dhcp_models/vl=1367_nl=87_ss=[98,125,101].pt'
    # parameters = dhcp_parameters

    file_list_name = 'ubc/masks/V02/FileList_ubc_Size=173_Split=0.9-0.075-0.025.csv'
    model_name = 'ubc_models/vl=4472_nl=12_ss=[72,100,120].pt'
    parameters = ubc_parameters

    # file_list_name = 'ubc/FileList_ubc_Size=30_Split=0.9-0.1-0.0.csv'
    # model_name = 'ubc_models/vl=4472_nl=12_ss=[72,100,120].pt'
    # out_dir='../runs/tests/testing_metrics_2'
    # parameters = ubc_parameters
    
    
    #### initialize other variables ####
    num_labels = parameters["num_labels"]
    subvolume_size = parameters["subvolume_size"]
    dropout_prob = parameters["dropout_prob"]
    model_dir = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/trained_models'
    file_list_dir = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/file_lists'
    model_path = os.path.join(model_dir,model_name)
    file_list_path = os.path.join(file_list_dir,file_list_name)

    predicted_dir = os.path.join(out_dir, 'predicted_labels')
    os.makedirs(predicted_dir,exist_ok=True)

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
    # best_state_dict["blocks.11.conv.weight"] = best_state_dict["blocks.11.0.conv.weight"]
    # best_state_dict["blocks.11.conv.bias"] = best_state_dict["blocks.11.0.conv.bias"]
    # best_state_dict["blocks.11.adn.N.bias"] = best_state_dict["blocks.11.0.adn.N.bias"]
    # best_state_dict["blocks.11.adn.N.weight"] = best_state_dict["blocks.11.0.adn.N.weight"]
    # best_state_dict["blocks.11.adn.N.running_mean"] = best_state_dict["blocks.11.0.adn.N.running_mean"]
    # best_state_dict["blocks.11.adn.N.running_var"] = best_state_dict["blocks.11.0.adn.N.running_var"]
    # best_state_dict.pop("blocks.11.0.conv.weight", None)
    # best_state_dict.pop('blocks.11.0.conv.bias', None)
    # best_state_dict.pop('blocks.11.0.adn.N.bias', None)
    # best_state_dict.pop('blocks.11.0.adn.N.weight', None)
    # best_state_dict.pop('blocks.11.0.adn.N.running_mean', None)
    # best_state_dict.pop('blocks.11.0.adn.N.running_var', None)
    # best_state_dict.pop("blocks.11.0.adn.N.num_batches_tracked", None)

    model.load_state_dict(best_state_dict)

    

    dataset_dict = {}
    dataset_dict['train'] = train_dataset
    dataset_dict['val'] = val_dataset
    #dataset_dict['test'] = test_dataset

    model.eval() # Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results (https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
    
    metrics = pd.DataFrame()
    metrics = metrics.append(pd.Series(dtype='object'), ignore_index=True)
    i = 0
    with torch.no_grad(): # No need to store gradients when doing inference.
        #all_dice_scores = []
        for split, dataset in dataset_dict.items():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # set batch_size to 1 since we want to make predictions for one image at a time, saving the result each time.
   
            for index, (img, y) in enumerate(dataloader):
                
                print("Index:",index)
                label_path = dataset.data_df.loc[index, dataset.label_path_col] # path to original label file.
                # Predict labels using best model.
                img = img.to(device)
                
                yhat = model(img) # shape (B=1, num_labels+1, H, W, D)
                y = one_hot(y, num_labels)
                y = y.to(device) 
                
              


                # print("YHAT",yhat[0, :, :10, :10, :10])
                # print("GROUND",y[0, :, :10, :10, :10])
                yhat_temp = yhat.to('cpu')
                yhat_temp = np.array(yhat_temp)
                yhat_temp = yhat_temp[0, :, :, :, :]
                yhat_temp = np.argmax(yhat_temp, axis=0)
                yhat_temp = np.expand_dims(np.expand_dims(yhat_temp, axis=0),axis=0)
                yhat_temp = torch.tensor(yhat_temp)
                yhat_temp = one_hot(yhat_temp, num_labels)
                
                dice_score, mean_dice_score = calculate_dice_score(index, yhat_temp, y)
                hausdorff_distance, mean_hausdorff_distance = calculate_hausdorff_distance(index, yhat_temp, y)

                # confusion_matrix = monai.metrics.compute_confusion_metric(yhat_temp,y.to('cpu'))
                # df_cm = pd.DataFrame(confusion_matrix, range(num_labels), range(num_labels))
                # plt.figure(figsize = (10,7))
                # sn.heatmap(df_cm, annot=True)
                # plt.imshow(df_cm)
                # plt.savefig(os.path.join(out_dir,'test.png'))
                # print("HD", mean_hausdorff_distance)
                # print("DS", mean_dice_score)
                try:
                    all_dice_scores= np.vstack([all_dice_scores, dice_score])
                    all_hausdorff_distance = np.vstack([all_hausdorff_distance, hausdorff_distance])
                except:
                    all_dice_scores = dice_score
                    all_hausdorff_distance = hausdorff_distance
                
                metrics.loc[i, 'image_index'] = index
                metrics.loc[i, 'mean_dice_score'] = mean_dice_score
                metrics.loc[i, 'mean_hausdorff_distance'] = mean_hausdorff_distance

                ## We want to save the predicted labels as a NIFTI file.
                yhat = yhat.to('cpu')
                yhat = np.array(yhat)
                

                yhat = yhat[0, :, :, :, :] # shape (num_labels+1, H, W, D) # get first item in batch (batch size is 1 here)
                yhat = np.argmax(yhat, axis=0) # shape (H, W, D) # Convert from one-hot-encoding back to labels.
                labels_orig = nib.load(label_path) # Load the original labels NIFTI file to use as a template.
                label_pred_data = np.zeros(shape=labels_orig.get_fdata().shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 
                # print("ARGMAX YHAT",yhat[:10, :10, :10])
                # print("ARGMAX Y",np.argmax(np.array(y.to('cpu')),axis=0)[:10, :10, :10])
                #### HACK: Fill with predicted values in subvolume cube in which prediction were made.
                xmin, ymin, zmin = dataset.getSubvolumeOrigin(index)
                label_pred_data[xmin:xmin+subvolume_size[0], ymin:ymin+subvolume_size[1], zmin:zmin+subvolume_size[2]] = yhat
                label_pred_nifti = nib.Nifti1Image(label_pred_data, labels_orig.affine, labels_orig.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

                # Save predicted labels image as a NIFTI file.
                out_name = os.path.basename(label_path)
                out_path = os.path.join(predicted_dir, out_name)
                nib.save(label_pred_nifti, out_path)

                # hausdorff_distance = monai.metrics.compute_hausdorff_distance(y, label_pred_data, index)
                # print(hausdorff_distance)
                i+=1

    metrics_name = 'metrics.csv'
    metrics_path = os.path.join(out_dir, metrics_name)
    metrics.to_csv(metrics_path, index=False)
    dice_score_name = 'dice_scores.csv'
    dice_score_path = os.path.join(out_dir, dice_score_name)
    np.savetxt(dice_score_path, all_dice_scores, delimiter=",")

    hausdorff_distance_name = 'hausdorff_distance.csv'
    hausdorff_distance_path = os.path.join(out_dir, hausdorff_distance_name)
    np.savetxt(hausdorff_distance_path, all_dice_scores, delimiter=",")



if __name__ == '__main__':
    main()

