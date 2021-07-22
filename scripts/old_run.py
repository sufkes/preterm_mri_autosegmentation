#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import highres_modified
import torch.nn as nn

from dataset import BrainDataset
import monai
import nibabel as nib
import json
import time

def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
    
def one_hot(label_batch, num_labels):
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


def transfer_model(model, num_labels, spatial_dims, dropout_prob):
    blocks = list(model.blocks)
    temp = []
    for i in range(len(blocks)-1):
        temp.append(blocks[i])
    modified_block = monai.networks.blocks.Convolution(
            dimensions= spatial_dims,
            in_channels= temp[len(blocks)-2].out_channels,
            out_channels = num_labels+1,
            kernel_size= 1,
            adn_ordering="NAD",
            act= ("relu", {"inplace": True}),
            norm= ("batch", {"affine": True}),
            dropout=dropout_prob,
    )
    temp.append(modified_block)
    model.blocks = nn.Sequential(*temp)
    return model

def train_loop(dataloader, model, loss_fn, optimizer, device, num_labels, verbose=False, amp=False):
    size = len(dataloader.dataset)
    total_loss = 0

    model.train()
    i = 0
    for batch, (X, y) in enumerate(dataloader):
        scaler = torch.cuda.amp.GradScaler() if amp else None
        
        X = X.to(device) # image.  shape: (batch_size, color_channels, H, W, D)
        y = one_hot(y, num_labels)   # labels. shape: (batch_size, 1, H, W, D) -> (batch_size, num_labels+1, H, W, D)
        y = y.to(device) 

        
        # Compute prediction and loss
        #pred = model(X)

        # print(y.shape, batch)
        # print(pred.shape, batch)
        # loss = loss_fn(pred, y)

        # if amp and scaler is not None:
        #     with torch.cuda.amp.autocast():
        #         pred = model(X)
        #         loss = loss_fn(pred, y)
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad() # Set the gradients back to zero (from the previous loop).
        loss.backward() # Compute the gradients.
        optimizer.step() # Update the parameters.

        total_loss += loss.item() * y.shape[0] # (mean Dice coef. of batch) * (size of batch); since batch size is 1 here, this is the same as loss.item(), but need to be careful if batch_size>1.
        
    mean_loss = total_loss/size # mean loss over training set
    if verbose:
        print(f'Mean loss (training)  : {mean_loss:>8f}')

    return mean_loss

def val_loop(dataloader, model, loss_fn, device, num_labels, verbose=False): # same as train_loop() except use model.eval() and torch.no_grad()
    size = len(dataloader.dataset)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = one_hot(y, num_labels)
            y = y.to(device)
            
            pred = model(X)            
            loss = loss_fn(pred, y)

            total_loss += loss.item() * y.shape[0]

    mean_loss = total_loss/size
    if verbose:
        print(f"Mean loss (validation) : {mean_loss:>8f}")
    
    return mean_loss

def main(seed=489, out_dir='../runs/tests/008'):
    ### Set Parameters
    use_config_file = True
    if(use_config_file): ## Set Parameters from Config File
        #Load Config File
        config_file_name = sys.argv[1]
        config_file = open(config_file_name, 'r')
        config_file_content = config_file.read()
        config = json.loads(config_file_content)

        #Run Parameters
        file_list_path = config['file_list_path']
        print(file_list_path)
        epochs = config['epochs']
        out_dir = config['out_dir']
    
        #Dataset Parameters
        num_labels = config['num_labels']
        subvolume_size = config['subvolume_size']

        #Resume/Transfer Parameters
        resume = config['resume_flag']
        #transfer_resume = config['transfer_resume_flag']
        transfer_train = config['transfer_train_flag']
        # new_num_labels = config['new_num_labels']
        # old_num_labels = config['old_num_labels']
        load_checkpoint_path = config['load_checkpoint_path']

        #Hyperparameters
        dropout_prob = config['dropout_prob']
        batch_size = config['batch_size']
        shuffle = config['shuffle']
        drop_last = config['drop_last']
        loss_name = config['loss_name']
        optimizer_name = config['optimizer_name']
        learning_rate = config['learning_rate']
        weight_decay = config['weight_decay']
        note = ''
        
    else: ## Set Parameters Inline
        dhcp_parameters = {
            "num_labels":87,
            "subvolume_size":[98,125,101],
            "dropout_prob": 0.0,}
        ubc_parameters = {
            "num_labels":12,
            "subvolume_size":[72,100,120],
            "dropout_prob": 0.0}
        test_parameters = {
            "num_labels":87,
            "subvolume_size":[72,100,120],
            "dropout_prob": 0.0}
        parameters = test_parameters
       
        ## File Path
        file_list_name = 'downsample/FileList_Size=100_Split=0.9-0.1-0.csv'
        file_list_dir = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data/file_lists'
        file_list_path = os.path.join(file_list_dir,file_list_name)
        #### Set hyperparameters. These will be saved to a file to keep track of test settings and results.    
        epochs = 36
        note = '' # a special note about this run. 
        num_labels = parameters["num_labels"] ## Number of Classes (The number of classes predicted by the model will be 1 more than this, since the background is treated as a class)
        ## Training loop settings:
        optimizer_name = 'Adam'
        learning_rate = 0.01 # (value in HiResNet paper = 0.01)
        weight_decay = 0.0
        loss_name = 'DiceLoss' ## Loss function settings
        ## Dataloader settings:
        subvolume_size = parameters["subvolume_size"] # For now, model only predicts for cube in the middle of the image of size (subvolume_size x subvolume_size x subvolume_size).
        ## Model settings:
        dropout_prob = parameters["dropout_prob"]
        batch_size = 1
        shuffle = True
        drop_last = False
         ## Resume Settings
        
        resume = False
        #transfer_resume = False
        transfer_train = True
        #if(transfer_train or transfer_resume):
        if(transfer_train):
            num_labels = 87 #original shape #12
            # new_num_labels = 87 #12
            # old_num_labels = 87
        #if(resume or transfer_resume or transfer_train):
        if(resume or transfer_train):
            load_checkpoint_path = '../runs/trained_models/checkpoint_5.pt'

    ## Print Parameters
    print('*****CONFIRM THE FOLLOWING PARAMETERS*****')
    print('EPOCHS        :', epochs)
    print('OUT_DIR       :', out_dir)
    print('FILE_LIST     :', file_list_path.split('/')[-1])
    print('NUM_LABELS    :', num_labels)
    print('SUBVOLUME_SIZE:', subvolume_size)
    #if(resume or transfer_resume):
    if(resume):
        print('****************RESUMING******************')
        print('CHECKPOINT    :', load_checkpoint_path)
    if(transfer_train):
        print('************TRANSFER LEARNING*************')
        print('CHECKPOINT    :', load_checkpoint_path)

    try:
        if(sys.argv[2]=='True'):
            if input("(y/n): ") != 'y':
                exit()
    except:
        if input("(y/n): ") != 'y':
            exit()

    ## Set seeds for determinism.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    #random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    monai.utils.set_determinism(seed)    

    ## Create output directory
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok=True)
    predicted_dir = os.path.join(out_dir, 'predicted_labels')
    os.makedirs(predicted_dir,exist_ok=True)
    
    ## Copy Relevant Files to Out Directory
    if use_config_file:
        os.system('cp '+config_file_name+' '+out_dir)
    os.system('cp '+file_list_path+' '+out_dir)             

    #### Select device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    #### Make datasets.
    train_dataset = BrainDataset(split='train', subvolume_size=subvolume_size, file_list_path=file_list_path)
    val_dataset = BrainDataset(split='val', subvolume_size=subvolume_size, file_list_path=file_list_path)
    #test_dataset = BrainDataset(split='test', subvolume_size=subvolume_size)

    ## Record the image dimensions.
    image_shape = train_dataset[0][0].shape
    dim_x = image_shape[1]
    dim_y = image_shape[2]
    dim_z = image_shape[3]

    ## Make dataloaders.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False, # no point in shuffling the validation set
                                drop_last=False)
    #test_dataloader = DataLoader(test_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             drop_last=False)


    #### Set model and hyperparameters
    ## Set model.
    spatial_dims = 3
    in_channels = 1
    # if transfer_train or transfer_resume:
    #     model = monai.networks.nets.HighResNet(spatial_dims=spatial_dims,
    #                                         in_channels=in_channels,
    #                                         out_channels=old_num_labels+1,
    #                                         dropout_prob=dropout_prob
    #                                     )
    # else:
    model = monai.networks.nets.HighResNet(spatial_dims=spatial_dims,
                                        in_channels=in_channels,
                                        out_channels=num_labels+1,
                                        dropout_prob=dropout_prob
                                    )
    # model.blocks      
    # model = highres_modified.HighResNet(spatial_dims=3,
    #                                     in_channels=1,
    #                                     out_channels=num_labels+1,
    #                                     dropout_prob=dropout_prob
    # )
    model.to(device)
    
    ## AMP
    # set_determinism(seed=0)
    # amp_start = time.time()
    # (
    #     max_epochs,
    #     amp_epoch_loss_values,
    #     amp_metric_values,
    #     amp_epoch_times,
    #     amp_best,
    # ) = train_loop(amp=True)
    # amp_total_time = time.time() - amp_start
    # print(
    #     f"total training time of {max_epochs} "
    #     f"epochs with AMP: {amp_total_time:.4f}"
    # )      
                                          
    ## Set loss function.
    losses = {'DiceLoss':monai.losses.DiceLoss(softmax=True), # I think this is 1-Dice, so we want to minimize it.
              'GeneralizedDiceLoss':monai.losses.GeneralizedDiceLoss(softmax=True),
          }
    loss_fn = losses[loss_name]


    ## Set optimizer.
    optimizers = {'SGD':torch.optim.SGD(model.parameters(), lr=learning_rate),
                  'Adam':torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False),
                  'AdamW':torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)}
    optimizer = optimizers[optimizer_name]
    #print("BEFORE",model)
    before_file = open("before.txt",'w')
    before_file.write(str(model))
    before_file.close

    #Resume
    if resume:
        model, optimizer, start_epoch = load_checkpoint(load_checkpoint_path, model, optimizer)
    # elif transfer_resume:
    #     model, optimizer, start_epoch = load_checkpoint(load_checkpoint_path, model, optimizer)
    #     model.blocks[len(model.blocks)-1] = monai.networks.blocks.Convolution(
    #         dimensions= spatial_dims,
    #         in_channels= model.blocks[len(model.blocks)-2].out_channels,
    #         out_channels = num_labels+1,
    #         kernel_size= 1,
    #         adn_ordering="NAD",
    #         act= ("relu", {"inplace": True}),
    #         norm= ("batch", {"affine": True}),
    #         dropout=dropout_prob,
    #     )
    elif transfer_train:
        #model, optimizer, start_epoch = load_checkpoint(load_checkpoint_path, model, optimizer)
        best_state_dict = torch.load(load_checkpoint_path)
        #model.load_state_dict(best_state_dict)
        start_epoch = 0
        # # model.fc1.weight.copy_(best_state_dict['fc1.weight'])
        # # model.fc1.bias.copy_(best_state_dict['fc1.bias'])
        # print(best_state_dict.keys())
        # # model['blocks.0.conv.weight']=best_state_dict['blocks.0.conv.weight']
        # #model.blocks[0].weight.copy_(best_state_dict['blocks.0.conv.weight'])
        # print("1: ",len(model.blocks))
        # last = nn.Linear(list(model.children())[-1][-1].in_channels, new_num_labels)
        # print("LAST",last)
        # block = nn.Sequential(*list(model.children())[-1][:-1], last)
        # print("BLOCK",block)      
        # model = nn.Sequential(*list(model.children())[:-1], block)
        # print("MODEL", model)
        # #model.blocks.conv.weight.copy_(best_state_dict['block.0'])
        # print("2: ",len(model))
        # print(model)
        # print(old_num_labels)
        # print(best_state_dict.keys())
        with torch.no_grad():
            model.blocks[0].conv.weight.copy_(best_state_dict['blocks.0.conv.weight'])
            model.blocks[0].conv.bias.copy_(best_state_dict['blocks.0.conv.bias'])
            
            for i in range(9):
                model.blocks[i+1].layers[1].conv.weight.copy_(best_state_dict['blocks.'+str(i+1)+'.layers.1.conv.weight'])
                model.blocks[i+1].layers[1].conv.bias.copy_(best_state_dict['blocks.'+str(i+1)+'.layers.1.conv.bias'])
                model.blocks[i+1].layers[3].conv.weight.copy_(best_state_dict['blocks.'+str(i+1)+'.layers.3.conv.weight'])
                model.blocks[i+1].layers[3].conv.bias.copy_(best_state_dict['blocks.'+str(i+1)+'.layers.3.conv.bias'])
            # model.blocks[1].layers[1].conv.weight.copy_(best_state_dict['blocks.1.conv.weight'])
            # model.blocks[1].layers[1].conv.bias.copy_(best_state_dict['blocks.1.conv.bias'])
            # model.blocks[1].layers[3].conv.weight.copy_(best_state_dict['blocks.1.conv.weight'])
            # model.blocks[1].layers[3].conv.bias.copy_(best_state_dict['blocks.1.conv.bias'])
            # model.blocks[2].conv.weight.copy_(best_state_dict['blocks.2.conv.weight'])
            # model.blocks[2].conv.bias.copy_(best_state_dict['blocks.2.conv.bias'])
            # model.blocks[3].conv.weight.copy_(best_state_dict['blocks.3.conv.weight'])
            # model.blocks[3].conv.bias.copy_(best_state_dict['blocks.3.conv.bias'])
            # model.blocks[4].conv.weight.copy_(best_state_dict['blocks.4.conv.weight'])
            # model.blocks[4].conv.bias.copy_(best_state_dict['blocks.4.conv.bias'])
            # model.blocks[5].conv.weight.copy_(best_state_dict['blocks.5.conv.weight'])
            # model.blocks[5].conv.bias.copy_(best_state_dict['blocks.5.conv.bias'])
            # model.blocks[6].conv.weight.copy_(best_state_dict['blocks.6.conv.weight'])
            # model.blocks[6].conv.bias.copy_(best_state_dict['blocks.6.conv.bias'])
            # model.blocks[7].conv.weight.copy_(best_state_dict['blocks.7.conv.weight'])
            # model.blocks[7].conv.bias.copy_(best_state_dict['blocks.7.conv.bias'])
            # model.blocks[8].conv.weight.copy_(best_state_dict['blocks.8.conv.weight'])
            # model.blocks[8].conv.bias.copy_(best_state_dict['blocks.8.conv.bias'])
            # model.blocks[9].conv.weight.copy_(best_state_dict['blocks.9.conv.weight'])
            # model.blocks[9].conv.bias.copy_(best_state_dict['blocks.9.conv.bias'])
            model.blocks[10].conv.weight.copy_(best_state_dict['blocks.10.conv.weight'])
            model.blocks[10].conv.bias.copy_(best_state_dict['blocks.10.conv.bias'])
            # model.blocks[11].conv.weight.copy_(best_state_dict['blocks.11.conv.weight'])
            # model.blocks[11].conv.bias.copy_(best_state_dict['blocks.11.conv.bias'])

            

        # model.blocks[len(model.blocks)-1] = monai.networks.blocks.Convolution(
        #     dimensions= spatial_dims,
        #     in_channels= model.blocks[len(model.blocks)-2].out_channels,
        #     out_channels = new_num_labels+1,
        #     kernel_size= 1,
        #     adn_ordering="NAD",
        #     act= ("relu", {"inplace": True}),
        #     norm= ("batch", {"affine": True}),
        #     dropout=dropout_prob,
        # )
        # model = transfer_model(model, new_num_labels, old_num_labels, spatial_dims, dropout_prob)
    else:
        start_epoch = 0

    model.to(device)
    #print("AFTER",model)
    after_file = open("after.txt",'w')
    after_file.write(str(model))
    after_file.close

    ## Create spreadsheet to store loss vs. epoch
    loss_df = pd.DataFrame(columns=['train', 'val'])
    loss_df.index.name = 'epoch'
    
    hyper_verb_name = 'hyperparameters_verbose.csv'
    hyper_verb_path = os.path.join(out_dir, hyper_verb_name)
    if os.path.exists(hyper_verb_path):
        hyper_verb_df = pd.read_csv(hyper_verb_path)
    else:
        hyper_verb_df = pd.DataFrame()

    # Begin timer
    start_time = time.time()
    model.train()
    #### Training and validation loop
    val_mean_loss_min = np.inf
    for t in range(start_epoch, epochs):
        print(f"\nEpoch {t+1}")
        train_mean_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device, num_labels)
        val_mean_loss = val_loop(val_dataloader, model, loss_fn, device, num_labels)
        #test_mean_loss = val_loop(test_dataloader, model, loss_fn, device)

        loss_df.loc[t+1, ['train', 'val']] = [train_mean_loss, val_mean_loss]

        print(f"Training         : {train_mean_loss}\nValidation       : {val_mean_loss}")
        
        if val_mean_loss < val_mean_loss_min:
            val_mean_loss_min = val_mean_loss
            train_mean_loss_at_val_min = train_mean_loss
            best_epoch = t+1
            print(f"* * * New best validation result * * *")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pt'))
       
        ## Save loss vs. epoch
        loss_df_name = 'loss.csv'
        loss_path = os.path.join(out_dir, loss_df_name)
        loss_df.to_csv(loss_path, index=True)
    
        hyper_verb_df = hyper_verb_df.append(pd.Series(dtype='object'), ignore_index=True)
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'out_dir'] = out_dir
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'dim_x'] = dim_x
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'dim_y'] = dim_y
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'dim_z'] = dim_z
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'subvolume_size'] = str(subvolume_size[0]) + ' ' + str(subvolume_size[1]) + ' ' + str(subvolume_size[2]) 
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'epochs'] = epochs
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'weight_decay'] = weight_decay
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'batch_size'] = batch_size
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'dropout_prob'] = dropout_prob
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'best_epoch'] = best_epoch
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'mse_mean_val'] = val_mean_loss_min
        hyper_verb_df.loc[hyper_verb_df.index[-1], 'mse_mean_train'] = train_mean_loss_at_val_min
        
        # Save
        hyper_verb_df.to_csv(hyper_verb_path, index=False)  

        if(t%3==0 or t==epochs-1):
            checkpoint = {
                'epoch': t + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            } 
            checkpoint_path = os.path.join(checkpoint_dir,'checkpoint_'+str(t+1)+'.pt')
            #checkpoint_path = os.path.join(checkpoint_dir,'checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
            
    end_time = time.time()

    print(f"Done training.\nBest validation loss: {val_mean_loss_min}")
    print("Time elapsed:",end_time-start_time)

    ## Save hyperparameters and losses to a spreadsheet.
    hyper_name = '../hyperparameters.csv'
    hyper_path = os.path.join(out_dir, hyper_name)
    if os.path.exists(hyper_path):
        hyper_df = pd.read_csv(hyper_path)
    else:
        hyper_df = pd.DataFrame()
    hyper_df = hyper_df.append(pd.Series(dtype='object'), ignore_index=True)
    hyper_df.loc[hyper_df.index[-1], 'dim_x'] = dim_x
    hyper_df.loc[hyper_df.index[-1], 'dim_y'] = dim_y
    hyper_df.loc[hyper_df.index[-1], 'dim_z'] = dim_z
    hyper_df.loc[hyper_df.index[-1], 'subvolume_size'] = str(subvolume_size[0]) + ' ' + str(subvolume_size[1]) + ' ' + str(subvolume_size[2]) 
    hyper_df.loc[hyper_df.index[-1], 'note'] = note
    hyper_df.loc[hyper_df.index[-1], 'epochs'] = epochs
    hyper_df.loc[hyper_df.index[-1], 'loss_name'] = loss_name
    hyper_df.loc[hyper_df.index[-1], 'learning_rate'] = learning_rate
    hyper_df.loc[hyper_df.index[-1], 'weight_decay'] = weight_decay
    hyper_df.loc[hyper_df.index[-1], 'batch_size'] = batch_size
    hyper_df.loc[hyper_df.index[-1], 'shuffle'] = shuffle
    hyper_df.loc[hyper_df.index[-1], 'drop_last'] = drop_last
    hyper_df.loc[hyper_df.index[-1], 'optimizer'] = optimizer_name
    hyper_df.loc[hyper_df.index[-1], 'dropout_prob'] = dropout_prob
    hyper_df.loc[hyper_df.index[-1], 'best_epoch'] = best_epoch
    hyper_df.loc[hyper_df.index[-1], 'mse_mean_val'] = val_mean_loss_min
    hyper_df.loc[hyper_df.index[-1], 'mse_mean_train'] = train_mean_loss_at_val_min
    hyper_df.loc[hyper_df.index[-1], 'time_elapsed'] = str(round(end_time - start_time,2))+' sec'

    # Reorder columns so that losses are last
    end_cols = ['best_epoch', 'mse_mean_val', 'mse_mean_train']
    cols = [c for c in hyper_df.columns if not c in end_cols] + end_cols
    hyper_df = hyper_df[cols]
    hyper_df.to_csv(hyper_path, index=False)    
    
    # #### Use model to predict results.
    # Load the model weights which minimized validation loss.
    best_state_dict = torch.load(os.path.join(checkpoint_dir, 'best.pt'))
    model.load_state_dict(best_state_dict)

    dataset_dict = {}
    dataset_dict['train'] = train_dataset
    dataset_dict['val'] = val_dataset

    metrics = pd.DataFrame()
    metrics = metrics.append(pd.Series(dtype='object'), ignore_index=True)
    i = 0
    model.eval() # Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results (https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).
    with torch.no_grad(): # No need to store gradients when doing inference.
        #all_dice_scores = []
        for split, dataset in dataset_dict.items():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # set batch_size to 1 since we want to make predictions for one image at a time, saving the result each time.      
            for index, (img, y) in enumerate(dataloader):
                label_path = dataset.data_df.loc[index, dataset.label_path_col] # path to original label file.
                # Predict labels using best model.
                img = img.to(device)
                yhat = model(img) # shape (B=1, num_labels+1, H, W, D)

                #yhat_temp = yhat[:,:1,:,:,:]
                #print("Shape y",y.shape)
                y = one_hot(y, num_labels)
                y = y.to(device) 

                # print("YHAT",yhat[0, :, :10, :10, :10])
                # print("GROUND",y[0, :, :10, :10, :10])
                #yhat_temp = one_hot(yhat[:,:1,:,:,:]) #kl added
                yhat_temp = yhat.to('cpu')
                yhat_temp = np.array(yhat_temp)
                yhat_temp = yhat_temp[0, :, :, :, :]
                yhat_temp = np.argmax(yhat_temp, axis=0)
                yhat_temp = np.expand_dims(np.expand_dims(yhat_temp, axis=0),axis=0)
                yhat_temp = torch.tensor(yhat_temp)

                yhat_temp = one_hot(yhat_temp, num_labels)
                # dice_score = monai.metrics.compute_meandice(yhat_temp,y.to('cpu'))

                # dice_score = dice_score.cpu().numpy()[0]
                # dice_score = np.append(index, dice_score)
                print("Index:",index)
                dice_score, mean_dice_score = calculate_dice_score(index, yhat_temp, y)
                hausdorff_distance, mean_hausdorff_distance = calculate_hausdorff_distance(index, yhat_temp, y)
                surface_distance, mean_surface_distance = calculate_surface_distance(index, yhat_temp, y)

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
                # print("ARGMAX YHAT",yhat[:10, :10, :10])
                # print("ARGMAX Y",np.argmax(y.to('cpu'),axis=0)[:10, :10, :10])
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

                i+=1

    metrics_name = 'metrics.csv'
    metrics_path = os.path.join(out_dir, metrics_name)
    metrics.to_csv(metrics_path, index=False)

    dice_score_name = 'dice_scores.csv'
    dice_score_path = os.path.join(out_dir, dice_score_name)
    np.savetxt(dice_score_path, all_dice_scores, delimiter=",")

    hausdorff_distance_name = 'hausdorff_distance.csv'
    hausdorff_distance_path = os.path.join(out_dir, hausdorff_distance_name)
    np.savetxt(hausdorff_distance_path, all_hausdorff_distance, delimiter=",")

    surface_distance_name = 'surface_distance.csv'
    surface_distance_path = os.path.join(out_dir, surface_distance_name)
    np.savetxt(surface_distance_path, all_surface_distance, delimiter=",")


if __name__ == '__main__':
    main()
