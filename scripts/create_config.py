'''
create_config.py

This script creates the config files used by run.py to initialize all parameters and hyperparameters.

arguments: output name [optional]
output: config text file in json format written to out_dir
'''
import json
import sys
import os

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def main(seed=489, out_dir='../config_files/V02'):
    try:
        out_name = sys.argv[1]
    except:
        #out_name = 'ubc_size=184_ep=40_out=ubc-V02-pretrain-02_1.txt'
        out_name = 'transfer_ubc_size=204_ep=40_ss=[72,100,120]_out=ubc-pretrain-07.txt'
        #out_name = 'nomask_dhcp_size=492_ep=50_ss=[98,125,101]_out=dhcp-nopretrain-nomask.txt'
        #out_name = 'transfer_dhcp_size=30_ep=10_ss=[98,125,101]_out=dhcp-pretrain-03.txt'

    run_parameters = {
        "file_list_path": '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/file_lists/ubc/V02/FileList_ubc_Size=183_Split=0.75-0.15-0.1.csv',
         "epochs": 120,
         "out_dir": '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/runs/ubc/V02/nomask_pretrainNoMask'
    }
    ''''
    "out_dir": '../runs/ubc/pretrain/07'

    #V02 FILE LISTS
    ubc/V02/FileList_ubc_Size=30_Split=0.9-0.1-0.0.csv
    ubc/V02/FileList_ubc_Size=183_Split=0.75-0.15-0.1.csv

    ubc/masks/V02/FileList_ubc_Size=172_Split=0.75-0.15-0.1.csv

    #V01 FILE LISTS
    ubc/V01/FileList_ubc_Size=30_Split=0.9-0.075-0.025.csv
    ubc/V01/FileList_ubc_Size=203_Split=0.75-0.15-0.1.csv

    ubc/masks/V01/FileList_ubc_Size=179_Split=0.75-0.15-0.1.csv

    #DHCP FILELISTS
    dhcp/masks/FileList_dhcp_Size=492_Split=0.9-0.1-0.0.csv

    dhcp/FileList_dhcp_Size=30_Split=0.9-0.1-0.0.csv
    dhcp/FileList_dhcp_Size=492_Split=0.9-0.1-0.0.csv
    
    '''
    dataset_parameters = {
        "num_labels":10,
        "subvolume_size":[100,120,105]
    }

    resume_transfer_parameters = {
        "resume_flag": False,   
        "transfer_train_flag": True,  
        # "load_checkpoint_path": '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/trained_models/dhcp_models/mask_vl=1367_nl=87_ss=[98,125,101].pt',
        "load_checkpoint_path": '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/trained_models/dhcp_models/nomask_vl=2112_nl=87_ss=[98,125,101].pt',
    }

    standard_config = {
        "dropout_prob": 0.0,
        "batch_size": 1,
        "shuffle": True,
        "drop_last": False,
        "loss_name": 'DiceLoss',
        "optimizer_name": 'Adam',
        "learning_rate": 0.01, # (value in HiResNet paper = 0.01),
        "weight_decay": 0.0
    }

    # file_list_name = (run_parameters['file_list_path'].split('/')[-1]).split('_')
    # out_name = 'config_'+'ep='+run_parameters['epochs']
    final_config = Merge(standard_config,Merge(resume_transfer_parameters, Merge(run_parameters, dataset_parameters)))
    y = json.dumps(final_config, indent=4)

    out_path = os.path.join(out_dir, out_name)  
    try:
        assert(out_name not in os.listdir(out_dir))
        fp = open(out_path, 'w')
        fp.write(y)
        fp.close()
    except:
        if(input('\nFile already exists. Would you like to overwrite this file? (y/n):\n') == 'y'):
            print('file overwritten')
            fp = open(out_path, 'w')
            fp.write(y)
            fp.close()
    print('OUTPUT FILE NAME:\n'+out_path)


if __name__ == '__main__':
    main()

'''
    dhcp = {
        "num_labels":87,
        "subvolume_size":[98,125,101],
    }
    v01 = {
        "num_labels":10,
        "subvolume_size":[72,100,120],
    }

    v02 = {
        "num_labels":10,
        "subvolume_size":[100,120,105], #need to add offset in dataset.py
    }
'''