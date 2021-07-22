'''
prepData.py [OUTDATED]

This script is outputs csv files that contain paths to T1w, T2w, structure labels,
and tissue labels for a particular dataset. It marks a given percentage as train, val, and test files. 

This script has been replaced by two similar scripts, one tailored to ubc data and the other tailored to dhcp data. 

arguments: num files to inclde, output folder after file_lists [optional]
output: filelist csv written to out_dir
'''
import pandas as pd
import os
import math
import random
import sys

#This script prepares a csv which holds the file paths to training, validation, and testing data for the segmentation pipeline

''''ALTER THE FOLLOWING PARAMETERS'''
#PATHS
# data_file_path = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data' #path from root to dir with data
# out_path = '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data/file_lists' #path from root to dir where you want csv to be output
# data_file_path = '/hpf/largeprojects/smiller/users/Katharine/data/dhcp_1mm_katharine'
# out_path = os.path.join('/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data/file_lists',sys.argv[2])
data_file_path = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation'
out_path = os.path.join('/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/data/file_lists',sys.argv[2])
separate_data_folders = True
os.makedirs(out_path, exist_ok=True)

#SPLIT (must sum to 1)
train_split = .8
val_split = .15
test_split = .05
num_files = int(sys.argv[1]) #number of files you want to include

#TOKENS
#UBC data:
token_subject = 'BC' #token in all nifti files
separate = '_' #token that separates information in file name
num_sep_in_subject_name = 0 #number of separation tokens in the subject portion of the name
token_T1_image = 'V01.nii.gz'
token_T2_image = 'V02.nii.gz'
token_tissue_label = 'NONE' #token in only files pertaining to tissue labels
token_structure_label = 'V01.nii.gz' #token in only files pertaining to structure labels

#DHCP data:
# token_subject = 'sub-' #token in all nifti files
# separate = '_' #token that separates information in file name
# num_sep_in_subject_name = 1 #number of separation tokens in the subject portion of the name
# token_T1_image = '_T1w_brain'
# token_T2_image = '_T2w_brain'
# token_tissue_label = '_desc-drawem9_' #token in only files pertaining to tissue labels
# token_structure_label = '_desc-drawem87_' #token in only files pertaining to structure labels

#token_structure_label = 'desc-bet_space'

'''ASSERT TESTS'''
assert(train_split+val_split+test_split==1)
print("WARNING: Any subjects missing an image or label will not appear in the csv.")

'''INITIALIZE STRUCTURES'''
#d = {'subject': [],'t1_brain_path':[],'t2_brain_path':[],'tissue_label_path':[],'structure_label_path':[]}
df = pd.DataFrame(columns=['subject', 't1_brain_path','t2_brain_path','tissue_label_path','structure_label_path'])
T1_images = []
T2_images = []
tissue_labels = []
structure_labels = []
subjects = set()

'''POPULATE LISTS'''
if (separate_data_folders):
    # all_paths = os.listdir('/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T1w')
    # all_paths.extend(os.listdir('/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T2w'))
    # all_paths.extend(os.listdir('/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T2w/labels'))

    image_dir = os.listdir('/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/images')
    label_dir = os.listdir('/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/newlabels')
else:
    all_paths = os.listdir(data_file_path)

##DHCP
# for path in all_paths:
#     print(path)
#     if token_T1_image in path:
#         T1_images.append(path)
#     elif token_T2_image in path:
#         T2_images.append(path)
#     elif token_tissue_label in path:
#         tissue_labels.append(path)
#     elif token_structure_label in path:
#         structure_labels.append(path)
    
#     if(token_subject in path):
#         split_name = path.split("_")
#         name = split_name[0]
#         for i in range(num_sep_in_subject_name):
#             name = name + "_"+ split_name[i+1]
#         subjects.add(name)

#UBC
for path in image_dir:
    print(path)
    if token_T1_image in path:
        T1_images.append(path)
    elif token_T2_image in path:
        T2_images.append(path)
    
    if(token_subject in path):
        split_name = path.split("_")
        name = split_name[0]
        for i in range(num_sep_in_subject_name):
            name = name + "_"+ split_name[i+1]
        subjects.add(name)

for path in label_dir:
    print(path)
    if token_tissue_label in path:
        tissue_labels.append(path)
    elif token_structure_label in path:
        structure_labels.append(path)

subjects = sorted(subjects)
print("No. Subjects",len(subjects),"\nNo. T1w Images",len(T1_images),"\nNo. T2w Images",len(T2_images),"\nNo. Tissue Labels",len(tissue_labels),"\nNo. Structure Labels",len(structure_labels))
print("If number of files are not equal, note that excess images and labels will not appear in the csv.")
assert(len(subjects)-1>num_files)
'''COMBINE LISTS INTO DATAFRAME'''

i = 0
while len(df) < num_files and i < len(subjects):
   
    # print(df)
    #print(i)
    s = subjects[i]

    if (separate_data_folders):
        # data_file_path_1 = '/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T1w'
        # data_file_path_2 = '/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T2w'
        # data_file_path_3 = '/hpf/largeprojects/smiller/users/Katharine/monaiPipeline/MONAI_DATA/dhcp_data_katharine/T2w/labels'
        data_file_path_1 = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/images'
        data_file_path_2 = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/newlabels'

        T1 = T2 = tis = stru = None
        # if(any(t for t in T1_images if s in t)):
        #     T1 = os.path.join(data_file_path_1,[t for t in T1_images if s in t][0])
        # if(any(t for t in T2_images if s in t)):
        #     T2 = os.path.join(data_file_path_2,[t for t in T2_images if s in t][0])
        # if(any(t for t in tissue_labels if s in t)):
        #     tis = os.path.join(data_file_path_3,[t for t in tissue_labels if s in t][0])
        # if(any(t for t in structure_labels if s in t)):
        #     stru = os.path.join(data_file_path_3,[t for t in structure_labels if s in t][0])

        if(any(t for t in T1_images if s in t)):
            T1 = os.path.join(data_file_path_1,[t for t in T1_images if s in t][0])
        if(any(t for t in T2_images if s in t)):
            T2 = os.path.join(data_file_path_1,[t for t in T2_images if s in t][0])
        if(any(t for t in tissue_labels if s in t)):
            tis = os.path.join(data_file_path_2,[t for t in tissue_labels if s in t][0])
        if(any(t for t in structure_labels if s in t)):
            stru = os.path.join(data_file_path_2,[t for t in structure_labels if s in t][0])
    else:
        T1 = T2 = tis = stru = None
        if(any(t for t in T1_images if s in t)):
            T1 = os.path.join(data_file_path,[t for t in T1_images if s in t][0])
        if(any(t for t in T2_images if s in t)):
            T2 = os.path.join(data_file_path,[t for t in T2_images if s in t][0])
        if(any(t for t in tissue_labels if s in t)):
            tis = os.path.join(data_file_path,[t for t in tissue_labels if s in t][0])
        if(any(t for t in structure_labels if s in t)):
            stru = os.path.join(data_file_path,[t for t in structure_labels if s in t][0])

        
    temp = pd.DataFrame([[s,T1,T2,tis,stru]], columns = ['subject','t1_brain_path','t2_brain_path','tissue_label_path','structure_label_path'])
    df = df.append(temp, ignore_index=True)
    df.dropna(subset = ["t1_brain_path"], inplace=True)

    i+=1


'''ASSIGN SPLIT LABELS'''
num_train = math.floor(train_split * num_files)
num_val = math.floor(val_split * num_files)
num_test = math.floor(test_split * num_files)
num_train += num_files - num_train - num_val - num_test

split = ['train' for i in range(num_train)]
split.extend(['val' for i in range(num_val)])
split.extend(['test' for i in range(num_test)])
random.shuffle(split)
df['split'] = split


'''CONVERT AND SAVE'''
df.to_csv(out_path+'/'+'FileList_Size='+str(num_files)+'_Split='+str(train_split)+'-'+str(val_split)+'-'+str(test_split)+'.csv') 
print('OUTPUT FILE NAME: '+sys.argv[2]+'/FileList_Size='+str(num_files)+'_Split='+str(train_split)+'-'+str(val_split)+'-'+str(test_split)+'.csv')
