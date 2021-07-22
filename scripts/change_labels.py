import os
import numpy as np
import nibabel as nib

image_dir = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/images'
original_dir = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/newlabels'
out_dir = '/hpf/largeprojects/smiller/users/Katharine/data/connorSegmentation/new_labels_without_STN'
# original_labels = [1,2,3,4,14,6,7,15,18,21,11,35]
# new_labels = [1,2,3,4,5,6,7,8,9,10,11,12]
original_labels = [1,2,3,4,5,6,7,8,9,10,11,12]
new_labels = [1,2,3,4,5,6,7,8,0,9,10,0]

all_paths = os.listdir(original_dir)
template_path = all_paths[0]


assert(len(original_labels) == len(new_labels))

for path in all_paths:
    if 'nii.gz' in path:
        print(path)
        image = np.array(nib.load(os.path.join(image_dir,path)).get_fdata(), dtype=np.float32)
        
        labels = np.array(nib.load(os.path.join(original_dir,path)).get_fdata(), dtype=np.float32)
        before = labels.shape
        
        for i in range(len(original_labels)):
            original = original_labels[i]
            new = new_labels[i]
            labels[labels==original] = new
        template = nib.load(os.path.join(original_dir,template_path)) # Load the original labels NIFTI file to use as a template.
        #label_pred_data = np.zeros(shape=template.get_fdata().shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 
        if(image.shape != labels.shape):
            print("IMAGE BEFORE",image.shape)
            print("IMAGE BEFORE",labels.shape)
            
        #### HACK: Fill with predicted values in subvolume cube in which prediction were made.
        #xmin, ymin, zmin = dataset.getSubvolumeOrigin(index)
        #label_pred_data[xmin:xmin+subvolume_size[0], ymin:ymin+subvolume_size[1], zmin:zmin+subvolume_size[2]] = labels
        new_nifti = nib.Nifti1Image(labels, template.affine, template.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

        # Save predicted labels image as a NIFTI file.
        out_name = os.path.basename(path)
        out_path = os.path.join(out_dir, out_name)
        nib.save(new_nifti, out_path)