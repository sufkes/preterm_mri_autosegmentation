from dataset import BrainDataset
import torch
import nibabel as nib
import numpy as np
import os

def getSubvolumeOrigin(img, subvolume_size): # HACK FUNCTION FOR TEMPORARY FIX OF MEMORY ISSUE.
        """Get the origin of the cubic subvolume at the centre of the image. This is just a temporary solution to the memory issue."""
        xlen = img.shape[0]
        ylen = img.shape[1]
        zlen = img.shape[2]
        xmin = max(int((xlen-subvolume_size)/2),0)
        ymin = max(int((ylen-subvolume_size)/2),0)
        zmin = max(int((zlen-subvolume_size)/2),0)
        return (xmin, ymin, zmin)


def main(seed=489, out_dir='../runs/tests/test_004'):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    subvolume_size = 100
    true_dir = os.path.join(out_dir, '/hpf/largeprojects/smiller/users/Katharine/data/dhcp_1mm_katharine')
    pred_dir = os.path.join(out_dir, '/hpf/largeprojects/smiller/users/Katharine/brain_segmentation/runs/test_006/01/predicted_labels')

    all_pred_paths = os.listdir(pred_dir)
    for path in all_pred_paths:
        if 'nii.gz' in path:
            temp = path.split("/")
            image_name = temp[len(temp)-1]
            print("Processing:",image_name)
            true_image = os.path.join(true_dir,image_name)
            pred_image = os.path.join(pred_dir,image_name)
            
            template_image = true_image
            #template_image = os.paths.join(out_dir, 'template')

            labels_true = np.array(nib.load(true_image).get_fdata(), dtype=np.float32)
            xmin, ymin, zmin = getSubvolumeOrigin(labels_true, subvolume_size)
            labels_true = labels_true[xmin:xmin+subvolume_size, ymin:ymin+subvolume_size, zmin:zmin+subvolume_size] # Take same cubic region as for image.

            labels_pred = np.array(nib.load(pred_image).get_fdata(), dtype=np.float32)
            labels_pred = labels_pred[xmin:xmin+subvolume_size, ymin:ymin+subvolume_size, zmin:zmin+subvolume_size] # Take same cubic region as for image.
        
            diff_image = np.where(labels_true == labels_pred, 1, 0)
            
            labels_orig = nib.load(template_image) # Load the original labels NIFTI file to use as a template.
            diff_data = np.zeros(shape=labels_orig.get_fdata().shape) # Initialize the predicted labels image as an array of zeros of the same size as original labels image. 
        
            #### HACK: Fill with predicted values in subvolume cube in which prediction were made.
            diff_data[xmin:xmin+subvolume_size, ymin:ymin+subvolume_size, zmin:zmin+subvolume_size] = diff_image
            diff_data_nifti = nib.Nifti1Image(diff_data, labels_orig.affine, labels_orig.header) # Construct a NIFTI file using the predicted label data, but the header and affine matrix from the original labels NIFTI file.

            # Save predicted labels image as a NIFTI file.
            out_name = os.path.basename(template_image)
            out_name = 'diff_image_'+out_name
            out_path = os.path.join(out_dir, out_name)
            nib.save(diff_data_nifti, out_path)

if __name__ == '__main__':
    main()