import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import pydicom
from nibabel import load as nib_load

class ImageDataIO():
    # arrange dimension for each type of view (axial / saggital / coronal)
    def __init__(self, extention, is2D=True, view=None):
        """        
        :param extention: can be dicom / nifti / jpeg or png
        :param is2D: 
        """
        # if is2D == True and view is not None:
        #     print("is2D variable can't have view property!")
        #     return
        self._extention = extention
        self._is2D = is2D
        self._view = view
        return

    def read_file(self, source_path):
        if self._extention == "jpg" or self._extention == "jpeg" or self._extention == "png":
            return self._read_popular_img(source_path)
        elif self._extention == "dcm" or self._extention == "dicom":
            return self._read_dicom_img(source_path)
        elif self._extention == "nii" or self._extention == "nifti":
            return self._read_nifti_img(source_path)

    # reading jpeg or png
    def _read_popular_img(self, source_path):
        img = Image.open(source_path)
        return img

    # reading dicom
    def _read_dicom_img(self, source_path):
        if self._is2D :
            dcm_img = np.array(pydicom.read_file(source_path).pixel_array)
            return Image.fromarray(dcm_img)
        elif not self._is2D : # source_path is list of dcm files for one subject
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            dcm_imgs = []
            for child_path in child_paths:
                dcm_img = pydicom.read_file(child_path)
                dcm_imgs.append(dcm_img)
            # sorting and save
            dcm_imgs.sort(key=lambda x : int(x.ImagePositionPatient[2]))
            dcm_imgs = np.array([dcm_img.pixel_array for dcm_img in dcm_imgs])
            if self._view == "axial" or self._view == "transaxial":
                dcm_imgs = dcm_imgs
            elif self._view == "coronal":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(1, 2))  #
                dcm_imgs = np.rot90(dcm_imgs, k=3, axes=(0, 1))
            elif self._view == "saggital":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(0, 2))  #
        return [Image.fromarray(dcm_img_pixels) for dcm_img_pixels in dcm_imgs]

    # reading nifti
    def _read_nifti_img(self, source_path):
        """
        
        :param source_path: 
        :return: list of Image obj 
        """
        nib_img = nib_load(source_path)
        nib_img = np.array(nib_img.get_data())
        if self._is2D :
            nib_img = np.rot90(nib_img, k=1, axes=(0, 1))
            return Image.fromarray(nib_img)
        else :
            if self._view == "axial" or self._view == "transaxial":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 1))  # (95, 79, 68)
            elif self._view == "saggital":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 2))  #
            elif self._view == "coronal":
                nib_img = np.rot90(nib_img, k=1, axes=(1, 2))  #
                nib_img = np.rot90(nib_img, k=3, axes=(0, 1))

            nib_img = nib_img.astype(np.uint8)
            nib_img = np.transpose(nib_img, [2, 0, 1])
            # print("3D shape", nib_img.shape)
            return [Image.fromarray(img) for img in nib_img]

    def _resizing_channel(self, img_obj, resize_size, channel_size=None):
        if sys.getsizeof(img_obj) == 0:
            print("size of img_data object is 0")
            return None
         
        if resize_size is not None:
            if not self._is2D :
                sub_list = []
                for sub in img_obj :
                    imgs = [img.resize(resize_size) for img in sub]
                    sub_list.append(imgs)
                img_obj = sub_list

            else:
                 
                if isinstance(img_obj, list):
                    img_obj = [img.resize(resize_size) for img in img_obj]
                else :
                    img_obj = img_obj.resize(resize_size)

        # check channel
        if not self._is2D :
            sub_list = []
            for sub in img_obj:
                if channel_size is not None and channel_size == 1:
                    imgs = [img.convert("L") for img in sub]
                elif channel_size is not None and channel_size == 3:
                    imgs = [img.convert("RGB") for img in sub]
                elif channel_size is not None and channel_size == 4:
                    imgs = [img.convert("RGBA") for img in sub]
                sub_list.append(imgs)
            img_obj = sub_list
        elif isinstance(img_obj, list):
            if channel_size is not None and channel_size == 1:
                img_obj = [img.convert("L") for img in img_obj]
            elif channel_size is not None and channel_size == 3:
                img_obj = [img.convert("RGB") for img in img_obj]
            elif channel_size is not None and channel_size == 4:
                img_obj = [img.convert("RGBA") for img in img_obj]
        else:
            if channel_size is not None and channel_size == 1:
                img_obj = img_obj.convert("L")
            elif channel_size is not None and channel_size == 3:
                img_obj = img_obj.convert("RGB")
            elif channel_size is not None and channel_size == 4:
                img_obj = img_obj.convert("RGBA")

        return img_obj

    def read_files_from_dir(self, source_dir):
        child_full_paths = [os.path.join(source_dir, path) for path in os.listdir(source_dir)]
        file_list = []
        filename_list = []
        for child_full_path in child_full_paths:
            file_list.append(self.read_file(child_full_path))
            filename_list.append(os.path.basename(child_full_path))
        return file_list, filename_list # list of PIL Image obj

    def convert_PIL_to_numpy(self, img_obj):

        if isinstance(img_obj, list):
            if self._is2D is False:
                sub_list = []
                for sub in img_obj:
                    img_list = []
                    for img in sub:
                        #print(np.asarray(img))
                        img_list.append(np.asarray(img))
                    img_list = np.array(img_list)
                    sub_list.append(img_list)
                return np.array(sub_list)

            return np.array([np.array(img) for img in img_obj])
        else:
            return np.asarray(img_obj)

    def convert_numpy_to_PIL(self, img_obj, single_obj=False):
        if single_obj :
            return Image.fromarray(img_obj)
        #print("convert_numpy_to_PIL",img_obj)
        if self._is2D is False:
            sub_list = []
            for sub in img_obj:
                img_list = []
                for img in sub:
                    #print(np.asarray(img))
                    img_list.append(Image.fromarray(img))
                sub_list.append(img_list)
            return sub_list

        return [Image.fromarray(img) for img in img_obj]


    def _get_unchoiced_ind(self, choiced_ind_list, num_total):
        unchoiced_ind_list = []
        for ind in range(len(num_total)):
            if ind not in choiced_ind_list:
                unchoiced_ind_list.append(ind)
        return np.array(unchoiced_ind_list)

if __name__ == "__main__":
    print("hello world")
    
    extention = "dcm"
    is2D = False
    view = "saggital"
    source_path = "..\\AD_sub1"
    idio = ImageDataIO(extention, is2D, view)
    img = idio.read_file(source_path)
    for ind in range(len(img)):

        img[ind].show()
    print("size", img[0].size)
