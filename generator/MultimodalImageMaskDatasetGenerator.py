# Copyright 2026 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2026/09/05
# MultimodalImageMaskDatasetGenerator.py

import os
import cv2
import glob
import nibabel as nib
import shutil
import traceback
import numpy as np
import traceback

class MultimodalImageMaskDatasetGenerator:
  def __init__(self, resize=512):
    self.RESIZE = (resize, resize)
    self.NORMALIZE = True
    self.ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE
  
    # RGB        NET/NCR:red, ED: green, ET:blue, CC: mazenta
    #            (255,0,0):1, (0,255,0):2,(0,0,255):3,(255,0,255):4
    self.BGR_MASK_COLORS  = [(0,0, 255), (0,255, 0),(255,0, 0),  (255,0, 255)]
  
  def colorize_mask(self, mask):
     h, w = mask.shape[:2]
     colorized = np.zeros((h, w, 3), dtype=np.uint8)
     for i, bgr_color in enumerate(self.BGR_MASK_COLORS):
      colorized[np.equal(mask, i+1)] = bgr_color
     return colorized

  def normalize(self, data):
    min = data.min()
    max = data.max()
    
    if max - min != 0:
      normalized = ((data - min) / (max - min) * 255).astype(np.uint8)
    else:
      normalized = np.zeros_like(data, dtype=np.uint8)
    return normalized
   
  def get_mask(self, i, data):
    # Get i-th mask slice from the data.
    mask = data[:, :, i]  
    if self.ROTATION:
      mask = cv2.rotate(mask, self.ROTATION)
    if self.RESIZE:
      mask = cv2.resize(mask, self.RESIZE)
    valid = False
    if mask.any() > 0:   
      mask = self.colorize_mask(mask)
      valid = True
    return valid, mask, 
  
  def get_image(self, i, data):
    # Get i-th image slice from the data.
    image = data[:,:,i]
    if self.NORMALIZE:
      image = self.normalize(image)
    if self.ROTATION:
      image = cv2.rotate(image, self.ROTATION)
    if self.RESIZE:
      image = cv2.resize(image, self.RESIZE)
    return image
  
  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
     t1c_image_files = sorted(glob.glob(images_dir + "/*-t1c.nii"))
     t1n_image_files = sorted(glob.glob(images_dir + "/*-t1n.nii"))
     t2f_image_files = sorted(glob.glob(images_dir + "/*-t2f.nii"))
     t2w_image_files = sorted(glob.glob(images_dir + "/*-t2w.nii"))

     num_t1c = len(t1c_image_files)
     num_t1n = len(t1n_image_files)
     num_t2f = len(t2f_image_files)
     num_t2w = len(t2w_image_files)

     print(num_t1c, num_t1n, num_t2f, num_t2w)
     if not (num_t1c == num_t1n and num_t1c == num_t2f and num_t1c == num_t2w):
       raise Exception("Error: Unmatched the number of NIfTI files")
     
     num_images = num_t1c

     mask_files = sorted(glob.glob(masks_dir + "/*-seg.nii"))
     num_masks = len(mask_files)

     print("Number of mask_files", num_masks)
     if num_images != num_masks:
       raise Exception("Error: Unmatched the number of images and masks")

     #index = NIfTI file name index
     index = 10000
     for i in range(num_masks):
       index += 1
       mask_data = nib.load(mask_files[i]).get_fdata()
       num_mask_slices = mask_data.shape[2]
      
       t1c_data = nib.load(t1c_image_files[i]).get_fdata()
       t1n_data = nib.load(t1n_image_files[i]).get_fdata()
       t2f_data = nib.load(t2f_image_files[i]).get_fdata()
       t2w_data = nib.load(t2w_image_files[i]).get_fdata()

       num_t1c_slices = t1c_data.shape[2]
       num_t1n_slices = t1n_data.shape[2]
       num_t2f_slices = t2f_data.shape[2]
       num_t2w_slices = t2w_data.shape[2]
       print(num_t1c_slices, num_t1n_slices, num_t2f_slices, num_t2w_slices)

       if not (num_t1c_slices == num_t1n_slices and num_t1c_slices == num_t2f_slices and num_t1c_slices == num_t2w_slices):
         raise Exception("Error: Unmatched the number of image slices.")
       
       if num_mask_slices != num_t1c_slices:
         raise Exception("Error: Unmatched the number of mask and image slices.")
         
       for j in range(num_mask_slices):
          valid, mask = self.get_mask(j, mask_data)
          #If a non-emtpry mask found, then generate a corresponding npy image file.
          if valid == True:
            # j = slice_index
            t1c = self.get_image(j, t1c_data)
            t1n = self.get_image(j, t1n_data)
            t2f = self.get_image(j, t2f_data)
            t2w = self.get_image(j, t2w_data)

            filename = str(index) + "_" + str(j) + ".png"
            mask_filepath = os.path.join(output_masks_dir, filename)
            cv2.imwrite(mask_filepath, mask)
            print("Mask shape", mask.shape)
            print("Saved", mask_filepath)

            multimodal_slice = np.stack([t1c, t1n, t2f, t2w], axis=-1)
            print("Multimodal slice shape", multimodal_slice.shape)

            npy_filename = str(index) + "_" + str(j) + ".npy"
            npy_filepath = os.path.join(output_images_dir, npy_filename)

            np.save(npy_filepath, multimodal_slice)
            print("Saved", npy_filepath)
       
          else:
            print("Skipped for an empty mask case.")


if __name__ == "__main__":
  try:
    images_dir = "./archive/"
    masks_dir  = "./archive/"

    output_dir = "./BraTS-PEDs-Multimodal-master"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    generator = MultimodalImageMaskDatasetGenerator(resize=512)
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


                   
      

