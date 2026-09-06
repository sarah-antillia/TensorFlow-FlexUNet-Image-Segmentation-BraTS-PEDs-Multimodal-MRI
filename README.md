<h2>TensorFlow-FlexUNet-Image-Segmentation-BraTS-PEDs-Multimodal-MRI (2026/09/06)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>BraTS-PEDs-Multimodal (T1C, T1N, T2F, and T2W) MRI</b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and a 512x512 pixels upscaled 
 NPY Images and PNG Masks
 <a href="https://drive.google.com/file/d/1VyPKBWrm95u4pO_6tXS79MXQaeLl66zW/view?usp=sharing">
BraTS-PEDs-Multimodal-ImageMask-Dataset.zip</a> with colorized masks (<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>),
  which was derived by us from the Kaggle website 
<br><br>
<a href="https://www.kaggle.com/datasets/awansaad6797/brats-peds-the-brain-tumor-segmentation-dataset">
<b>BraTS-PEDs | The Brain Tumor Segmentation Dataset</b>
</a> by Malik Saad Ahmed.
<br><br>
<hr>
<b>Actual Image Segmentation for BraTS-PEDs-Multimodal Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks, but differ in detail.<br>
In the following table, for simplicity, we showed only the T2W slices in NPY files that contain T1C. T1N, T2F, and T2W 
Multimodal MRI images.<br>
<br>
<a href="#1"><b>class_color_mapping_table</b></a><br>
<br>
<table>
<tr>
<th>T2W slice in NPY</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10017_83.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10017_83.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10017_83.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10002_64.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10002_64.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10002_64.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10009_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10009_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10009_66.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle website:<br><br>
<a href="https://www.kaggle.com/datasets/awansaad6797/brats-peds-the-brain-tumor-segmentation-dataset">
<b>BraTS-PEDs | The Brain Tumor Segmentation Dataset</b>
</a> by Malik Saad Ahmed.
<br><br>
The following explanation(abstract) was taken from the website <a href="https://www.cancerimagingarchive.net/collection/brats-peds/">
BraTS-PEDs | The Brain Tumor Segmentation in Pediatric Magnetic Resonance Imaging</a>.
<br><br>
<b>Abstract</b><br>
Pediatric central nervous system tumors are the leading cause of cancer-related mortality in children. 
Pediatric high-grade gliomas, particularly diffuse midline gliomas (DMGs), have a dismal prognosis, 
with five-year survival rates below 20%. The BraTS-PEDs dataset provides a comprehensive, multi-institutional, 
international resource focused on this disease. It includes 457 pediatric patients with high-grade gliomas, 
primarily DMGs, aggregated from major pediatric neuro-oncology consortia and institutions, including the 
Children’s Brain Tumor Network (CBTN), DMG/DIPG Registry, and multiple academic centers.<br> 
The dataset contains multiparametric structural MRI, expert tumor segmentations, imaging acquisition parameters, 
demographic variables, and clinical outcomes such as overall and progression-free survival. 
By integrating standardized imaging, annotations, and clinical data, BraTS-PEDs enables reproducible 
research and accelerates translation of imaging science into clinical impact.
<br>
<br>
<b>License</b><br>
<a href="<a href="https://creativecommons.org/licenses/by-sa/4.0/">
CC BY-SA 4.0</a>.
<br>
<br>
For more information, please refer to 
<a href="https://www.cancerimagingarchive.net/collection/brats-peds/">
BraTS-PEDs | The Brain Tumor Segmentation in Pediatric Magnetic Resonance Imaging</a>.
<br><br>
<h3>
2 BraTS-PEDs-Multimodal  ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this BraTS-PEDs-Multimodal Segmentation model,
 please download the dataset from Google Drive  
 <a href="https://drive.google.com/file/d/1VyPKBWrm95u4pO_6tXS79MXQaeLl66zW/view?usp=sharing">
BraTS-PEDs-Multimodal-ImageMask-Dataset.zip</a> ( <a href="<a href="https://creativecommons.org/licenses/by-sa/4.0/">
CC BY-SA 4.0</a>). 
Expand the downloaded ImageMaskDataset and put it under the <b>./dataset</b> folder
<br>
<pre>
./dataset
└─BraTS-PEDs-Multimodal
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
<b>BraTS-PEDs-Multimodal Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/BraTS-PEDs-Multimodal_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large enough to use for the
 training set of our segmentation model.
<br>
<br>
<h3>2.2 Derivation of BraTS-PEDs-Multimodal ImageMask Dataset</h3>
The folder structure of the original dataset is as follows.
It contains a lot of types of NIfTI image files,
but we used only one type of T2F (<b>BraTS-PED-*-t2f.nii</b>) NIfTI file in this experiment.
<pre>
./dataset
 ├─BraTS-PED-00001-000-seg.nii
 ├─BraTS-PED-00001-000-t1c.nii
 ├─BraTS-PED-00001-000-t1n.nii
 ├─BraTS-PED-00001-000-t2f.nii
 ├─BraTS-PED-00001-000-t2w.nii
,,,
 ├─BraTS-PED-00021-000-seg.nii
 ├─BraTS-PED-00021-000-t1c.nii
 ├─BraTS-PED-00021-000-t1n.nii
 ├─BraTS-PED-00021-000-t1f.nii
 └─BraTS-PED-00021-000-t2w.nii  
</pre>
<b>Note</b><br>
We removed all BraTS-PED-00022-000-*.nii files because BraTS-PED-00022-000-t2w.nii was missing.<br>
<br> 
<b>Step 1</b><br>
We generated a 512x512 pixels upscaled NPY Images and PNG Masks Dataset with colorized masks 
 from the image slices of <b>BraTS-PED-*-t*.nii</b> 
and the correspoding mask slices of <b>BraTS-PED-*-seg.nii</b> in <b>dataset</b> folder by using a simple Python script 
<a href="./generator/MultimodalImageMaskDatasetGenerator.py">MultimodalImageMaskDatasetGenerator.py</a>.
However, for simplicity, we excluded all black empty masks and the corresponding images because they are irrelevant for 
training our segmentation model.<br><br>
We also used the following class_color_mapping table to generate the colorized masks.
<br><br>
<a id="1"><b>class_color_mapping_table</b></a>
<br><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/1.png' widith='40' height='25'></td><td>(255, 0, 0)</td>
<td>Non-enhancing Tumor Core / Necrosis: NET/NCR</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/2.png' widith='40' height='25'></td><td>(0, 255, 0)</td>
<td>Peritumoral Edema: ED</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/3.png' widith='40' height='25'></td><td>(0, 0, 255)</td>
<td>GD-Enhancing Tumor: ET</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/4.png' widith='40' height='25'></td><td>(255, 0, 255)</td>
<td>Cystic Component / Infiltration: CC</td></tr>
</table>
<br>
<br>
<h3>2.3 Train Sample T2W Slices and Masks</h3>
<b>Train_sample_tw2_slices</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/train_images_t2w_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained the BraTS-PEDs-Multimodal TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 5
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Define a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
Specified RGB color map dict for BraTS-PEDs-Multimodal 1+4 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
; BraTS-PEDs-Multimodal RGB color map dict for 1+4 classes.
;        Background: black, NET/NCR: red, ED: green, ET: blue, CC: magenta
rgb_map = {(0,0,0):0,(255,0,0):1, (0,255,0):2,(0,0,255):3,(255,0,255):4}
</pre>

<b>Infer section</b><br>
<pre>
[infer] 
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output/"
;2026/09/05
; Specify npy mini_test_images
; Specify a modality index.
; 0=t1c, 1=t1n, 2=t2f,3=t2w
npy_slice_index =3
; Specify a slice output_dir
npy_slices_dir   = "./mini_test_t2w_slices/"
</pre>
<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<!--
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br>
--> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 34,35,36)</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 69,70,71)</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 71 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/train_console_output_at_epoch71.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for BraTS-PEDs-Multimodal.<br>
<pre>
./2.evaluate.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/evaluate_console_output_at_epoch71.png" width="1024" height="auto">
<br><br>Image-Segmentation-BraTS-PEDs-Multimodal

<a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this BraTS-PEDs-Multimodal/test was very low, and dice_coef_multiclass very  
high, as shown below.
<br>
<pre>
categorical_crossentropy,0.0094
dice_coef_multiclass,0.9956
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for
 BraTS-PEDs-Multimodal.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_t2w_images</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/mini_test_t2w_slices.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for BraTS-PEDs-Multimodal Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks, but differ in detail.<br>
In the following table, for simplicity, we showed only the T2W slices in NPY files that contain T1C. T1N, T2F, and T2W
Multimodal MRI images.
<br><br>
<a href="#1"><b>class-color-mapping-talbe</b></a><br>
<br>
<table>
<tr>
<th>T2W slices in NPY</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10001_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10001_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10001_104.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10004_48.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10004_48.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10004_48.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10009_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10009_66.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10009_66.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10014_78.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10014_78.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10014_78.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10016_106.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10016_106.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10016_106.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_t2w_slices/10017_83.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test/masks/10017_83.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_output/10017_83.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<!--
<h3>
6 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal</b> folder
and run the following bat file to infer image segmentation for 2D slices of 3D volume NIfTI files
 using the trained TensorFlowFlexUNet model for BraTS-PEDs-Multimodal.<br>
<pre>
>./5.infer3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNet3DInferencer.py ./train_eval_infer.config
</pre>
<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
; Specify an images_dir which contains NIfTI or NPY files
images_dir    = "./mini_test_3d/images/"
output_dir    = "./mini_test_3d_output/"
slice_shape_order = "hwd"
slice_normalize = True
slice_resize   = (512,512)
; Specify a cv2.rotation mode as a string.
slice_rotation = "cv2.ROTATE_90_COUNTERCLOCKWISE" 

mask_overlay  = True
</pre>
<hr>
<b>Acutual Image Segmentation for 2D Slices of a BraTS-PEDs-Multimodal NIfTI</b><br>
Some Slices, Inferred Masks and Mask overlays for a 3D volume <b>BraTS-PED-00001-000-t2f.nii</b> file in 
<b>archive/</b> folder.<br>
<br>
<a href="#1"><b>class-color-mapping-talbe</b></a><br>
<br>
<table>
<tr>
<th>Image</th>
<th>Inferred-mask</th>
<th>Mask overlay</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10034.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10038.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10038.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10038.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10041.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10041.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10041.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10044.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10044.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10044.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10047.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/slices/10086.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/masks/10086.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/mini_test_3d_output/BraTS-PED-00001-000-t2f.nii/overlays/10086.png" width="320" height="auto"></td>
</tr>

</table>

<h3>
7 MaskOverlay Video of 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal</b> folder, and run the following bat file 
to generate <b>overlays.mp4</b> or <b>overlay.gif</b> for MaskOverlays of 3D Volume Segmentation. <br>
<pre>
>./6.video3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/MaskOverlayVideoGenerator.py ./train_eval_infer.config
</pre>
<br>
<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
mask_overlay  = True
; Specify ".mp4" or ".gif".
;video_fileformat  = ".mp4"
video_fileformat  = ".gif"
</pre>
<br>
<b>overlays.gif</b><br>
<img src="./projects/TensorFlowFlexUNet/BraTS-PEDs-Multimodal/video_3d/overlays.gif">
<br>
-->

<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-BraTS-PEDs-T2F</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BraTS-PEDs-T2F">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BraTS-PEDs-T2F
</a>
<br><br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2023-Subset
</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-BraTS-Africa-Glioma-T2W</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BraTS-Africa-Glioma-T2W">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-BraTS-Africa-Glioma-T2W
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Multiclass-BraTS2020
</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
