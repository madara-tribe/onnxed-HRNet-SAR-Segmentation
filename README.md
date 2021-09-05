# HRNet-SAR-Segmentation

# Version
- python 3.7.0
- tensorflow=='2.3.0'
- keras=='2.3.1'
- onnx==1.10.1
- keras2onnx=='1.9.0'
- onnxruntime=='1.8.1'

# abstract

semantic segmentation for in SAR(synthetic aperture radar) images which is took from satelliate.

1 SAR image has VV and HV type. its task to detect new bulding that did not exist in the past

<img src="https://user-images.githubusercontent.com/48679574/132123393-eef67a13-83b5-46b1-845b-7539ef6dba01.png" width="400px">


## Network : HRNet

The high-resolution network (HRNet) is a universal architecture for visual recognition.

is a general purpose convolutional neural network for tasks like semantic segmentation, object detection and image classification. It is able to maintain high resolution representations through the whole process

<img src="https://user-images.githubusercontent.com/48679574/132123348-925e4648-cdfa-43f0-98fd-f97e87947056.png" width="800px">


# approach: crop, prediction and concat




