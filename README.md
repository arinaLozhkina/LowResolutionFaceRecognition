# LowResolutionFaceRecognition
## Abstract
Despite significant advances in high-resolution recognition, low-resolution facial recogni-tion remains a challenge. Many methods have been proposed for its solution, which can be divided by 2 categories: super resolution methods and resolution-invariant feature extraction. In this work the focus is on the second one.

In the project the problem of low resolution face recognition is studied including methods of solving cross-resolution and low-resolution face recognition problem, the pipeline of face recognition based on deep learning and the low resolution datasets and their evaluation protocols.

The high resolution facial recognition methods are implemented: CosFace, SphereFace and ArcFace. They are also adapted for low resolution face recognition problem using the Cross Resolution Batch training. Also the finetuning methods: Octuplet Loss and DeriveNet are implemented. The implementation of face recognition pipeline based on the existing implementation is included. All methods are tested and compared using low resolution face images.

Three low resolution datasets: LFW, SCface, QMUL-SurvFace, and their evaluation protocols are studied and implemented.

One of the modern directions is the application of adaptive margin function. The pro-posed methods are based on it, which adapt margins for images of different quality. The quality of the images is considered using the Laplacian operator. The proposed methods sur-passed the state-of-the-art algorithm of Cross Resolution Face Recognition, AdaFace, when tested on low resolution images.

## Repository overview
```bash
.
├── cropped_data
│   ├── lfw
│   │   ├── lfw_crop [5749 entries exceeds filelimit, not opening dir]
│   │   └── pairs.txt
│   ├── qmul
│   │   └── Face_Verification_Test_Set
│   │       ├── negative_pairs_names.mat
│   │       ├── positive_pairs_names.mat
│   │       └── verification_images [10051 entries exceeds filelimit, not opening dir]
│   ├── scface
│   │   ├── models_croppped [130 entries exceeds filelimit, not opening dir]
│   │   └── sc_cropped [2586 entries exceeds filelimit, not opening dir]
│   └── webface
│       ├── casia-112x112 [10576 entries exceeds filelimit, not opening dir]
│       └── train.txt
├── requirements.txt
└── src
    ├── data
    │   ├── __pycache__
    │   │   ├── test_dataset.cpython-37.pyc
    │   │   └── training_dataset.cpython-37.pyc
    │   ├── test_dataset.py
    │   └── training_dataset.py
    ├── data_conf.yaml
    ├── finetuning
    │   ├── derive_net.py
    │   └── octuplet.py
    ├── head
    │   ├── adaptiveface.py
    │   ├── arcface.py
    │   ├── cosface.py
    │   ├── dmargin.py
    │   ├── __pycache__
    │   │   ├── adaptiveface.cpython-37.pyc
    │   │   ├── arcface.cpython-37.pyc
    │   │   ├── cosface.cpython-37.pyc
    │   │   ├── dmargin.cpython-37.pyc
    │   │   └── sphereface.cpython-37.pyc
    │   └── sphereface.py
    ├── model.py
    ├── __pycache__
    │   ├── model.cpython-37.pyc
    │   └── resnet.cpython-37.pyc
    ├── resnet.py
    ├── test.py
    ├── train.py
    └── weights
        ├── Approach1.pt
        ├── Approach2.pt
        ├── ArcFace.pt
        ├── CosFace.pt
        ├── DeriveNet.pt
        ├── Octuplet.pt
        └── SphereFace.pt

19 directories, 35 files
```

## Requirements 
- opencv-python==4.7.0.68
- pandas==1.5.2
- PyYAML==6.0
- scipy==1.9.3
- torch==1.11.0
- torchvision==0.12.0
- tqdm==4.64.1

## Weights 
Weights of pretrained models can be found [on GDrive](https://drive.google.com/file/d/11pkV06g3I6Avwj2lNz8To5_LVyP_MecC/view?usp=sharing).

## Training 
The training is implemented for ResNet50-re. There are 5 approaches implemented. 

### Step 1: Prepare the training data
Align the face images to 112*112 using [MTCNN](https://github.com/ipazc/mtcnn)

### Step 2: Generate the train paths file
Generate the file with all paths and labels as follows: 
```
2986449/002.jpg 2986449
2986449/001.jpg 2986449
1038067/007.jpg 1038067
...
```
### Step 3: Run python script with your parameters
For training loss functions you can choose: CosFace, SphereFace, ArcFace, Approach1, Approach2.
```
python train.py --loss_type CosFace --data_root ../cropped_data/webface/casia-112x112 --train_file ../cropped_data/webface/train.txt --out_dir ./weights --epoch 18 --learning_rate 0.1
```
___
## Fine-tuning
For fine-tuning there are 2 approaches: Octuplet Loss and DeriveNet. 

### Step 1: Prepare the training data
Align the face images to 112*112 using [MTCNN](https://github.com/ipazc/mtcnn)

### Step 2: Generate the train paths file
Generate the file with all paths and labels as follows: 
```
2986449/002.jpg 2986449
2986449/001.jpg 2986449
1038067/007.jpg 1038067
...
```

### Step 3: Run python script with your parameters
- For Octuplet Loss:
```
python octuplet.py --data_root ../../cropped_data/webface/casia-112x112 --train_file ../../cropped_data/webface/train.txt --out_dir ../weights --pretrain_model ../weights/ArcFace.pt --epoch 18 --learning_rate 0.1
```
- For DeriveNet:
```
python derive_net.py --data_root ../../cropped_data/webface/casia-112x112 --train_file ../../cropped_data/webface/train.txt --out_dir ../weights --pretrain_model ../weights/ArcFace.pt --epoch 18 --learning_rate 0.1
```
___
## Evaluation 
### Step 1: Prepare the testing data
Align the face images to 112*112 using [MTCNN](https://github.com/ipazc/mtcnn) if using SCface or LFW

### Step 2: Prepare data_conf.yaml file
| Argument     | Description                                                                                                  | 
| -------------|:------------------------------------------------------------------------------------------------------------:| 
| path_model   | Path to weights                                                                                              |
| path_data    | Path to test images                                                                                          | 
| path_pairs   | Path to pairs if using LFW, path to the dir containg positive and negative pairs if using QMUL-SurvFace      |
| path_gallery | Path to high resolution gallery images if using SCface                                                       |
| scface_dist  | Distance to test SCface (can be 1, 2, 3)                                                                     |

### Step 3: Run python script with the argument of testing dataset: LFW, QMUL-SurvFace, SCface 
```
python test.py --dataset SCface
```

## Credits
The code is based on the repositories: 
- https://github.com/JDAI-CV/FaceX-Zoo
- https://github.com/deepinsight/insightface
