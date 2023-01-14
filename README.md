# LowResolutionFaceRecognition
Implementation of CosFace, SphereFace, ArcFace, OctupletLoss, DeriveNet and 2 proposed approaches 
'''
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

19 directories, 35 files'''
