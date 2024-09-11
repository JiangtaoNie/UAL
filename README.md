# UAL-Unsupervised Adaptation Learning & UTAL-Unsupervised Test-time Adaptation Learning

1. Unsupervised Adaptation Learning for Hyperspectral Imagery Super-resolution

The main process of the proposed UAL framework contains two steps, first is use the "Train_FusionModel.py" to train the Fusionmodel. Second, we utilize the pre-trained Fusionmodel to provide the initial input of the adaptor network and then we train the adaptor under unsupervised mode to generate the specific reconstructed HR HSI.

2. Unsupervised Test-Time Adaptation Learning for Effective Hyperspectral Image Super-Resolution With Unknown Degeneration

The main process of the proposed UTAL framework is same as UAL. To implement the meta-al-UTAL, you need to train the fusion model with "meta-al-UTAL/Train_FusionModel.py". Then, you should conduct meta train on the adaptor. At last, optimizing the adaptor to fit different testing samples.
