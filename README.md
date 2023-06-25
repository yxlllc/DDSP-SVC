## 1. Introduction
This is a branch of [DDSP_SVC](https://github.com/yxlllc/DDSP-SVC) that adds a Makefile to DDSP-SVC to allow it to more easily run on machines like Paperspace. 

## 2. Usage

1. Run "make install" to install the python depencies. You will need to run this every time the Paperspace machine runs.
2. Run "make files" to download the required files. You will only need to run this one, as the files will persist across Paperspace runs.
3. Run "make folders name=<model_name>" to create the folder structure
4. Upload your audio files into the 'datasets/<model_name>/train/audio' directory, and upload your config files into the 'configs/<model_name>' directory
5. Run "make preprocess name=<model_name>" to pre-process the audio
6. Open a second terminal and run "make tensorboard" to run the Tensorboard
7. Run "make train-ddps name=<model_name>" to train the DDSP model
8. Run "make train-diffusion name=<model_name>" to train the Diffusion model

## 2. Acknowledgement
* [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)
* [ddsp](https://github.com/magenta/ddsp)
* [pc-ddsp](https://github.com/yxlllc/pc-ddsp)
* [soft-vc](https://github.com/bshall/soft-vc)
* [ContentVec](https://github.com/auspicious3000/contentvec)
* [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)
* [Diff-SVC](https://github.com/prophesier/diff-svc)

