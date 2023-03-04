Language: **English** [简体中文](./cn_README.md)
# DDSP-SVC
<div align="center">
<img src="https://storage.googleapis.com/ddsp/github_images/ddsp_logo.png" width="200px" alt="logo"></img>
</div>
End-to-end singing voice conversion system based on DDSP.

## 1. Installing the dependencies
We recommend first installing PyTorch from the [**official website**](https://pytorch.org/), then run:
```bash
pip install -r requirements.txt 
```
## 2. Configuring the pretrained model
- **(Required)** Download the pretrained [**HubertSoft**](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)   encoder and put it under `pretrain/hubert` folder.
-  Get the pretrained vocoder-based enhancer from the [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) and unzip it into `pretrain/` folder
## 3. Preprocessing

Put all the training dataset (.wav format audio clips) in the below directory:
`data/train/audio`.
Put all the validation dataset (.wav format audio clips) in the below directory:
`data/val/audio`.
Then run
```bash
python preprocess.py -c configs/combsub.yaml
```
for a model of combtooth substractive synthesiser, or run
```bash
python preprocess.py -c configs/sins.yaml
```
for a model of sinusoids additive synthesiser.

You can modify the configuration file `config/<model_name>.yaml` before preprocessing. The default configuration is suitable for training 44.1khz high sampling rate synthesiser with GTX-1660 graphics card.

NOTE 1: Please keep the sampling rate of all audio clips consistent with the sampling rate in the yaml configuration file ! If it is not consistent, the program can be executed safely, but the resampling during the training process will be very slow.

NOTE 2: The total number of the audio clips for training dataset is recommended to be about 1000,  especially long audio clip can be cut into short segments, which will speed up the training, but the duration of all audio clips should not be less than 2 seconds. If there are too many audio clips, you need a large internal-memory or set the 'cache_all_data' option to false in the configuration file.

NOTE 3: The total number of the audio clips for validation dataset is recommended to be about 10, please don't put too many or it will be very slow to do the validation.

## 4. Training
```bash
# train a combsub model as an example
python train.py -c configs/combsub.yaml
```
The command line for training other models is similar.

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## 5. Visualization
```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```
## 6. Testing
```bash
# origin output of ddsp-svc
# fast, but relatively low audio quality
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)>
```
```bash
# enhanced the output using the pretrained vocoder-based enhancer 
# high audio quality in the normal vocal range, but slow
python main.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -e true
```
```bash
# other options about the f0 extractor, see
python main.py -h
```

