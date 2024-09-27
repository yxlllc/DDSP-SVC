Language: **English** [简体中文](./cn_README.md)

# DDSP-SVC

## 0. Introduction

DDSP-SVC is a new open source singing voice conversion project dedicated to the development of free AI voice changer software that can be popularized on personal computers.

Compared with the famous [SO-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc), its training and synthesis have much lower requirements for computer hardware, and the training time can be shortened by orders of magnitude, which is close to the training speed of [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

In addition, when performing real-time voice changing, the hardware resource consumption of this project is significantly lower than that of SO-VITS-SVC，but probably slightly higher than the latest version of RVC.

Although the original synthesis quality of DDSP is not ideal (the original output can be heard in tensorboard while training), after enhancing the sound quality with a pre-trained vocoder based enhancer (old version) or with a shallow diffusion model (new version) , for some datasets, it can achieve the synthesis quality no less than SOVITS-SVC and RVC.

The old version models are still compatible, the following chapters are the instructions for the old version. Some operations of the new version are the same, see the previous chapters.

Disclaimer: Please make sure to only train DDSP-SVC models with **legally obtained authorized data**, and do not use these models and any audio they synthesize for illegal purposes. The author of this repository is not responsible for any infringement, fraud and other illegal acts caused by the use of these model checkpoints and audio.

Update log: I am too lazy to translate, please see the Chinese version readme.

## 1. Installing the dependencies

We recommend first installing PyTorch from the [official website](https://pytorch.org/), then run:

```bash
pip install -r requirements.txt
```

python 3.8 (windows) + cuda 11.8 + torch 2.0.0 + torchaudio 2.0.1

## 2. Configuring the pretrained model

- Feature Encoder (choose only one):

(1) Download the pre-trained [ContentVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) encoder and put it under `pretrain/contentvec` folder.

(2) Download the pre-trained [HubertSoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) encoder and put it under `pretrain/hubert` folder, and then modify the configuration file at the same time.

- Vocoder:

Download and unzip the pre-trained [NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip) vocoder 

or use the https://github.com/openvpi/SingingVocoders project to fine-tune the vocoder for higher sound quality.

Then rename the checkpoint file and place it at the location specified by the 'vocoder.ckpt' parameter in the configuration file. The default value is `pretrain/nsf_hifigan/model`.

The 'config.json' of the vocoder needs to be at the same directory, for example, `pretrain/nsf_hifigan/config.json`.

- Pitch extractor:

Download the pre-trained [RMVPE](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) extractor and unzip it into `pretrain/` folder.

## 3. Preprocessing

Put all the training dataset (.wav format audio clips) in the below directory: `data/train/audio`. Put all the validation dataset (.wav format audio clips) in the below directory: `data/val/audio`. You can also run

```bash
python draw.py
```

to help you select validation data (you can adjust the parameters in `draw.py` to modify the number of extracted files and other parameters)

Then run the preprocessor:

```bash
python preprocess.py -c configs/reflow.yaml
```

NOTE 1: The default configuration is suitable for with GTX-1660 graphics card.

NOTE 2: Please keep the sampling rate of all audio clips consistent with the sampling rate in the yaml configuration file ! If it is not consistent, the program can be executed safely, but the resampling during the training process will be very slow.

NOTE 3: The total number of the audio clips for training dataset is recommended to be about 1000, especially long audio clip can be cut into short segments, which will speed up the training, but the duration of all audio clips should not be less than 2 seconds. If there are too many audio clips, you need a large internal-memory or set the 'cache_all_data' option to false in the configuration file.

NOTE 4: The total number of the audio clips for validation dataset is recommended to be about 10, please don't put too many or it will be very slow to do the validation.

NOTE 5: If your dataset is not very high quality, set 'f0_extractor' to 'rmvpe' in the config file.

NOTE 6: Multi-speaker training is supported now. The 'n_spk' parameter in configuration file controls whether it is a multi-speaker model. If you want to train a **multi-speaker** model, audio folders need to be named with **positive integers not greater than 'n_spk'** to represent speaker ids, the directory structure is like below:

```bash
# training dataset
# the 1st speaker
data/train/audio/1/aaa.wav
data/train/audio/1/bbb.wav
...
# the 2nd speaker
data/train/audio/2/ccc.wav
data/train/audio/2/ddd.wav
...

# validation dataset
# the 1st speaker
data/val/audio/1/eee.wav
data/val/audio/1/fff.wav
...
# the 2nd speaker
data/val/audio/2/ggg.wav
data/val/audio/2/hhh.wav
...
```

If 'n_spk' \= 1, The directory structure of the **single speaker** model is still supported, which is like below:

```bash
# training dataset
data/train/audio/aaa.wav
data/train/audio/bbb.wav
...
# validation dataset
data/val/audio/ccc.wav
data/val/audio/ddd.wav
...
```

## 4. Training

```bash
# train a combsub model as an example
python train_reflow.py -c configs/reflow.yaml
```

After training starts, a weight is temporarily saved every ‘interval_val’ step, and a weight is permanently saved every ‘interval_force_save’ step. These two configuration items can be modified according to the situation.

You can safely interrupt training, then running the same command line will resume training.

You can also finetune the model if you interrupt training first, then re-preprocess the new dataset or change the training parameters (batchsize, lr etc.) and then run the same command line.

## 5. Visualization

```bash
# check the training status using tensorboard
tensorboard --logdir=exp
```

Test audio samples will be visible in TensorBoard after the first validation.


## 6. Non-real-time VC

```bash
python main_reflow.py -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -id <speaker_id> -step <infer_step> -method <method> -ts <t_start>
```

'infer_step' is the number of sampling steps for rectified-flow ODE, 'method' is 'euler' or 'rk4', 't_start' is the start time point of ODE, which needs to be larger than or equal to t_start in the configuration file, it is recommended to keep it equal (the default is 0.7).

You can use "-mix" option to design your own vocal timbre, below is an example:

```bash
# Mix the timbre of 1st and 2nd speaker in a 0.5 to 0.5 ratio
python main_reflow.py -i <input.wav> -m <model_file.pt> -o <output.wav> -k <keychange (semitones)> -mix "{1:0.5, 2:0.5}" -eak 0
```

Other options about the f0 extractor and response threhold，see:

```bash
python main_reflow.py -h
```

## 7. Real-time VC

Start a simple GUI with the following command:

```bash
python gui_reflow.py
```

The front-end uses technologies such as sliding window, cross-fading, SOLA-based splicing and contextual semantic reference, which can achieve sound quality close to non-real-time synthesis with low latency and resource occupation.

## 8. Acknowledgement

- [ddsp](https://github.com/magenta/ddsp)

- [pc-ddsp](https://github.com/yxlllc/pc-ddsp)

- [soft-vc](https://github.com/bshall/soft-vc)

- [ContentVec](https://github.com/auspicious3000/contentvec)

- [DiffSinger (OpenVPI version)](https://github.com/openvpi/DiffSinger)

- [Diff-SVC](https://github.com/prophesier/diff-svc)

- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)
