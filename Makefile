.PHONY:
.ONESHELL:

MODEL_NAME := test_model

help: ## Show this help and exit
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	

install: ## Install dependencies (Do everytime you start up a paperspace machine)
	pip install --upgrade setuptools wheel
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install tensorboard tensorflow
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 -U
	pip install --upgrade lxml
	apt-get update
	apt -y install -qq aria2

files: folders ## Download the required files (only do once)
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt -d pretrain/hubert
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/Mojobones/base-pt/resolve/main/checkpoint_best_legacy_500.pt -d pretrain/contentvec -o checkpoint_best_legacy_500.pt
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip -d pretrain/nsf_hifigan
	unzip -j nsf_hifigan/nsf_hifigan_20221211.zip "nsf_hifigan/*" -d pretrain/nsf_hifigan

folders:
	echo Creating folders for model $(name)
	mkdir configs/$(MODEL_NAME)
	mkdir datasets
	mkdir datasets/$(MODEL_NAME)
	mkdir datasets/$(MODEL_NAME)/train
	mkdir datasets/$(MODEL_NAME)/val
	mkdir datasets/$(MODEL_NAME)/train/audio
	mkdir datasets/$(MODEL_NAME)/val/audio

extract: ## Unzips the source audio into the model folder
	unzip $(zip) -d datasets/$(MODEL_NAME)/train/audio/

process: ## Preprocesses the file.
	python preprocess.py -c configs/$(MODEL_NAME)/diffusion.yaml

train-ddsp: ## Trains the DDSP, pass in the model name
	python train.py -c configs/$(MODEL_NAME)/combsub.yaml

train-diffusion: ## Trains the diffusion model, pass in the model name
	python train_diff.py -c configs/$(MODEL_NAME)/diffusion.yaml

tensorboard: ## Start the tensorboard (Run on separate terminal)
	echo https://tensorboard-$$(hostname).clg07azjl.paperspacegradient.com
	tensorboard --logdir=exp