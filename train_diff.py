import os
import argparse
import torch
from torch.optim import lr_scheduler
from logger import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel
from diffusion.vocoder import Vocoder


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                use_speaker_encoder=args.model.use_speaker_encoder,
                speaker_encoder_out_channels=args.data.speaker_encoder_out_channels)
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid)
    
