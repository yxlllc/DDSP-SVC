import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nsf_hifigan.nvSTFT import STFT
from nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from .reflow import RectifiedFlow
from .lynxnet import LYNXNet
from ddsp.vocoder import CombSubSuperFast

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_model_vocoder(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        model = Unit2Wav(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_attention,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.n_chans)
                   
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device = device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device = device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
                
        # resample
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio
        
        
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_attention=False,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=6, 
            n_chans=512):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.ddsp_model = CombSubSuperFast(sampling_rate, block_size, win_length, n_unit, n_spk, use_attention, use_pitch_aug)
        self.reflow_model = RectifiedFlow(LYNXNet(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans), out_dims=out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None,
                gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, 
                silence_front=0, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        ddsp_wav, hidden = self.ddsp_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, infer=infer)
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav[:, start_frame * self.block_size:])
        else:
            ddsp_mel = None
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            reflow_loss = self.reflow_model(ddsp_mel, gt_spec=gt_spec, t_start=t_start, infer=False)
            return ddsp_loss, reflow_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if t_start < 1.0:
                mel = self.reflow_model(ddsp_mel, gt_spec=ddsp_mel, infer=True, infer_step=infer_step, method=method, t_start=t_start, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0[:, -mel.shape[1]:])
            else:
                return mel