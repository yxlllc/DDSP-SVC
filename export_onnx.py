import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
from reflow.reflow import RectifiedFlow
from reflow.lynxnet import LYNXNet
from ddsp.model_conformer_naive import ConformerNaiveEncoder
from onnxruntime import InferenceSession
from nsf_hifigan.nvSTFT import STFT

class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.return_complex = False

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=self.return_complex,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        from librosa.filters import mel as librosa_mel_fn
        mel_basis = torch.from_numpy(librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=self.f_min, fmax=self.f_max))
        self.register_buffer(
            "mel_basis",
            mel_basis,
            persistent=False
        )

    def forward(
        self, x
    ):
        linear = self.spectrogram(x)
        spec = torch.matmul(self.mel_basis, linear)
        return torch.log(torch.clamp(spec, min=1e-5))


class iSTFT(nn.Module):
    def __init__(
            self, win_len=1024, win_hop=512, fft_len=1024,
            window=None, enframe_mode='continue',
            win_sqrt=False
    ):
        """
        Implement of STFT using 1D convolution and 1D transpose convolutions.
        Implement of framing the signal in 2 ways, `break` and `continue`.
        `break` method is a kaldi-like framing.
        `continue` method is a librosa-like framing.
        More information about `perfect reconstruction`:
        1. https://ww2.mathworks.cn/help/signal/ref/stft.html
        2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`. Defaults to 'continue'.
            window (tensor, optional): The window tensor. Defaults to hann window.
            win_sqrt (bool, optional): using square root window. Defaults to True.
        """
        super(iSTFT, self).__init__()
        assert enframe_mode in ['break', 'continue']
        assert fft_len >= win_len
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.mode = enframe_mode
        self.win_sqrt = win_sqrt
        self.pad_amount = self.fft_len // 2

        if window is None:
            window = torch.hann_window(win_len)
        ifft_k, ola_k = self.__init_kernel__(window)
        self.register_buffer('ifft_k', ifft_k, persistent=False)
        self.register_buffer('ola_k', ola_k, persistent=False)

    def __init_kernel__(self, window):
        """
        Generate enframe_kernel, fft_kernel, ifft_kernel and overlap-add kernel.
        ** enframe_kernel: Using conv1d layer and identity matrix.
        ** fft_kernel: Using linear layer for matrix multiplication. In fact,
        enframe_kernel and fft_kernel can be combined, But for the sake of
        readability, I took the two apart.
        ** ifft_kernel, pinv of fft_kernel.
        ** overlap-add kernel, just like enframe_kernel, but transposed.
        Returns:
            tuple: four kernels.
        """
        tmp = torch.fft.rfft(torch.eye(self.fft_len))
        fft_kernel = torch.stack([tmp.real, tmp.imag], dim=2)
        if self.mode == 'break':
            fft_kernel = fft_kernel[:self.win_len]
        fft_kernel = torch.cat(
            (fft_kernel[:, :, 0], fft_kernel[:, :, 1]), dim=1)
        ifft_kernel = torch.pinverse(fft_kernel)[:, None, :]

        if self.mode == 'continue':
            left_pad = (self.fft_len - self.win_len) // 2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = F.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.padded_window = window
            window = torch.sqrt(window)
        else:
            self.padded_window = window ** 2

        ifft_kernel = ifft_kernel * window
        ola_kernel = torch.eye(self.fft_len)[:self.win_len, None, :]
        if self.mode == 'continue':
            ola_kernel = torch.eye(self.fft_len)[:, None, :self.fft_len]
        return ifft_kernel, ola_kernel

    def forward(self, spec):
        """Call the inverse STFT (iSTFT), given tensors produced
        by the `transform` function.
        Args:
            spec (tensors): Input tensor with shape
            complex [num_batch, num_frequencies, num_frames]
            or real [num_batch, num_frequencies, num_frames, 2]
            length (int): Expected number of samples in the output audio.
        Returns:
            tensors: Reconstructed audio given magnitude and phase. Of
                shape [num_batch, num_samples]
        """
        if torch.is_complex(spec):
            real, imag = spec.real, spec.imag
        else:
            assert spec.size(-1) == 2
            real, imag = spec[..., 0], spec[..., 1]

        inputs = torch.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.ifft_k, stride=self.win_hop)
        t = (self.padded_window[None, :, None]).repeat(1, 1, inputs.size(-1))
        t = t.to(inputs.device)
        coff = F.conv_transpose1d(t, self.ola_k, stride=self.win_hop)
        rm_start, rm_end = self.pad_amount, -self.pad_amount
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coff = torch.where(coff > 1e-8, coff, torch.ones_like(coff))
        outputs /= coff
        return outputs.squeeze(dim=1)


def split_to_(tensor, tensor_splits):
    tensors = torch.split(tensor, tensor_splits, dim=-1)
    return tensors


class Unit2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            block_size,
            n_spk,
            output_splits,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, dim_model)
        self.phase_embed = nn.Linear(1, dim_model)
        self.volume_embed = nn.Linear(1, dim_model)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, dim_model)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, dim_model, bias=False)
        else:
            self.aug_shift_embed = None
            
        self.stack = nn.Sequential(
                weight_norm(nn.Conv1d(input_channel, 512, 3, 1, 1)),
                nn.PReLU(num_parameters=512),
                weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.stack2 = nn.Sequential(
                weight_norm(nn.Conv1d(2 * block_size, 512, 3, 1, 1)),
                nn.PReLU(num_parameters=512),
                weight_norm(nn.Conv1d(512, dim_model, 3, 1, 1)))
        self.decoder = ConformerNaiveEncoder(
                num_layers=num_layers,
                num_heads=8,
                dim_model=dim_model,
                use_norm=use_norm,
                conv_only=not use_attention,
                conv_dropout=0,
                atten_dropout=0.1)
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.o_sp_k = [k for k, v in output_splits.items()]
        self.o_sp = [v for k, v in output_splits.items()]
        self.dense_out = weight_norm(nn.Linear(dim_model, self.n_out))
        self.gin_channels = dim_model
        
    def export_chara_mix(self, n_spk):
        speaker_map = torch.zeros((n_spk, 1, 1, self.gin_channels))
        for i in range(n_spk):
            speaker_map[i] = self.spk_embed(torch.LongTensor([[i]]))
        speaker_map = speaker_map.unsqueeze(0)
        self.register_buffer("speaker_map", speaker_map)

    def forward(self, units, source, noise, volume, g):
        exciter = torch.cat((source, noise), dim=-1).transpose(1,2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)

        if self.n_spk is not None and self.n_spk > 1:
            g = g.permute(2, 0, 1)  # [S, B, N]
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2], 1))  # [1, S, B, N, 1]
            g = g * self.speaker_map  # [1, S, B, N, H]
            g = torch.sum(g, dim=1).squeeze(0) # [B, N, H]
            x = x + g

        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        return split_to_(e, self.o_sp)


class CombSubSuperFast(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            win_length,
            n_unit=256,
            n_spk=1,
            num_layers=3,
            dim_model=256,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            f0_min = 65):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("win_length", torch.tensor(win_length))
        self.register_buffer("window", torch.hann_window(win_length))
        self.sr = sampling_rate
        self.bs = block_size
        self.wl = win_length
        #Unit2Control
        split_map = {
            'harmonic_magnitude': win_length // 2 + 1, 
            'harmonic_phase': win_length // 2 + 1,
            'noise_magnitude': win_length // 2 + 1,
            'noise_phase': win_length // 2 + 1
        }
        self.unit2ctrl = Unit2Control(
                            n_unit, 
                            block_size, 
                            n_spk, 
                            split_map,
                            num_layers=num_layers,
                            dim_model=dim_model,
                            use_norm=use_norm,
                            use_attention=use_attention, 
                            use_pitch_aug=use_pitch_aug)
        
        self.istft_method = iSTFT(
            win_len = win_length,
            win_hop = block_size,
            fft_len = win_length,
            window = self.window
        )

        self.melext = LogMelSpectrogram(
            sampling_rate,
            win_length,
            win_length,
            block_size,
            128,
            False,
            40,
            16000,
        )

        self.spec_max = 2
        self.spec_min = -12
        self.f0_min = f0_min
    
    def fast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = self.msinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
    
    @staticmethod
    def msinc(input):
        input = np.pi*input
        output = torch.sin(input)/input
        #output[torch.abs(input) < 1e-5] = 1.0
        return output
    
    @staticmethod
    def complex_exp(real, imag):
        mod = torch.exp(real)
        rp = torch.cos(imag)
        ip = torch.sin(imag)
        return torch.cat((rp, ip),dim=-1)*mod
    
    @staticmethod
    def complex_mul(left, right):
        assert(len(left.shape) == len(right.shape))
        real = (left[:,:,:,0] * right[:,:,:,0] - left[:,:,:,1] * right[:,:,:,1]).unsqueeze(-1)
        imag = (left[:,:,:,0] * right[:,:,:,1] + left[:,:,:,1] * right[:,:,:,0]).unsqueeze(-1)
        return torch.cat((real, imag),dim=-1)
    
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1
    
    def sfast_source_gen(self, f0_frames):
        n = torch.arange(self.block_size, device=f0_frames.device)
        s0 = f0_frames / self.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / self.block_size
        s0 = s0 + ds0 * n / self.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad -= torch.round(rad)
        combtooth = torch.sinc(rad / (s0 + 1e-5)).reshape(f0_frames.shape[0], -1)
        return combtooth
    
    def sforward(self, units_frames, mel2ph, f0_frames, volume_frames, g=None, noise=None):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''

        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)
        
        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.unfold(1, self.block_size, self.block_size)
        
        noise_frames = noise.unfold(1, self.block_size, self.block_size)
        
        # parameter prediction
        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = self.unit2ctrl(
            units_frames, combtooth_frames, noise_frames, volume_frames, g
        )
        
        src_filter = torch.exp(harmonic_magnitude + 1.j * np.pi * harmonic_phase)
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(noise_magnitude + 1.j * np.pi * noise_phase) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.win_length,
                            win_length = self.win_length,
                            hop_length = self.block_size,
                            window = self.window,
                            center = True,
                            return_complex = True,
                            pad_mode = pad_mode)
        
        # apply the filters 
        signal_stft = combtooth_stft * src_filter.permute(0, 2, 1) + noise_stft * noise_filter.permute(0, 2, 1)
        
        # take the istft to resynthesize audio.
        signal = torch.istft(
                        signal_stft,
                        n_fft = self.win_length,
                        win_length = self.win_length,
                        hop_length = self.block_size,
                        window = self.window,
                        center = True)
        
        return STFT(self.sr, 128, self.wl, self.wl, self.bs).get_mel(signal)
    
    def tforward(self, units_frames, mel2ph, f0_frames, volume_frames, g=None, noise=None):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames
            volume_frames: B x n_frames
        '''
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)
        
        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.view(1, -1, self.block_size)
        
        noise_frames = noise.view(1, -1, self.block_size)

        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = self.unit2ctrl(
            units_frames, combtooth_frames, noise_frames, volume_frames, g
        )

        src_filter = self.complex_exp(
            harmonic_magnitude.unsqueeze(-1),
            torch.ones_like(harmonic_magnitude, dtype=torch.float32).unsqueeze(-1) * np.pi * harmonic_phase.unsqueeze(-1)
        )
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)

        noise_filter = self.complex_exp(
            noise_magnitude.unsqueeze(-1),
            torch.ones_like(noise_magnitude, dtype=torch.float32).unsqueeze(-1) * np.pi * noise_phase.unsqueeze(-1)
        ) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.wl,
                            win_length = self.wl,
                            hop_length = self.bs,
                            window = self.window,
                            center = True,
                            return_complex = False,
                            pad_mode = pad_mode)
        
        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.wl,
                            win_length = self.wl,
                            hop_length = self.bs,
                            window = self.window,
                            center = True,
                            return_complex = False,
                            pad_mode = pad_mode)
        
        # apply the filters 
        signal_stft = self.complex_mul(combtooth_stft, src_filter.permute(0, 2, 1, 3)) + self.complex_mul(noise_stft, noise_filter.permute(0, 2, 1, 3))
        
        signal = self.istft_method(signal_stft)

        return self.melext(signal)
        
    def forward(self, units_frames, mel2ph, f0_frames, volume_frames, g=None, noise=None):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames
            volume_frames: B x n_frames
        '''
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units_frames.shape[-1]])
        units_frames = torch.gather(units_frames, 1, mel2ph_)
        
        volume_frames = volume_frames.unsqueeze(-1)

        combtooth = self.fast_source_gen(f0_frames.unsqueeze(-1))
        combtooth_frames = combtooth.view(1, -1, self.block_size)
        
        noise_frames = noise.view(1, -1, self.block_size)

        harmonic_magnitude, harmonic_phase, noise_magnitude, noise_phase = self.unit2ctrl(
            units_frames, combtooth_frames, noise_frames, volume_frames, g
        )

        src_filter = self.complex_exp(
            harmonic_magnitude.unsqueeze(-1),
            torch.ones_like(harmonic_magnitude, dtype=torch.float32).unsqueeze(-1) * np.pi * harmonic_phase.unsqueeze(-1)
        )
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)

        noise_filter = self.complex_exp(
            noise_magnitude.unsqueeze(-1),
            torch.ones_like(noise_magnitude, dtype=torch.float32).unsqueeze(-1) * np.pi * noise_phase.unsqueeze(-1)
        ) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # harmonic part filter
        if combtooth.shape[-1] > self.win_length // 2:
            pad_mode = 'reflect'
        else:
            pad_mode = 'constant'
        combtooth_stft = torch.stft(
                            combtooth,
                            n_fft = self.wl,
                            win_length = self.wl,
                            hop_length = self.bs,
                            window = self.window,
                            center = True,
                            return_complex = False,
                            pad_mode = pad_mode)
        
        # noise part filter
        noise_stft = torch.stft(
                            noise,
                            n_fft = self.wl,
                            win_length = self.wl,
                            hop_length = self.bs,
                            window = self.window,
                            center = True,
                            return_complex = False,
                            pad_mode = pad_mode)
        
        # apply the filters 
        signal_stft = self.complex_mul(combtooth_stft, src_filter.permute(0, 2, 1, 3)) + self.complex_mul(noise_stft, noise_filter.permute(0, 2, 1, 3))
        
        signal = self.istft_method(signal_stft)
        mel = self.melext(signal)

        return self.norm_spec(mel).unsqueeze(1), mel


class Unit2Wav(nn.Module):
    def __init__(
            self,
            sampling_rate,
            block_size,
            win_length,
            n_unit,
            n_spk,
            use_norm=False,
            use_attention=False,
            use_pitch_aug=False,
            out_dims=128,
            n_aux_layers=3,
            n_aux_chans=256,
            n_layers=6, 
            n_chans=512,
            f0_min=65):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.ddsp_model = CombSubSuperFast(
                            sampling_rate, 
                            block_size, 
                            win_length, 
                            n_unit, 
                            n_spk, 
                            n_aux_layers if n_aux_layers is not None else 3,
                            n_aux_chans if n_aux_chans is not None else 256,
                            use_norm,
                            use_attention, 
                            use_pitch_aug,
                            f0_min)
        self.reflow_model = RectifiedFlow(LYNXNet(in_dims=out_dims, dim_cond=out_dims, n_layers=n_layers, n_chans=n_chans), out_dims=out_dims)


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


import os
import yaml


class After(nn.Module):
    def __init__(self, log = False):
        super().__init__()
        self.spec_max = 2
        self.spec_min = -12
        self.log = log

    def forward(self, x):
        x = x.squeeze(1)
        x = (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
        if self.log:
            x *= 0.434294
        return x


def export_onnx(model_path, output_path):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = Unit2Wav(
        args.data.sampling_rate,
        args.data.block_size,
        args.model.win_length,
        args.data.encoder_out_channels, 
        args.model.n_spk,
        args.model.use_norm,
        args.model.use_attention,
        args.model.use_pitch_aug,
        128,
        args.model.n_aux_layers,
        args.model.n_aux_chans,
        args.model.n_layers,
        args.model.n_chans,
        args.data.f0_min)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.to('cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    frame_c = 25
    hu = torch.randn((1, frame_c, args.data.encoder_out_channels))
    mel2ph = torch.arange(0, frame_c).long().unsqueeze(0)
    f0 = torch.randn(1, frame_c)
    vol = torch.randn(1, frame_c)
    randn_input = torch.randn(1, frame_c * model.block_size)

    n_spk = args.model.n_spk
    if n_spk is not None and n_spk > 1:
        spk_mix = []
        for _ in range(n_spk):
            spk_mix.append(1.0/float(n_spk))
        test_sid = torch.tensor(spk_mix)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(frame_c, 1).unsqueeze(0)
        model.ddsp_model.unit2ctrl.export_chara_mix(n_spk)
        outtest = model.ddsp_model(hu, mel2ph, f0, vol, test_sid, randn_input)
        a = model.ddsp_model.sforward(hu, mel2ph, f0, vol, test_sid, randn_input)
        b = model.ddsp_model.tforward(hu, mel2ph, f0, vol, test_sid, randn_input)
        print(torch.max(torch.abs(b - a)))
        #print(torch.sum(torch.abs(outtest[0] - model.ddsp_model.sforward(hu, mel2ph, f0, vol, test_sid, randn_input)[0])))
        torch.onnx.export(
            model.ddsp_model,
            (hu, mel2ph, f0, vol, test_sid, randn_input),
            f"{output_path}/encoder.onnx",
            dynamic_axes={
                "hubert" : {0: "batch", 1: "frame"},
                "mel2ph" : {0: "batch", 1: "frame"},
                "f0" : {0: "batch", 1: "frame"},
                "volume": {0: "batch", 1: "frame"},
                "spk_mix": {0: "batch", 1: "frame"},
                "randn": {0: "batch", 1: "audio_length"}
            },
            do_constant_folding=False,
            opset_version=18,
            verbose=False,
            input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix", "randn"],
            output_names=["x", "cond"]
        )

        import onnxruntime as ort
        sess = ort.InferenceSession(f"{output_path}/encoder.onnx")
        test = model.ddsp_model.forward(hu, mel2ph, f0, vol, test_sid, randn_input)
        res = sess.run(None, {
            "hubert": hu.numpy(),
            "mel2ph": mel2ph.numpy(),
            "f0": f0.numpy(),
            "volume": vol.numpy(),
            "spk_mix": test_sid.numpy(),
            "randn": randn_input.numpy()
        })
        print(torch.max(torch.abs(torch.tensor(res[0]) - test[0])))
    else:
        outtest = model.ddsp_model(hu, mel2ph, f0, vol, test_sid, randn_input)
        torch.onnx.export(
            model.ddsp_model,
            (hu, mel2ph, f0, vol, test_sid, randn_input),
            f"{output_path}/encoder.onnx",
            dynamic_axes={
                "hubert" : {1: "frame"},
                "mel2ph" : {1: "frame"},
                "f0" : {1: "frame"},
                "volume": {1: "frame"},
                "randn": {1: "audio_length"}
            },
            do_constant_folding=False,
            opset_version=18,
            verbose=False,
            input_names=["hubert", "mel2ph", "f0", "volume", "randn"],
            output_names=["x", "cond"]
        )

    t = torch.tensor([0], dtype=torch.int64)
    torch.onnx.export(
            model.reflow_model.velocity_fn,
            (outtest[0].cpu(), t.cpu(), outtest[1].cpu()),
            f"{output_path}/velocity.onnx",
            input_names=["x", "t", "cond"],
            output_names=["o"],
            dynamic_axes={
                "x": {3: "frame"},
                "cond": {2: "frame"}
            },
            opset_version=18
        )
    
    return args.data.encoder_out_channels, model.block_size, n_spk


if __name__ == "__main__":
    oc, bs, ns = export_onnx("model/model_170000.pt", "model")
    
    ortmodel = InferenceSession("model/encoder.onnx")
    velocity = InferenceSession("model/velocity.onnx")

    frame_c = 25
    hu = torch.randn((1, frame_c, oc))                      # units feature, 1 x frame x units
    mel2ph = torch.arange(0, frame_c).long().unsqueeze(0)   # alignment idx, units -> f0, eg. units(1, 5, ...) -> f0(1, 10) [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    f0 = torch.randn(1, frame_c)                            # f0, 1 x frame
    vol = torch.randn(1, frame_c)                           # volume, 1 x frame
    randn_input = torch.randn(1, frame_c * bs)              # noise input, 1 x frame x model.block_size512
    
    if ns is not None and ns > 1:
        spk_mix = []
        for _ in range(ns):
            spk_mix.append(1.0/float(ns))
        test_sid = torch.tensor(spk_mix)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(frame_c, 1).unsqueeze(0) # 1 x frame x n_spk [[speaker1, speaker2, ...], [speaker1, speaker2, ...], ...]
        ortinput = {
            "hubert": hu.numpy(),
            "mel2ph": mel2ph.numpy(),
            "f0": f0.numpy(),
            "volume": vol.numpy(),
            "spk_mix": test_sid.numpy(),
            "randn": randn_input.numpy()
        }
    else:
        ortinput = {
            "hubert": hu.numpy(),
            "mel2ph": mel2ph.numpy(),
            "f0": f0.numpy(),
            "volume": vol.numpy(),
            "randn": randn_input.numpy()
        }
    ortout = ortmodel.run(None, ortinput)
    x = ortout[0]
    cond = ortout[1]
    dt = 0.05
    t = np.array([0.0], dtype=np.float32)

    '''
    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, 1000 * t, cond) * dt
        t += dt
        return x, t
    '''
    while t[0] <= 1.0:
        ortout = velocity.run(None, {
            "x": x,
            "t": (t * 1000).astype(np.int64),
            "cond": cond
        })
        x += ortout[0] * dt
        t += dt

    pass
    # audio = vocoder(x, f0)
    

    
