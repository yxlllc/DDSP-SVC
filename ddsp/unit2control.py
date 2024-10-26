import gin

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .model_conformer_naive import ConformerNaiveEncoder


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            block_size,
            n_spk,
            output_splits,
            num_layers=3,
            dim_model=256,
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
                use_norm=False,
                conv_only=not use_attention,
                conv_dropout=0,
                atten_dropout=0.1)
        self.norm = nn.LayerNorm(dim_model)
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(dim_model, self.n_out))

    def forward(self, units, source, noise, volume, spk_id = None, spk_mix_dict = None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        exciter = torch.cat((source, noise), dim=-1).transpose(1,2)
        x = self.stack(units.transpose(1, 2)) + self.stack2(exciter)
        x = x.transpose(1, 2) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x