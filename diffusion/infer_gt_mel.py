import numpy as np
import torch
import torch.nn.functional as F
from diffusion.unit2mel import load_model_vocoder


class DiffGtMel:
    def __init__(self, project_path=None, device=None):
        self.project_path = project_path
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.vocoder = None
        self.args = None

    def flush_model(self, project_path, ddsp_config=None):
        if (self.model is None) or (project_path != self.project_path):
            model, vocoder, args = load_model_vocoder(project_path, device=self.device)
            if self.check_args(ddsp_config, args):
                self.model = model
                self.vocoder = vocoder
                self.args = args

    def check_args(self, args1, args2):
        if args1.data.block_size != args2.data.block_size:
            raise ValueError("DDSP与DIFF模型的block_size不一致")
        if args1.data.sampling_rate != args2.data.sampling_rate:
            raise ValueError("DDSP与DIFF模型的sampling_rate不一致")
        if args1.data.encoder != args2.data.encoder:
            raise ValueError("DDSP与DIFF模型的encoder不一致")
        return True

    def __call__(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, use_dpm=True,
                 spk_mix_dict=None, start_frame=None):
        input_mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        if use_dpm:
            method = 'dpm-solver'
        else:
            method = 'pndm'
        out_mel = self.model(
            hubert,
            f0,
            volume,
            spk_id=spk_id,
            spk_mix_dict=spk_mix_dict,
            gt_spec=input_mel,
            infer=True,
            infer_speedup=acc,
            method=method,
            k_step=k_step)
        if start_frame is not None:
            out_mel = out_mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
        output = self.vocoder.infer(out_mel, f0)
        if start_frame is not None:
            output = F.pad(output, (int(np.round(self.vocoder.vocoder_sample_rate * (start_frame * self.vocoder.vocoder_hop_size / self.vocoder.vocoder_sample_rate))), 0))
        return output

    def infer(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, use_dpm=True, silence_front=0,
              use_silence=False, spk_mix_dict=None):
        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        real_silence_front = start_frame * self.vocoder.vocoder_hop_size / self.vocoder.vocoder_sample_rate
        if use_silence:
            audio = audio[:, int(np.round(real_silence_front * self.vocoder.vocoder_sample_rate)):]
            f0 = f0[:, start_frame:, :]
            hubert = hubert[:, start_frame:, :]
            volume = volume[:, start_frame:, :]
            _start_frame = None
        else:
            _start_frame = start_frame
        audio = self.__call__(audio, f0, hubert, volume, acc=acc, spk_id=spk_id, k_step=k_step,
                              use_dpm=use_dpm, spk_mix_dict=spk_mix_dict, start_frame=_start_frame)
        if use_silence:
            if start_frame > 0:
                audio = F.pad(audio, (int(np.round(self.vocoder.vocoder_sample_rate * real_silence_front)), 0))
        return audio
