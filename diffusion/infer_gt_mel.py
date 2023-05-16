import numpy as np
import os
import torch
import torch.nn.functional as F
from diffusion.unit2mel import load_model_vocoder, DotDict
from ddsp.vocoder import SpeakerEncoder
import librosa
import yaml


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
        self.spk_emb_dict = None
        self.spk_emb = None

    def flush_model(self, project_path, ddsp_config=None, spk_emb_dict_path=None, spk_emb_path=None):
        if (self.model is None) or (project_path != self.project_path):
            # load model
            model, vocoder, args = load_model_vocoder(project_path, device=self.device)
            # check config and init
            if self.check_args(ddsp_config, args):
                self.model = model
                self.vocoder = vocoder
                self.args = args
                if args.model.use_speaker_encoder:
                    args_spk_emb_dict_path = os.path.join(os.path.split(project_path)[0], 'spk_emb_dict.npy')
                    self.spk_emb_dict = np.load(args_spk_emb_dict_path, allow_pickle=True).item()
                    # cover spk_emb
                    if spk_emb_dict_path is not None:
                        self.spk_emb_dict = np.load(spk_emb_dict_path, allow_pickle=True).item()
                        print(f"Load spk_emb_dict from {spk_emb_dict_path}")
                    if spk_emb_path is not None:
                        if spk_emb_path[-4:] == '.npy':
                            _spk_emb = np.load(spk_emb_path)
                            print(f"Load spk_emb from {spk_emb_path} for spk_emb")
                        else:
                            speaker_encoder = SpeakerEncoder(args.data.speaker_encoder, args.data.speaker_encoder_config,
                                                             args.data.speaker_encoder_ckpt,
                                                             args.data.speaker_encoder_sample_rate,
                                                             device=self.device)
                            spk_emb_audio, spk_emb_sample_rate = librosa.load(spk_emb_path, sr=None)
                            if len(spk_emb_audio.shape) > 1:
                                spk_emb_audio = librosa.to_mono(spk_emb_audio)
                            _spk_emb = speaker_encoder(audio=spk_emb_audio, sample_rate=spk_emb_sample_rate)
                            print(f"Load audio from {spk_emb_path} for spk_emb")
                        if len(_spk_emb.shape) > 1:
                            self.spk_emb = np.mean(_spk_emb, axis=0)

    def check_args(self, args1, args2):
        if args1 is None:
            return True
        if args1.data.block_size != args2.data.block_size:
            raise ValueError("DDSP与DIFF模型的block_size不一致")
        if args1.data.sampling_rate != args2.data.sampling_rate:
            raise ValueError("DDSP与DIFF模型的sampling_rate不一致")
        if args1.data.encoder != args2.data.encoder:
            raise ValueError("DDSP与DIFF模型的encoder不一致")
        return True

    def __call__(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, method='pndm',
                 spk_mix_dict=None, start_frame=0, spk_emb=None):
        if self.args.model.use_speaker_encoder:
            spk_id = spk_id
        else:
            spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)

        _spk_emb = None
        if self.spk_emb_dict is not None:
            _spk_emb = self.spk_emb_dict[str(spk_id)]
        if spk_emb is not None:
            _spk_emb = spk_emb
        if self.spk_emb is not None:
            _spk_emb = self.spk_emb
        if _spk_emb is not None:
            _spk_emb = np.tile(_spk_emb, (len(hubert), 1))
            spk_emb = torch.from_numpy(_spk_emb).float().to(hubert.device)

        if audio is not None:
            input_mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        else:
            input_mel = None

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
            k_step=k_step,
            use_tqdm=False,
            spk_emb=spk_emb,
            spk_emb_dict=self.spk_emb_dict)

        if start_frame > 0:
            out_mel = out_mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
        output = self.vocoder.infer(out_mel, f0)
        if start_frame > 0:
            output = F.pad(output, (start_frame * self.vocoder.vocoder_hop_size, 0))
        return output

    def infer(self, audio, f0, hubert, volume, acc=1, spk_id=1, k_step=0, method='pndm', silence_front=0,
              use_silence=False, spk_mix_dict=None, spk_emb=None):
        start_frame = int(silence_front * self.vocoder.vocoder_sample_rate / self.vocoder.vocoder_hop_size)
        if use_silence:
            if audio is not None:
                audio = audio[:, start_frame * self.vocoder.vocoder_hop_size:]
            else:
                k_step = None
            f0 = f0[:, start_frame:, :]
            hubert = hubert[:, start_frame:, :]
            volume = volume[:, start_frame:, :]
            _start_frame = 0
        else:
            _start_frame = start_frame
        audio = self.__call__(audio, f0, hubert, volume, acc=acc, spk_id=spk_id, k_step=k_step,
                              method=method, spk_mix_dict=spk_mix_dict, start_frame=_start_frame, spk_emb=spk_emb)
        if use_silence:
            if start_frame > 0:
                audio = F.pad(audio, (start_frame * self.vocoder.vocoder_hop_size, 0))
        return audio


def get_args_only(model_path):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args
