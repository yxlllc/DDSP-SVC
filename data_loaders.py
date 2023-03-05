import os
import random
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio)
    loader_train = torch.utils.data.DataLoader(
        data_train ,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers,
        persistent_workers=(args.train.num_workers > 0),
        pin_memory=True
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.train.num_workers,
        persistent_workers=(args.train.num_workers > 0),
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extension='wav',
            is_pure=True,
            is_sort=True,
            is_ext=False
        )
        self.whole_audio = whole_audio
        self.data_buffer={}
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)
        for name in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            
            path_f0 = os.path.join(self.path_root, 'f0', name) + '.npy'
            f0 = np.load(path_f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1)
                
            path_volume = os.path.join(self.path_root, 'volume', name) + '.npy'
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1)
            
            if load_all_data:
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).float()
                
                path_units = os.path.join(self.path_root, 'units', name) + '.npy'
                units = np.load(path_units)
                units = torch.from_numpy(units).float()
                
                self.data_buffer[name] = {
                        'duration': duration,
                        'audio': audio,
                        'units': units,
                        'f0': f0,
                        'volume': volume
                        }
            else:
                self.data_buffer[name] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume
                        }
           

    def __getitem__(self, file_idx):
        name = self.paths[file_idx]
        data_buffer = self.data_buffer[name]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
        
        # get item
        return self.get_data(name, data_buffer)

    def get_data(self, name, data_buffer):
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        
        # load units
        units = data_buffer.get('units')
        if units is None:
            units  = os.path.join(self.path_root, 'units', name) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        f0_frames = f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        volume = data_buffer.get('volume')
        volume_frames = volume[start_frame : start_frame + units_frame_len]
                
        return dict(audio=audio, f0=f0_frames, volume=volume_frames, units=units, name=name)

    def __len__(self):
        return len(self.paths)