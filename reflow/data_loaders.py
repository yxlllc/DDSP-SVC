from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import random
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

from logger.utils import Timer, load_config


def traverse_dir(
    root_dir,
    extensions,
    amount=None,
    str_include=None,
    str_exclude=None,
    is_pure=False,
    is_sort=False,
    is_ext=True,
):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1 :] if is_pure else mix_path

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
                    ext = pure_path.split(".")[-1]
                    pure_path = pure_path[: -(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    use_big_chunks = args.data.get("use_big_chunks", False)
    if use_big_chunks:
        print("Using big chunks version")
    data_train = (AudioDatasetBigChunkVer if use_big_chunks else AudioDataset)(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=["npz"],
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True,
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device == "cpu" else 0,
        persistent_workers=(args.train.num_workers > 0)
        if args.train.cache_device == "cpu"
        else False,
        pin_memory=True if args.train.cache_device == "cpu" else False,
    )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=["npz"],
        n_spk=args.model.n_spk,
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
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
        extensions=["wav"],
        n_spk=1,
        device="cpu",
        fp16=False,
        use_aug=False,
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, "features"),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True,
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer = {}

        if load_all_data:
            print("Load all the data from :", path_root)
        else:
            print("Load the f0, volume data from :", path_root)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            print("Appending futures...")
            for name_ext in self.paths:
                futures.append(
                    executor.submit(
                        self.load_feature,
                        name_ext,
                        device,
                        n_spk,
                        load_all_data,
                        fp16,
                    )
                )
            print("Waiting for futures...")
            for future in tqdm(futures, total=len(futures)):
                name_ext, data = future.result()
                self.data_buffer[name_ext] = data

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[self.remove_npz_suffix(name_ext)]
        # check duration. if too short, then skip
        if data_buffer["duration"] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(name_ext, data_buffer)

    def load_feature(
        self,
        name_ext,
        device,
        n_spk,
        load_all_data,
        fp16,
    ):
        path_features = os.path.join(self.path_root, "features", name_ext)
        features = np.load(path_features, allow_pickle=True)

        duration = self.format_feature(features["duration"], device=device)
        f0 = self.format_feature(features["f0"], device=device)
        volume = self.format_feature(features["volume"], device=device)
        aug_vol = self.format_feature(features["aug_vol"], device=device)
        if n_spk is not None and n_spk > 1:
            dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
            spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
            if spk_id < 1 or spk_id > n_spk:
                raise ValueError(
                    " [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk "
                )
        else:
            spk_id = 1
        spk_id = torch.LongTensor(np.array([spk_id])).to(device)

        data = {
            "duration": duration,
            "f0": f0,
            "volume": volume,
            "aug_vol": aug_vol,
            "spk_id": spk_id,
        }

        if load_all_data:
            mel = self.format_feature(features["mel"], device=device, unsqueeze=False)
            aug_mel = self.format_feature(
                features["aug_mel"], device=device, unsqueeze=False
            )
            units = self.format_feature(
                features["units"], device=device, unsqueeze=False
            )

            if fp16:
                mel = mel.half()
                aug_mel = aug_mel.half()
                units = units.half()

            data.update(
                {
                    "mel": mel,
                    "aug_mel": aug_mel,
                    "units": units,
                }
            )

        return self.remove_npz_suffix(name_ext), data

    def remove_npz_suffix(self, name_ext):
        return os.path.splitext(name_ext)[0]

    def format_feature(
        self,
        feature: np.ndarray,
        device: str = None,
        unsqueeze: bool = True,
    ):
        tmp = torch.from_numpy(feature).float()
        if device:
            tmp = tmp.to(device)
        if unsqueeze:
            tmp = tmp.unsqueeze(-1)
        return tmp

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer["duration"]
        waveform_sec = duration if self.whole_audio else self.waveform_sec

        # load audio
        idx_from = (
            0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        )
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        """
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
        """
        # load mel
        mel_key = "aug_mel" if aug_flag else "mel"
        mel = data_buffer.get(mel_key)
        features = None

        if mel is None:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    os.path.join(self.path_root, "features", name_ext),
                    allow_pickle=True,
                )

            mel = self.format_feature(features["mel"], unsqueeze=False)
            mel = mel[start_frame : start_frame + units_frame_len]
        else:
            mel = mel[start_frame : start_frame + units_frame_len]

        # load units
        units = data_buffer.get("units")
        if units is None:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    os.path.join(self.path_root, "features", name_ext),
                    allow_pickle=True,
                )

            units = self.format_feature(features["units"], unsqueeze=False)

            units = units[start_frame : start_frame + units_frame_len]
            # units = torch.from_numpy(units).float()
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get("f0")
        aug_shift = 0
        if aug_flag:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    os.path.join(self.path_root, "features", name_ext),
                    allow_pickle=True,
                )
            aug_shift = features["aug_shift"]
        f0_frames = (
            2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        )

        # load volume
        vol_key = "aug_vol" if aug_flag else "volume"
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]

        # load spk_id
        spk_id = data_buffer.get("spk_id")

        aug_shift = self.format_feature(np.array([[aug_shift]]), unsqueeze=False)

        return dict(
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            aug_shift=aug_shift,
            name=name,
            name_ext=name_ext,
        )

    def __len__(self):
        return len(self.paths)


class AudioDatasetBigChunkVer(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=["wav"],
        n_spk=1,
        device="cpu",
        fp16=False,
        use_aug=False,
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.chunks_root = os.path.join(path_root, "chunks")

        self.paths = traverse_dir(
            os.path.join(path_root, "chunks"),
            extensions=["npz"],
        )

        self.features_paths = []

        chunk_info = load_config(os.path.join(self.chunks_root, "info.yaml"))
        self.num_features = chunk_info.num_features
        self.num_chunks = chunk_info.num_chunks
        self.last_chunk_length = chunk_info.last_chunk_length
        self.num_features_per_chunk = chunk_info.num_features_per_chunk

        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer = []

        self.device = device
        self.n_spk = n_spk
        self.load_all_data = load_all_data
        self.fp16 = fp16

        if load_all_data:
            print("Load all the data from :", path_root)
        else:
            print("Load the f0, volume data from :", path_root)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            print("Appending futures...")
            for chunk_path in self.paths:
                futures.append(
                    executor.submit(
                        self.load_chunk,
                        chunk_path,
                    )
                )
            print("Waiting for futures...")
            for future in tqdm(futures, total=len(futures)):
                data = future.result()
                self.data_buffer.extend(data)
                for d in data:
                    self.features_paths.append(d["features_path"])

    def __getitem__(self, file_idx):
        data_buffer = self.data_buffer[file_idx]

        if data_buffer["duration"] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % self.num_features)

        # get item
        return self.get_data(file_idx, data_buffer)

    def load_chunk(
        self,
        chunk_path,
    ):
        device = self.device
        n_spk = self.n_spk
        load_all_data = self.load_all_data
        fp16 = self.fp16

        """
        返回的是一个 list，list 中的每个元素是一个 dict，dict 就是 features
        """
        result = []
        chunk = np.load(chunk_path, allow_pickle=True)
        for features in chunk["data"]:
            duration = self.format_feature(features["duration"], device=device)
            f0 = self.format_feature(features["f0"], device=device)
            volume = self.format_feature(features["volume"], device=device)
            aug_vol = self.format_feature(features["aug_vol"], device=device)
            if n_spk is not None and n_spk > 1:
                spk_id = features["spk_id"]
            else:
                spk_id = 1

            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            data = {
                "duration": duration,
                "f0": f0,
                "volume": volume,
                "aug_vol": aug_vol,
                "spk_id": spk_id,
                "features_path": features["features_path"],
            }

            if load_all_data:
                mel = self.format_feature(
                    features["mel"], device=device, unsqueeze=False
                )
                aug_mel = self.format_feature(
                    features["aug_mel"], device=device, unsqueeze=False
                )
                units = self.format_feature(
                    features["units"], device=device, unsqueeze=False
                )

                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()

                data.update(
                    {
                        "mel": mel,
                        "aug_mel": aug_mel,
                        "units": units,
                    }
                )
            result.append(data)
        return result

    def load_features_by_features_idx(
        self,
        idx,
    ):
        # 输入的 idx 是 features 的 idx，而不是 chunk 的 idx
        chunk_idx = idx // self.num_features_per_chunk
        chunk_path = os.path.join(self.chunks_root, f"chunk_{chunk_idx}.npz")
        chunk = np.load(chunk_path, allow_pickle=True)
        return chunk["data"][idx % self.num_features_per_chunk]

    def remove_npz_suffix(self, name_ext):
        return os.path.splitext(name_ext)[0]

    def format_feature(
        self,
        feature: np.ndarray,
        device: str = None,
        unsqueeze: bool = True,
    ):
        tmp = torch.from_numpy(feature).float()
        if device:
            tmp = tmp.to(device)
        if unsqueeze:
            tmp = tmp.unsqueeze(-1)
        return tmp

    def get_data(self, file_idx, data_buffer):
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer["duration"]
        waveform_sec = duration if self.whole_audio else self.waveform_sec

        # load audio
        idx_from = (
            0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        )
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug

        # load mel
        mel_key = "aug_mel" if aug_flag else "mel"
        mel = data_buffer.get(mel_key)
        features = None

        if mel is None:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    self.features_paths[file_idx],
                    allow_pickle=True,
                )

            mel = self.format_feature(features["mel"], unsqueeze=False)
            mel = mel[start_frame : start_frame + units_frame_len]
        else:
            mel = mel[start_frame : start_frame + units_frame_len]

        # load units
        units = data_buffer.get("units")
        if units is None:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    self.features_paths[file_idx],
                    allow_pickle=True,
                )

            units = self.format_feature(features["units"], unsqueeze=False)

            units = units[start_frame : start_frame + units_frame_len]
            # units = torch.from_numpy(units).float()
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get("f0")
        aug_shift = 0
        if aug_flag:
            # 我也觉得屎，但是这样解释效率高
            if not features:
                features = np.load(
                    self.features_paths[file_idx],
                    allow_pickle=True,
                )
            aug_shift = features["aug_shift"]
        f0_frames = (
            2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        )

        # load volume
        vol_key = "aug_vol" if aug_flag else "volume"
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]

        # load spk_id
        spk_id = data_buffer.get("spk_id")

        aug_shift = self.format_feature(np.array([[aug_shift]]), unsqueeze=False)

        return dict(
            mel=mel,
            f0=f0_frames,
            volume=volume_frames,
            units=units,
            spk_id=spk_id,
            aug_shift=aug_shift,
        )

    def __len__(self):
        return self.num_features
