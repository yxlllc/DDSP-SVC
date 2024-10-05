import os
import numpy as np
import random
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from reflow.vocoder import Vocoder
from logger.utils import traverse_dir
import concurrent.futures


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the config file"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set",
    )
    return parser.parse_args(args=args, namespace=namespace)


def preprocess(
    path,
    f0_extractor,
    volume_extractor,
    mel_extractor,
    units_encoder,
    sample_rate,
    hop_size,
    device="cuda",
    use_pitch_aug=False,
    extensions=["wav"],
):
    path_srcdir = os.path.join(path, "audio")
    path_featuresdir = os.path.join(path, "features")
    path_skipdir = os.path.join(path, "skip")

    # list files
    filelist = traverse_dir(
        path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True
    )

    # pitch augmentation dictionary
    pitch_aug_dict = {}

    # run
    def process(file):
        binfile = file
        path_srcfile = os.path.join(path_srcdir, file)
        path_featuresfile = os.path.join(path_featuresdir, binfile)
        if os.path.exists(path_featuresfile + ".npz"):
            try:
                _ = np.load(path_featuresfile + ".npz", allow_pickle=True)
                # print(path_srcfile, "skip because exist")
                return
            except Exception as e:
                print(path_srcfile, "exist but load failed")
                pass

        features = {}

        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        features["duration"] = audio.shape[0] / sample_rate

        # extract volume
        volume = volume_extractor.extract(audio)
        features["volume"] = volume

        # extract mel and volume augmentaion
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to("cpu").numpy()
            features["mel"] = mel

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0

            aug_mel_t = mel_extractor.extract(
                audio_t * (10**log10_vol_shift), sample_rate, keyshift=keyshift
            )
            features["aug_mel"] = aug_mel_t.squeeze().to("cpu").numpy()

            features["aug_vol"] = volume_extractor.extract(
                audio * (10**log10_vol_shift)
            )

        # units encode
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        features["units"] = units_t.squeeze().to("cpu").numpy()

        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp=False)
        features["f0"] = f0

        uv = f0 == 0

        if len(f0[~uv]) <= 0:
            path_skipfile = os.path.join(path_skipdir, file)
            print("\n[Error] F0 extraction failed: " + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print("This file has been moved to " + path_skipfile)
        else:
            features["aug_shift"] = keyshift
            os.makedirs(os.path.dirname(path_featuresfile), exist_ok=True)
            np.savez_compressed(path_featuresfile, **features)

    print("Preprocess the audio clips in :", path_srcdir)

    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    # multi-process (have bugs)
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    """


if __name__ == "__main__":
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    extensions = args.data.extensions

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
        args.data.f0_extractor,
        args.data.sampling_rate,
        args.data.block_size,
        args.data.f0_min,
        args.data.f0_max,
    )

    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)

    # initialize mel extractor
    mel_extractor = None
    use_pitch_aug = False
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    if (
        mel_extractor.vocoder_sample_rate != sample_rate
        or mel_extractor.vocoder_hop_size != hop_size
    ):
        mel_extractor = None
        print("Unmatch vocoder parameters, mel extraction is ignored!")
    elif args.model.use_pitch_aug:
        use_pitch_aug = True

    # initialize units encoder
    if args.data.encoder == "cnhubertsoftfish":
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
        args.data.encoder,
        args.data.encoder_ckpt,
        args.data.encoder_sample_rate,
        args.data.encoder_hop_size,
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device,
    )

    # preprocess training set
    preprocess(
        args.data.train_path,
        f0_extractor,
        volume_extractor,
        mel_extractor,
        units_encoder,
        sample_rate,
        hop_size,
        device=device,
        use_pitch_aug=use_pitch_aug,
        extensions=extensions,
    )

    # preprocess validation set
    preprocess(
        args.data.valid_path,
        f0_extractor,
        volume_extractor,
        mel_extractor,
        units_encoder,
        sample_rate,
        hop_size,
        device=device,
        use_pitch_aug=False,
        extensions=extensions,
    )
