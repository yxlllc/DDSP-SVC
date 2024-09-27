import os
import pathlib

import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
import hashlib
from ast import literal_eval
from slicer import Slicer
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from reflow.vocoder import load_model_vocoder
from tqdm import tqdm


def traverse_dir(
        root_dir,
        extensions: list,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_rel=False,
        is_sort=False,
        is_ext=True
):
    """
    Iterate the files matching the given condition in the given directory and its subdirectories.
    :param root_dir: the root directory
    :param extensions: a list of required file extensions (without ".")
    :param amount: limit the number of files
    :param str_include: require the relative path to include this string
    :param str_exclude: require the relative path not to include this string
    :param is_rel: whether to return the relative path instead of full path
    :param is_sort: whether to sort the final results
    :param is_ext: whether to reserve extensions in the filenames
    """
    root_dir = pathlib.Path(root_dir)
    file_list = []
    cnt = 0
    for file in root_dir.rglob("*"):
        if not any(file.suffix == f".{ext}" for ext in extensions):
            continue
        # path
        pure_path = file.relative_to(root_dir)
        mix_path = pure_path if is_rel else file
        # check string
        if (str_include is not None) and (str_include not in pure_path.as_posix()):
            continue
        if (str_exclude is not None) and (str_exclude in pure_path.as_posix()):
            continue
        # amount
        if (amount is not None) and (cnt == amount):
            if is_sort:
                file_list.sort()
            return file_list

        if not is_ext:
            mix_path = mix_path.with_suffix('')
        file_list.append(mix_path)
        cnt += 1

    if is_sort:
        file_list.sort()
    return file_list


def check_args(ddsp_args, diff_args):
    if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
        print("Unmatch data.sampling_rate!")
        return False
    if ddsp_args.data.block_size != diff_args.data.block_size:
        print("Unmatch data.block_size!")
        return False
    if ddsp_args.data.encoder != diff_args.data.encoder:
        print("Unmatch data.encoder!")
        return False
    return True


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_ckpt",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio directory",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-step",
        "--infer_step",
        type=str,
        required=False,
        default='auto',
        help="sample steps | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="euler or rk4 | default: auto",
    )
    parser.add_argument(
        "-ts",
        "--t_start",
        type=str,
        required=False,
        default=0.0,
        help="t_start | default: auto",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        required=False,
        nargs="*",
        default=["wav", "flac"],
        help="list of using file extensions, e.g.) -f wav flac ... | default: wav flac"
    )
    return parser.parse_args(args=args, namespace=namespace)


def infer(input_path, output_path, cmd, device, model, vocoder, args, units_encoder):
    # load input
    audio, sample_rate = librosa.load(input_path, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate

    # get MD5 hash from wav file
    md5_hash = ""
    with open(input_path, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
        print("MD5: " + md5_hash)

    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path,
                                   f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")

    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
        # f0 cache load
        print('Loading pitch curves for input audio from cache directory...')
        f0 = np.load(cache_file_path, allow_pickle=False)
    else:
        # extract f0
        print('Pitch extractor type: ' + cmd.pitch_extractor)
        pitch_extractor = F0_Extractor(
            cmd.pitch_extractor,
            sample_rate,
            hop_size,
            float(cmd.f0_min),
            float(cmd.f0_max))
        print('Extracting the pitch curve of the input audio...')
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=device)

        # f0 cache save
        os.makedirs(cache_dir_path, exist_ok=True)
        np.save(cache_file_path, f0, allow_pickle=False)

    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

    # key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)

    # formant change
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)

    # extract volume
    print('Extracting the volume envelope of the input audio...')
    volume_extractor = Volume_Extractor(hop_size)
    volume = volume_extractor.extract(audio)
    mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

    input = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    units = units_encoder.encode(input, sample_rate, hop_size)

    # speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('Speaker ID: '+ str(int(cmd.spk_id)))
    
    # sampling method    
    if cmd.method == 'auto':
        method = args.infer.method
    else:
        method = cmd.method
        
    # infer step
    if cmd.infer_step == 'auto':
        infer_step = args.infer.infer_step
    else:
        infer_step = int(cmd.infer_step)
    
    # t_start
    if cmd.t_start == 'auto':
        if args.model.t_start is not None:
            t_start = float(args.model.t_start)
        else:
            t_start = 0.0
    else:
        t_start = float(cmd.t_start)
        if args.model.t_start is not None and t_start < args.model.t_start:
            t_start = args.model.t_start
            
    if infer_step > 0:
        print('Sampling method: '+ method)
        print('infer step: '+ str(infer_step))
        print('t_start: '+ str(t_start))
    elif infer_step < 0:
        print('infer step cannot be negative!')
        exit(0)

    with torch.no_grad():
        output = model(
            units,
            f0,
            volume,
            spk_id=spk_id,
            spk_mix_dict=spk_mix_dict,
            aug_shift=formant_shift_key,
            vocoder=vocoder,
            infer=True,
            return_wav=True,
            infer_step=infer_step, 
            method=method,
            t_start=t_start)
        output *= mask
        output = output.squeeze().cpu().numpy()
        sf.write(output_path, output, args.data.sampling_rate)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    # device
    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extensions = cmd.extensions

    # load reflow model
    model, vocoder, args = load_model_vocoder(cmd.model_ckpt, device=device)

    # load units encoder
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
        args.data.encoder,
        args.data.encoder_ckpt,
        args.data.encoder_sample_rate,
        args.data.encoder_hop_size,
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device)
    wav_paths = traverse_dir(
        cmd.input,
        extensions=extensions,
        is_rel=True,
        is_sort=True,
        is_ext=True
    )
    for rel_path in tqdm(wav_paths):
        input_path = pathlib.Path(cmd.input) / rel_path
        output_path = (pathlib.Path(cmd.output) / rel_path).with_suffix('.wav')
        print('_______________________________')
        print('Input: ' + str(input_path))
        infer(input_path, output_path, cmd, device, model, vocoder, args, units_encoder)
