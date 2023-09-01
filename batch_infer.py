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
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from diffusion.vocoder import load_model_vocoder
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
        "-diff",
        "--diff_ckpt",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
    parser.add_argument(
        "-ddsp",
        "--ddsp_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to the DDSP model checkpoint (for shallow diffusion)",
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
        help="pitch extrator type: parselmouth, dio, harvest, crepe, rmvpe (default)",
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
        "-diffid",
        "--diff_spk_id",
        type=str,
        required=False,
        default='auto',
        help="diffusion speaker id (for multi-speaker model) | default: auto",
    )
    parser.add_argument(
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default='auto',
        help="speed up | default: auto",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='auto',
        help="ddim, pndm, dpm-solver or unipc | default: auto",
    )
    parser.add_argument(
        "-kstep",
        "--k_step",
        type=str,
        required=False,
        default=None,
        help="shallow diffusion steps | default: None",
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


def infer(input_path, output_path, cmd, device, model, vocoder, args, ddsp, units_encoder):
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
    if cmd.diff_spk_id == 'auto':
        diff_spk_id = spk_id
    else:
        diff_spk_id = torch.LongTensor(np.array([[int(cmd.diff_spk_id)]])).to(device)
    if spk_mix_dict is not None:
        print('Mix-speaker mode')
    else:
        print('DDSP Speaker ID: '+ str(int(cmd.spk_id)))
        print('Diffusion Speaker ID: '+ str(cmd.diff_spk_id))

    # speed up
    if cmd.speedup == 'auto':
        infer_speedup = args.infer.speedup
    else:
        infer_speedup = int(cmd.speedup)
    if cmd.method == 'auto':
        method = args.infer.method
    else:
        method = cmd.method
    if infer_speedup > 1:
        print('Sampling method: '+ method)
        print('Speed up: '+ str(infer_speedup))
    else:
        print('Sampling method: DDPM')

    input_mel = None
    k_step = None
    if args.model.type == 'DiffusionNew':
        if cmd.k_step is not None:
            k_step = int(cmd.k_step)
            if k_step > args.model.k_step_max:
                k_step = args.model.k_step_max
        else:
            k_step = args.model.k_step_max
        print('Shallow diffusion step: ' + str(k_step))
    else:
        if cmd.k_step is not None:
            k_step = int(cmd.k_step)
            print('Shallow diffusion step: ' + str(k_step))
            if ddsp is None:
                print('DDSP model is not identified!')
                print('Extracting the mel spectrum of the input audio for shallow diffusion...')
                audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)
                input_mel = vocoder.extract(audio_t, sample_rate)
                input_mel = torch.cat((input_mel, input_mel[:, -1:, :]), 1)
        else:
            print('Shallow diffusion step is not identified, gaussian diffusion will be used!')

    with torch.no_grad():
        if ddsp is not None:
            ddsp_f0 = 2 ** (-float(cmd.formant_shift_key) / 12) * f0
            ddsp_output, _, (_, _) = ddsp(units, ddsp_f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
            input_mel = vocoder.extract(ddsp_output, args.data.sampling_rate, keyshift=float(cmd.formant_shift_key))
        mel = model(
            units,
            f0,
            volume,
            spk_id=diff_spk_id,
            spk_mix_dict=spk_mix_dict,
            aug_shift=formant_shift_key,
            vocoder=vocoder,
            gt_spec=input_mel[:, :units.size(1)] if input_mel is not None else None,
            infer=True,
            infer_speedup=infer_speedup,
            method=method,
            k_step=k_step)
        output = vocoder.infer(mel, f0)
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

    # load diffusion model
    model, vocoder, args = load_model_vocoder(cmd.diff_ckpt, device=device)

    ddsp = None
    if args.model.type == 'DiffusionNew':
        if cmd.ddsp_ckpt is not None:
            # load ddsp model
            ddsp, ddsp_args = load_model(cmd.ddsp_ckpt, device=device)
            if not check_args(ddsp_args, args):
                print("Cannot use this DDSP model for shallow diffusion, the built-in DDSP model will be used!")
                ddsp = None
        else:
            print("DDSP model is not identified, the built-in DDSP model will be used!")

    else:
        if cmd.k_step is not None and cmd.ddsp_ckpt is not None:
            # load ddsp model
            ddsp, ddsp_args = load_model(cmd.ddsp_ckpt, device=device)
            if not check_args(ddsp_args, args):
                print("Cannot use this DDSP model for shallow diffusion, gaussian diffusion will be used!")
                ddsp = None

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
        infer(input_path, output_path, cmd, device, model, vocoder, args, ddsp, units_encoder)
