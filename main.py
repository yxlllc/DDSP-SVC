import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import parselmouth
from slicer import Slicer
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from enhancer import Enhancer

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="path to the model file",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones)",
    )
    parser.add_argument(
        "-e",
        "--enhance",
        type=str,
        required=False,
        default='false',
        help="true or false",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='parselmouth',
        help="pitch extrator type: parselmouth, dio, harvest, crepe",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz)",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz)",
    )
    return parser.parse_args(args=args, namespace=namespace)

    
def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            result.append((
                    start_frame, 
                    audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


if __name__ == '__main__':
    #device = 'cpu' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load ddsp model
    model, args = load_model(cmd.model_path, device=device)
    
    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
    
    # extract f0 
    pitch_extractor = F0_Extractor(
                        cmd.pitch_extractor, 
                        sample_rate, 
                        hop_size, 
                        float(cmd.f0_min), 
                        float(cmd.f0_max))
    f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
   
    # key change
    f0 = f0 * 2** ( float(cmd.key) / 12)
    
    # extract volume 
    volume_extractor = Volume_Extractor(hop_size)
    volume = volume_extractor.extract(audio)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)
    
    # load units encoder
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size, 
                        device = device)
                        
    # load enhancer
    if cmd.enhance == 'true':
        enhancer = Enhancer(args.enhancer.type, args.enhancer.ckpt, device=device)
       
    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    with torch.no_grad():
        for segment in segments:
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
            seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
           
            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            
            seg_output, _, (s_h, s_n) = model(seg_units, seg_f0, seg_volume)

            if cmd.enhance == 'true':
                seg_output, output_sample_rate = enhancer.enhance(seg_output, args.data.sampling_rate, seg_f0, args.data.block_size)
            else:
                output_sample_rate = args.data.sampling_rate
            
            seg_output = seg_output.squeeze().cpu().numpy()
            
            silent_length = round(start_frame * args.data.block_size * output_sample_rate / args.data.sampling_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)
        sf.write(cmd.output, result, output_sample_rate)
    