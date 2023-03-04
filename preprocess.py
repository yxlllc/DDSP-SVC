import os
import numpy as np
import librosa
import torch
import pyworld as pw
import parselmouth
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from logger.utils import traverse_dir
import concurrent.futures

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(path, f0_extractor, volume_extractor, units_encoder, sample_rate, hop_size, device = 'cuda'):
    
    path_srcdir  = os.path.join(path, 'audio')
    path_unitsdir  = os.path.join(path, 'units')
    path_f0dir  = os.path.join(path, 'f0')
    path_volumedir  = os.path.join(path, 'volume')
    path_skipdir = os.path.join(path, 'skip')
    
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extension='wav',
        is_pure=True,
        is_sort=True,
        is_ext=True)
        
    # run  
    def process(file):
        ext = file.split('.')[-1]
        binfile = file[:-(len(ext)+1)]+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        
        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)
        
        # extract volume
        volume = volume_extractor.extract(audio)
        
        # units encode
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()
        
        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp = False)
        
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # save npy
            os.makedirs(path_unitsdir, exist_ok=True)
            np.save(path_unitsfile, units)
            os.makedirs(path_f0dir, exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(path_volumedir, exist_ok=True)
            np.save(path_volumefile, volume)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(path_skipdir, exist_ok=True)
            shutil.move(path_srcfile, path_skipdir)
            print('This file has been moved to ' + os.path.join(path_skipdir, file))
    print('Preprocess the audio clips in :', path_srcdir)
    
    # single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
    # multi-process (have bugs)
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        list(tqdm(executor.map(process, filelist), total=len(filelist)))
    '''
                
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    
    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        args.data.sampling_rate, 
                        args.data.block_size, 
                        args.data.f0_min, 
                        args.data.f0_max)
    
    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)
                        
    # initialize units encoder
    units_encoder = Units_Encoder(
                        args.data.encoder, 
                        args.data.encoder_ckpt, 
                        args.data.encoder_sample_rate, 
                        args.data.encoder_hop_size, 
                        device = device)    
    
    # preprocess training set
    preprocess(args.data.train_path, f0_extractor, volume_extractor, units_encoder, sample_rate, hop_size, device = device)
    
    # preprocess validation set
    preprocess(args.data.valid_path, f0_extractor, volume_extractor, units_encoder, sample_rate, hop_size, device = device)
    
