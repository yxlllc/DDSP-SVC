import os
import argparse
import torch
from torch.optim import lr_scheduler
from optimizer.muon import Muon_AdamW
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.vocoder import Vocoder, Unit2Wav


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


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    # normalize in place (dict-item access: attribute access on DotDict returns a copy)
    # so the config saved to expdir always records the glu_type explicitly
    args['model']['glu_type'] = args.model.glu_type if args.model.glu_type is not None else 'softsign_glu'
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        from reflow.solver import train
        model = Unit2Wav(
                    args.data.sampling_rate,
                    args.data.block_size,
                    args.model.win_length,
                    args.data.encoder_out_channels,
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_aux_layers,
                    args.model.n_aux_chans,
                    args.model.n_layers,
                    args.model.n_chans,
                    glu_type=args.model.glu_type)

    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)

    # fused Triton kernels
    if args.train.use_fused_kernels:
        from reflow.kernels.integration import patch_unit2wav, warmup_fused_backbone, warmup_fused_blocks
        glu_type = args.model.glu_type
        n_patched = patch_unit2wav(model, glu_type=glu_type)
        print(f' > fused kernels: patched {n_patched} LYNXNet2 blocks (glu_type={glu_type})')
        if n_patched > 0 and args.device == 'cuda':
            amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.train.amp_dtype)
            if amp_dtype is None:
                print(' > fused kernels: amp_dtype=fp32 has no autocast dtype; '
                      'fused kernels will fall back to eager at runtime.')
            else:
                max_frames = args.train.batch_size * int(
                    args.data.duration * args.data.sampling_rate / args.data.block_size + 0.5)
                warmup_fused_backbone(model.reflow_model.velocity_fn,
                                      max_frames=max_frames,
                                      autocast_dtype=amp_dtype)
                warmup_fused_blocks(getattr(model, 'ddsp_model', None),
                                    max_frames=max_frames,
                                    autocast_dtype=amp_dtype)
                print(' > fused kernels: all blocks autotune and warmup done.')
        if args.device == 'cuda':
            # Muon's Gram Newton-Schulz runs on Triton as well
            from optimizer.muon import get_params_for_muon, gram_ns_shape_groups
            from optimizer.gram_ns_triton import warmup_gram_ns
            warmup_gram_ns(gram_ns_shape_groups(get_params_for_muon(model)),
                           device=args.device)
            print(' > fused kernels: gram-NS kernel autotune and warmup done.')

    # load parameters
    optimizer = Muon_AdamW(model,
                    muon_args={'weight_decay': args.train.weight_decay,
                               'use_fused_kernels': bool(args.train.use_fused_kernels)},
                    adamw_args={'weight_decay': 0})
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
                        
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid)
    
