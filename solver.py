import os
import time
import numpy as np
import torch

from logger.saver import Saver
from logger import utils

def test(args, model, loss_func, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    test_loss_rss = 0.
    test_loss_uv = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            signal, _, (s_h, s_n) = model(data['units'], data['f0'], data['volume'], data['spk_id'])
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss = loss_func(signal, data['audio'])

            test_loss += loss.item()

            # log
            saver.log_audio({fn+'/gt.wav': data['audio'], fn+'/pred.wav': signal})
            
    # report
    test_loss /= num_batches
    
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, loss_func, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            
            # forward
            signal, _, (s_h, s_n) = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], infer=False)

            # loss
            loss = loss_func(signal, data['audio'])
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item()
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                    
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')

                # run testing set
                test_loss = test(args, model, loss_func, loader_test, saver)
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
    
                saver.log_value({
                    'validation/loss': test_loss
                })
                
                model.train()

                          
