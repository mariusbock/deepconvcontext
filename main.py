# ------------------------------------------------------------------------
# Main script used to commence experiments
# ------------------------------------------------------------------------

import warnings

from munch import Munch
import wandb
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time

import numpy as np

from src.train import run_inertial_network
from src.utils.torch_utils import fix_random_seed
from src.utils.os_utils import Logger, load_config
from src.utils.logging_utils import classification_scores, save_confusion_matrix

def main(args):
    # load config files
    dataset_config = load_config(args.dataset_cfg)
    training_config = load_config(args.training_cfg)
    model_config = load_config(args.model_cfg)

    if args.wandb:
        wandb.init(
        project="",  
        entity="",
        
        config={
            "args": vars(args),
            "dataset_config": Munch.toDict(dataset_config),
            "training_config": Munch.toDict(training_config),
            "model_config": Munch.toDict(model_config),
            },
            reinit=True,
        )
        wandb.config.seed = args.seed

    # Setup logger
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    run_name =  "test"
    if args.wandb:
        run_name = ts.strftime('%Y%m%d_%H%M%S') + '_' + wandb.run.id + '_' + run_name
    else:
        # run id is not available, use a timestamp
        run_name = ts.strftime('%Y%m%d_%H%M%S') + '_' + run_name
    
    log_dir = os.path.join('logs', run_name)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # Save the current config
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(vars(args), stream=fid)
        pprint(Munch.toDict(dataset_config), stream=fid)
        pprint(Munch.toDict(training_config), stream=fid)
        pprint(Munch.toDict(model_config), stream=fid)
    

    rng_generator = fix_random_seed(args.seed, include_cuda=True)    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(dataset_config['dataset']['tiou_thresholds'])))
        
    for i, anno_split in enumerate(dataset_config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if dataset_config['has_null'] == True:
            dataset_config['labels'] = ['null'] + list(file['label_dict'])
        else:
            dataset_config['labels'] = list(file['label_dict'])
        dataset_config['label_dict'] = dict(zip(dataset_config['labels'], list(range(len(dataset_config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        test_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        print('Split {} / {}'.format(i + 1, len(dataset_config['anno_json'])))
        dataset_config['dataset']['json_anno'] = anno_split
        
        t_losses, test_losses, test_mAP, test_preds, test_gt, net = \
            run_inertial_network(train_sbjs, test_sbjs, test_sbjs, dataset_config, training_config, model_config, log_dir, args.ckpt_freq, args.resume, 
                                 rng_generator, i, wandb if args.wandb else None, device=args.gpu, causal=training_config['causal'], causal_type=training_config['causal_type'], causal_l=training_config['causal_l'], benchmark=args.benchmark)
            
        # raw results
        (v_acc, v_prec, v_rec, v_f1) = classification_scores(test_gt, test_preds, len(dataset_config['labels']))

        # print to terminal
        block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(test_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(test_mAP) * 100)
        for tiou, tiou_mAP in zip(dataset_config['dataset']['tiou_thresholds'], test_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
        if args.wandb:
            wandb.log({'loso_results/sbj_{}/accuracy'.format(int(i)): np.mean(v_acc), 
                       'loso_results/sbj_{}/f1'.format(int(i)): np.mean(v_f1), 
                       'loso_results/sbj_{}/mAP'.format(int(i)): np.mean(test_mAP)}
                       )  
        all_v_mAP = np.append(all_v_mAP, test_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, test_gt)
        all_v_pred = np.append(all_v_pred, test_preds)

        # save raw confusion matrix
        save_confusion_matrix(test_gt, test_preds, dataset_config['labels'], os.path.join(log_dir, 'sbj_' + str(i) + '.png'), 'sbj_' + str(i), normalize='true')

    # final raw results across all splits
    (v_acc, v_prec, v_rec, v_f1) = classification_scores(all_v_gt, all_v_pred, len(dataset_config['labels']))
    
    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(dataset_config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    save_confusion_matrix(all_v_gt, all_v_pred, dataset_config['labels'], os.path.join(log_dir, 'all_raw.png'), 'all', normalize='true')

    # submit final values to neptune 
    if args.wandb:
        wandb.log({'final_results/overall_accuracy': np.mean(v_acc), 
                   'final_results/overall_f1': np.mean(v_f1), 
                   'final_results/overall_mAP': np.mean(all_v_mAP)}
                   )
        wandb.finish()

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_cfg', default='configs/dataset/wetlab.yaml', type=str)
    parser.add_argument('--model_cfg', default='configs/model/deepconvcontext.yaml', type=str)
    parser.add_argument('--training_cfg', default='configs/training/causal_sequence_deepconvcontext.yaml', type=str)
    parser.add_argument('--run_id', default='', type=str)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument('--benchmark', action='store_true', default=False)
    args = parser.parse_args()
    main(args)  

