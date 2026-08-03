# ------------------------------------------------------------------------
# Postprocessing script to calculate metrics and confusion matrices.
# ------------------------------------------------------------------------

import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.utils.map_metric import ANETdetection
from src.utils.data_utils import convert_samples_to_segments
from src.utils.logging_utils import classification_scores
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# postprocessing parameters
types = ['main', 'ablation/attention', 'ablation/contextlength', 'ablation/twolayered']
datasets = ['hangtime', 'opportunity', 'wetlab', 'sbhar', 'rwhar', 'wear'] 
seeds = [1, 2, 3]

for type in types:
    for dataset in datasets:
        if type == 'main':
            models = ['causalbatch', 'deepconvcontext', 'deepconvlstm', 'shallowdeepconvlstm']
        elif type == 'ablation/attention':
            models = ['attention', 'causalattention', 'causaltransformer', 'transformer', 'bilstm']
        elif type == 'ablation/contextlength':
            models = ['25', '50', '200']
        elif type == 'ablation/twolayered':
            models = ['causalbatch', 'deepconvcontext', 'deepconvlstm', 'shallowdeepconvlstm']
        for model in models:
            path_to_preds = ['experiments/{}/{}/{}'.format(type, model, dataset)]
            for path in path_to_preds:
                if not os.path.exists(path):
                    print("Path does not exist: {}".format(path))
                    continue
                else:
                    print("\nProcessing path: {}".format(path))
                if dataset == 'opportunity':
                    num_classes = 18
                    sampling_rate = 30
                    input_dim = 113
                    classes = ['null', 'open_door_1', 'open_door_2', 'close_door_1', 'close_door_2', 'open_fridge', 'close_fridge', 'open_dishwasher', 'close_dishwasher', 'open_drawer_1', 'close_drawer_1', 'open_drawer_2', 'close_drawer_2', 'open_drawer_3', 'close_drawer_3', 'clean_table', 'drink_from_cup', 'toggle_switch']
                    json_files = [
                    'data/opportunity/annotations/loso_sbj_0.json',
                    'data/opportunity/annotations/loso_sbj_1.json',
                    'data/opportunity/annotations/loso_sbj_2.json',
                    'data/opportunity/annotations/loso_sbj_3.json',
                    ]
                elif dataset == 'wetlab':
                    num_classes = 9
                    sampling_rate = 50
                    input_dim = 3
                    classes = ['null', 'cutting', 'inverting', 'peeling', 'pestling', 'pipetting', 'pouring', 'stirring', 'transfer']
                    json_files = [
                    'data/wetlab/annotations/loso_sbj_0.json',
                    'data/wetlab/annotations/loso_sbj_1.json',
                    'data/wetlab/annotations/loso_sbj_2.json',
                    'data/wetlab/annotations/loso_sbj_3.json',
                    'data/wetlab/annotations/loso_sbj_4.json',
                    'data/wetlab/annotations/loso_sbj_5.json',
                    'data/wetlab/annotations/loso_sbj_6.json',
                    'data/wetlab/annotations/loso_sbj_7.json',
                    'data/wetlab/annotations/loso_sbj_8.json',
                    'data/wetlab/annotations/loso_sbj_9.json',
                    'data/wetlab/annotations/loso_sbj_10.json',
                    'data/wetlab/annotations/loso_sbj_11.json',
                    'data/wetlab/annotations/loso_sbj_12.json',
                    'data/wetlab/annotations/loso_sbj_13.json',
                    'data/wetlab/annotations/loso_sbj_14.json',
                    'data/wetlab/annotations/loso_sbj_15.json',
                    'data/wetlab/annotations/loso_sbj_16.json',
                    'data/wetlab/annotations/loso_sbj_17.json',
                    'data/wetlab/annotations/loso_sbj_18.json',
                    'data/wetlab/annotations/loso_sbj_19.json',
                    'data/wetlab/annotations/loso_sbj_20.json',
                    'data/wetlab/annotations/loso_sbj_21.json'
                    ]
                elif dataset == 'sbhar':
                    num_classes = 13
                    sampling_rate = 50
                    input_dim = 3
                    classes = ['null', 'walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'lying', 'stand-to-sit', 'sit-to-stand', 'sit-to-lie', 'lie-to-sit', 'stand-to-lie', 'lie-to-stand']
                    json_files = [
                    'data/sbhar/annotations/loso_sbj_0.json',
                    'data/sbhar/annotations/loso_sbj_1.json',
                    'data/sbhar/annotations/loso_sbj_2.json',
                    'data/sbhar/annotations/loso_sbj_3.json',
                    'data/sbhar/annotations/loso_sbj_4.json',
                    'data/sbhar/annotations/loso_sbj_5.json',
                    'data/sbhar/annotations/loso_sbj_6.json',
                    'data/sbhar/annotations/loso_sbj_7.json',
                    'data/sbhar/annotations/loso_sbj_8.json',
                    'data/sbhar/annotations/loso_sbj_9.json',
                    'data/sbhar/annotations/loso_sbj_10.json',
                    'data/sbhar/annotations/loso_sbj_11.json',
                    'data/sbhar/annotations/loso_sbj_12.json',
                    'data/sbhar/annotations/loso_sbj_13.json',
                    'data/sbhar/annotations/loso_sbj_14.json',
                    'data/sbhar/annotations/loso_sbj_15.json',
                    'data/sbhar/annotations/loso_sbj_16.json',
                    'data/sbhar/annotations/loso_sbj_17.json',
                    'data/sbhar/annotations/loso_sbj_18.json',
                    'data/sbhar/annotations/loso_sbj_19.json',
                    'data/sbhar/annotations/loso_sbj_20.json',
                    'data/sbhar/annotations/loso_sbj_21.json',
                    'data/sbhar/annotations/loso_sbj_22.json',
                    'data/sbhar/annotations/loso_sbj_23.json',
                    'data/sbhar/annotations/loso_sbj_24.json',
                    'data/sbhar/annotations/loso_sbj_25.json',
                    'data/sbhar/annotations/loso_sbj_26.json',
                    'data/sbhar/annotations/loso_sbj_27.json',
                    'data/sbhar/annotations/loso_sbj_28.json',
                    'data/sbhar/annotations/loso_sbj_29.json'
                    ]
                elif dataset == 'rwhar':
                    num_classes = 8
                    sampling_rate = 50
                    input_dim = 21
                    classes = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
                    json_files = [
                    'data/rwhar/annotations/loso_sbj_0.json',
                    'data/rwhar/annotations/loso_sbj_1.json',
                    'data/rwhar/annotations/loso_sbj_2.json',
                    'data/rwhar/annotations/loso_sbj_3.json',
                    'data/rwhar/annotations/loso_sbj_4.json',
                    'data/rwhar/annotations/loso_sbj_5.json',
                    'data/rwhar/annotations/loso_sbj_6.json',
                    'data/rwhar/annotations/loso_sbj_7.json',
                    'data/rwhar/annotations/loso_sbj_8.json',
                    'data/rwhar/annotations/loso_sbj_9.json',
                    'data/rwhar/annotations/loso_sbj_10.json',
                    'data/rwhar/annotations/loso_sbj_11.json',
                    'data/rwhar/annotations/loso_sbj_12.json',
                    'data/rwhar/annotations/loso_sbj_13.json',
                    'data/rwhar/annotations/loso_sbj_14.json'
                    ]
                elif dataset == 'wear':
                    num_classes = 19
                    sampling_rate = 50
                    input_dim = 12
                    classes = ['null', 'jogging', 'jogging (rotating arms)', 'jogging (skipping)', 'jogging (sidesteps)', 'jogging (butt-kicks)', 'stretching (triceps)', 'stretching (lunging)', 'stretching (shoulders)', 'stretching (hamstrings)', 'stretching (lumbar rotation)', 'push-ups', 'push-ups (complex)', 'sit-ups', 'sit-ups (complex)', 'burpees', 'lunges', 'lunges (complex)', 'bench-dips']
                    json_files = [
                    'data/wear/annotations/loso_sbj_0.json',
                    'data/wear/annotations/loso_sbj_1.json',
                    'data/wear/annotations/loso_sbj_2.json',
                    'data/wear/annotations/loso_sbj_3.json',
                    'data/wear/annotations/loso_sbj_4.json',
                    'data/wear/annotations/loso_sbj_5.json',
                    'data/wear/annotations/loso_sbj_6.json',
                    'data/wear/annotations/loso_sbj_7.json',
                    'data/wear/annotations/loso_sbj_8.json',
                    'data/wear/annotations/loso_sbj_9.json',
                    'data/wear/annotations/loso_sbj_10.json',
                    'data/wear/annotations/loso_sbj_11.json',
                    'data/wear/annotations/loso_sbj_12.json',
                    'data/wear/annotations/loso_sbj_13.json',
                    'data/wear/annotations/loso_sbj_14.json',
                    'data/wear/annotations/loso_sbj_15.json',
                    'data/wear/annotations/loso_sbj_16.json',
                    'data/wear/annotations/loso_sbj_17.json',
                    'data/wear/annotations/loso_sbj_18.json',
                    'data/wear/annotations/loso_sbj_19.json',
                    'data/wear/annotations/loso_sbj_20.json',
                    'data/wear/annotations/loso_sbj_21.json',
                    ]
                elif dataset == 'hangtime':
                    num_classes = 6
                    sampling_rate = 50
                    input_dim = 3
                    classes = ['null', 'dribbling', 'shot', 'pass', 'rebound', 'layup']
                    json_files = [
                    'data/hangtime/annotations/loso_sbj_0.json',
                    'data/hangtime/annotations/loso_sbj_1.json',
                    'data/hangtime/annotations/loso_sbj_2.json',
                    'data/hangtime/annotations/loso_sbj_3.json',
                    'data/hangtime/annotations/loso_sbj_4.json',
                    'data/hangtime/annotations/loso_sbj_5.json',
                    'data/hangtime/annotations/loso_sbj_6.json',
                    'data/hangtime/annotations/loso_sbj_7.json',
                    'data/hangtime/annotations/loso_sbj_8.json',
                    'data/hangtime/annotations/loso_sbj_9.json',
                    'data/hangtime/annotations/loso_sbj_10.json',
                    'data/hangtime/annotations/loso_sbj_11.json',
                    'data/hangtime/annotations/loso_sbj_12.json',
                    'data/hangtime/annotations/loso_sbj_13.json',
                    'data/hangtime/annotations/loso_sbj_14.json',
                    'data/hangtime/annotations/loso_sbj_15.json',
                    'data/hangtime/annotations/loso_sbj_16.json',
                    'data/hangtime/annotations/loso_sbj_17.json',
                    'data/hangtime/annotations/loso_sbj_18.json',
                    'data/hangtime/annotations/loso_sbj_19.json',
                    'data/hangtime/annotations/loso_sbj_20.json',
                    'data/hangtime/annotations/loso_sbj_21.json',
                    'data/hangtime/annotations/loso_sbj_22.json',
                    'data/hangtime/annotations/loso_sbj_23.json',
                    ]

                all_mAP = np.zeros((5, len(json_files)))
                all_prec_seeds = np.zeros((num_classes, len(seeds)))
                all_recall_seeds = np.zeros((num_classes, len(seeds)))
                all_f1_seeds = np.zeros((num_classes, len(seeds)))

                for s_pos, seed in enumerate(seeds):
                    all_preds = np.array([])
                    all_gt = np.array([])
                    for i, j in enumerate(json_files):
                        with open(j) as fi:
                            file = json.load(fi)
                            anno_file = file['database']
                            if dataset == 'rwhar':
                                labels = list(file['label_dict'])
                            else:
                                labels = ['null'] + list(file['label_dict'])
                            label_dict = dict(zip(labels, list(range(len(labels)))))
                            val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

                        v_data = np.empty((0, input_dim + 2))

                        for sbj in val_sbjs:
                            data = pd.read_csv(os.path.join('data/{}/raw/inertial'.format(dataset), sbj + '.csv'), index_col=False, low_memory=False).replace({"label": label_dict}).fillna(0).to_numpy()
                            v_data = np.append(v_data, data, axis=0)

                        v_preds = np.array([])
                        if 'loso' in j:
                            v_orig_preds = np.load(os.path.join(path, str(seed), 'unprocessed_results/v_preds_loso_sbj_{}.npy'.format(int(i))))
                        else:
                            v_orig_preds = np.load(os.path.join(path, str(seed), 'unprocessed_results/v_preds_split_{}.npy'.format(int(i) + 1)))

                        for sbj in val_sbjs:
                            sbj_pred = v_orig_preds[v_data[:, 0] == int(sbj.split("_")[-1])]
                            v_preds = np.append(v_preds, sbj_pred) 

                        seg_data = convert_samples_to_segments(v_data[:, 0], v_preds, sampling_rate)
                        det_eval = ANETdetection(j, 'validation', tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
                        v_mAP, _ = det_eval.evaluate(seg_data)

                        all_mAP[:, i] += v_mAP
                        all_preds = np.append(all_preds, v_preds)
                        all_gt = np.append(all_gt, v_data[:, -1])

                    _, v_prec, v_rec, v_f1 = classification_scores(all_gt.astype(int), all_preds.astype(int), num_classes)
                    all_prec_seeds[:, s_pos] = v_prec
                    all_recall_seeds[:, s_pos] = v_rec
                    all_f1_seeds[:, s_pos] = v_f1

                    if seed == 1:
                        comb_conf = confusion_matrix(all_gt.astype(int), all_preds.astype(int), normalize='true', labels=range(num_classes))
                        comb_conf = np.around(comb_conf, 2)
                        comb_conf[comb_conf == 0] = np.nan

                        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                        sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Purples, cbar=False, annot_kws={'fontsize': 16,}, linecolor='black', vmin=0, vmax=1)
                        ax.set_xlabel('Predicted', fontsize=20)
                        ax.set_ylabel('True', fontsize=20)
                        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=16)
                        ax.set_yticklabels(classes, rotation=0, ha='right', fontsize=16)
                        ax.set_xlim(0, comb_conf.shape[1])
                        ax.set_ylim(comb_conf.shape[0], 0)
                        ax.tick_params(which="both", length=0)
                        pred_name = path.split('/')[-3] + "_" + path.split('/')[-2] + "_" + path.split('/')[-1]
                        ax.set_title(pred_name, fontsize=20)
                        os.path.exists('confusion_matrices') or os.makedirs('confusion_matrices')
                        _.savefig(os.path.join('confusion_matrices', pred_name + ".svg"), format='svg')
                        plt.close()

                print("Individual mAP:")
                print("mAP@0.3: {:.2f} (±{:.2f})".format(np.mean(all_mAP[0, :]) / len(seeds) * 100, np.std(all_mAP[0, :] / len(seeds)) * 100))
                print("mAP@0.4: {:.2f} (±{:.2f})".format(np.mean(all_mAP[1, :]) / len(seeds) * 100, np.std(all_mAP[1, :] / len(seeds)) * 100))
                print("mAP@0.5: {:.2f} (±{:.2f})".format(np.mean(all_mAP[2, :]) / len(seeds) * 100, np.std(all_mAP[2, :] / len(seeds)) * 100))
                print("mAP@0.6: {:.2f} (±{:.2f})".format(np.mean(all_mAP[3, :]) / len(seeds) * 100, np.std(all_mAP[3, :] / len(seeds)) * 100))
                print("mAP@0.7: {:.2f} (±{:.2f})".format(np.mean(all_mAP[4, :]) / len(seeds) * 100, np.std(all_mAP[4, :] / len(seeds)) * 100))

                print("Average mAP:")
                print("{:.2f} (±{:.2f})".format(np.mean(np.mean(all_mAP / len(seeds), axis=0)) * 100, np.std(np.mean(all_mAP / len(seeds), axis=0)) * 100))

                print("Average precision:")
                per_seed_prec = np.mean(all_prec_seeds, axis=0)
                print("{:.2f}% (±{:.2f})".format(np.mean(per_seed_prec) * 100, np.std(per_seed_prec) * 100))

                print("Average recall:")
                per_seed_rec = np.mean(all_recall_seeds, axis=0)
                print("{:.2f}% (±{:.2f})".format(np.mean(per_seed_rec) * 100, np.std(per_seed_rec) * 100))

                print("Average f1:")
                per_seed_f1 = np.mean(all_f1_seeds, axis=0)
                print("{:.2f}% (±{:.2f})".format(np.mean(per_seed_f1) * 100, np.std(per_seed_f1) * 100))