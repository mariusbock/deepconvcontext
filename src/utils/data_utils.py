# ------------------------------------------------------------------------
# Data operation utilities
# ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd


def unwindow_inertial_data(orig, ids, preds, win_size, win_overlap):
    """
    Method to unwindow the predictions of the model.
    
    Args:
        orig: numpy array
            Original data
        ids: numpy array
            Subject IDs
        preds: numpy array
            Predictions
        win_size: int
            Window size
        win_overlap: int
            Window overlap
            
    Returns:
        unseg_preds: numpy array
            Unsegmented predictions
        orig_labels: numpy array
            Original labels
    """
    unseg_preds = []
    unseg_gt = []
    if not ((win_overlap / 100) * win_size).is_integer():
        float_prec = True
    else:
        float_prec = False

    for sbj in np.unique(orig[:, 0]):
        sbj_data = orig[orig[:, 0] == sbj]
        sbj_preds = preds[ids==sbj]
        sbj_unseg_preds = []
        changing_bool = True
        for i, pred in enumerate(sbj_preds):
            if (float_prec == True) and (changing_bool == True):
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)) + 1)))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (win_overlap * 0.01)) + 1)))
                changing_bool = False
            else:
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)))))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * int(win_size * (win_overlap * 0.01))))
                changing_bool = True
        sbj_unseg_preds = np.concatenate((sbj_unseg_preds, np.full(len(sbj_data) - len(sbj_unseg_preds), sbj_preds[-1])))
        unseg_preds = np.concatenate((unseg_preds, sbj_unseg_preds))
        unseg_gt = np.concatenate((unseg_gt, sbj_data[:, -1].flatten().astype(int)))
    assert len(unseg_preds) == len(orig)    
    return unseg_preds, unseg_gt

def convert_samples_to_segments(ids, labels, sampling_rate):
    """
    Method to convert samples to segments.
    
    Args:
        ids: numpy array
            Subject IDs
        labels: numpy array
            Labels
        sampling_rate: int
            Sampling rate
    
    Returns:
        dict: Dictionary with video IDs, labels, start time, end time, and score
    """
    
    f_video_ids, f_labels, f_t_start, f_t_end, f_score = [], np.array([]), np.array([]), np.array([]), np.array([])

    for id in np.unique(ids):
        sbj_labels = labels[(ids == id)]
        curr_start_i = 0
        curr_end_i = 0
        curr_label = sbj_labels[0]
        for i, l in enumerate(sbj_labels):
            if curr_label != l:
                act_start = curr_start_i / sampling_rate
                act_end = curr_end_i / sampling_rate
                act_label = curr_label - 1
                if curr_label != 0:
                    # create annotation
                    f_video_ids.append('sbj_' + str(int(id)))
                    f_labels = np.append(f_labels, act_label)
                    f_t_start = np.append(f_t_start, act_start)
                    f_t_end = np.append(f_t_end, act_end)
                    f_score = np.append(f_score, 1)
                curr_label = l
                curr_start_i = i + 1
                curr_end_i = i + 1    
            else:
                curr_end_i += 1        
    return {
        'video-id': f_video_ids,
        'label': f_labels,
        't-start': f_t_start,
        't-end': f_t_end,
        'score': f_score
    }
