# ------------------------------------------------------------------------
# Torch utilities
# ------------------------------------------------------------------------

import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn


class SequentialInertialDataset(Dataset):
    """
    Sequence-based batch loader. Sequences are randomly sampled from the dataset, but each sequence consists of batch_size consecutive windows from the same subject.
    Use with DataLoader(batch_size=1, shuffle=True) or iterate directly.

    Args:
        data: numpy array
            Raw input data.
        window_size: int
            Number of time steps per window.
        window_overlap: int
            Overlap between successive windows.
        batch_size: int
            Number of consecutive windows per item (= context sequence length).
        stride: int | None
            Step between sequence starts within a subject. Defaults to
            `batch_size` (non-overlapping). Set to 1 for fully sliding extraction.
        model: str
            Model identifier forwarded from the outer training script.
    """

    def __init__(self, data, window_size, window_overlap, batch_size,
                 stride=None, model='deepconvlstm'):
        self.ids, self.features, self.labels = apply_sliding_window(
            data, window_size, window_overlap)

        self.batch_size  = batch_size
        self.stride      = stride if stride is not None else batch_size
        self.model       = model
        self.window_size = window_size
        self.classes     = len(np.unique(self.labels))
        self.channels    = self.features.shape[2] - 1
        self.sequences = []

        for subject_id in np.unique(self.ids):
            subject_idx = np.where(self.ids == subject_id)[0]

            gap = np.diff(subject_idx)
            split_points = np.where(gap > 1)[0] + 1
            consecutive_runs = np.split(subject_idx, split_points)

            for run in consecutive_runs:
                if len(run) < batch_size:
                    continue
                for s in range(0, len(run) - batch_size + 1, self.stride):
                    self.sequences.append(run[s : s + batch_size])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        idx = self.sequences[index]
        return self.features[idx, :, 1:].astype(np.float32), self.labels[idx].astype(np.uint8)


class InertialDataset(Dataset):
    """
    Inertial dataset for time series classification.
    
    Args:
        data: numpy array
            Input data
        window_size: int
            Size of the sliding window
        window_overlap: int
            Overlap of the sliding window
        model: str
            Model type (default is 'deepconvlstm')
    """
    def __init__(self, data, window_size, window_overlap, model='deepconvlstm'):
        self.ids, self.features, self.labels = apply_sliding_window(data, window_size, window_overlap)
        self.classes = len(np.unique(self.labels))
        self.channels = self.features.shape[2] - 1
        self.window_size = window_size
        self.model = model

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if 'attendanddiscriminate' in self.model:
            data = torch.FloatTensor(self.features[index, :, 1:])
            target = torch.LongTensor([int(self.labels[index])])
            return data, target
        else:
            return self.features[index, :, 1:].astype(np.float32), self.labels[index].astype(np.uint8)
        

class BoundedCausalInertialDataset(Dataset):
    """
    CausalBatch dataset.

    Args:
        data: numpy array
            Input data, subject ID in column 0, label in last column.
        window_size: int
            Size of the sliding window (w in the paper).
        window_overlap: int
            Overlap between consecutive windows.
        batch_size: int
            Number of parallel chunks per CausalBatch (B in the paper).
        l: int
            Chunk length in windows. Temporal horizon τ = l × window_size.
            Each subject's windows are divided into floor(N / l) chunks.
        model: str
            Model type tag.
    """

    def __init__(self, data, window_size, window_overlap, batch_size, l, model='deepconvlstm'):
        ids, features, labels = apply_sliding_window(data, window_size, window_overlap)

        self.all_ids      = ids
        self.all_features = features
        self.all_labels   = labels
        self.batch_size   = batch_size
        self.l            = l
        self.labels       = labels
        self.window_size  = window_size
        self.model        = model
        self.classes      = len(np.unique(labels))
        self.channels     = features.shape[2] - 1

        self.chunks = []  
        for subject in np.unique(ids):
            subject_idx = np.where(ids == subject)[0]
            n_chunks    = len(subject_idx) // l
            if n_chunks == 0:
                print(f"Warning: subject {subject} has fewer than l={l} windows "
                      f"and will be skipped entirely.")
                continue
            n_dropped = len(subject_idx) - n_chunks * l
            if n_dropped > 0:
                print(f"Subject {subject}: dropping last {n_dropped} window(s) "
                      f"to fit into chunks of l={l}.")
            for c in range(n_chunks):
                # Store the global index of the first window in this chunk
                self.chunks.append((subject, subject_idx[c * l]))

        self.epoch_indices    = None
        self.n_causal_batches = 0
        self.resample()

    def resample(self):
        """
        Shuffle chunks and rebuild the flat epoch index array. Call at the start of every training epoch.
        """
        chunks = self.chunks.copy()
        np.random.shuffle(chunks)

        # Trim to a multiple of batch_size so every CausalBatch is fully populated
        n_causal_batches = len(chunks) // self.batch_size
        chunks           = chunks[:n_causal_batches * self.batch_size]

        epoch_indices = []
        for cb in range(n_causal_batches):
            # batch_size chunks assigned to this CausalBatch
            batch_chunks = chunks[cb * self.batch_size:(cb + 1) * self.batch_size]

            # Emit l consecutive minibatches by stepping through each chunk
            for i in range(self.l):
                for subject, chunk_start in batch_chunks:
                    epoch_indices.append(chunk_start + i)

        self.epoch_indices    = np.array(epoch_indices)
        self.n_causal_batches = n_causal_batches

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, index):
        actual_idx = self.epoch_indices[index]
        x          = self.all_features[actual_idx, :, 1:].astype(np.float32)
        y          = int(self.all_labels[actual_idx])
        return x, y


def init_weights(network, weight_init):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    Args:
        network: torch.nn.Module
            The network to initialize.
        weight_init: str
            The weight initialization method. Options are 'normal', 'orthogonal', 'xavier_uniform', 'xavier_normal',
            'kaiming_uniform', 'kaiming_normal'.
            
    Returns:
        network: torch.nn.Module
            The initialized network.
    """
    for m in network.modules():
        # conv initialisation
        if isinstance(m, nn.Conv2d):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                m.bias.data.fill_(0.0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    if weight_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif weight_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif weight_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif weight_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif weight_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif weight_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
        elif isinstance(m, nn.LayerNorm):
            # Typically, the scale (weight) is initialized to 1 and the bias to 0.
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        # Transformer-related: MultiheadAttention and TransformerEncoderLayer
        elif isinstance(m, nn.MultiheadAttention):
            for attr in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                param = m
                for part in attr.split('.'):
                    param = getattr(param, part)
                if 'weight' in attr:
                    if weight_init == 'normal':
                        nn.init.normal_(param)
                    elif weight_init == 'orthogonal':
                        nn.init.orthogonal_(param)
                    elif weight_init == 'xavier_uniform':
                        nn.init.xavier_uniform_(param)
                    elif weight_init == 'xavier_normal':
                        nn.init.xavier_normal_(param)
                    elif weight_init == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(param)
                    elif weight_init == 'kaiming_normal':
                        nn.init.kaiming_normal_(param)
                else:
                    nn.init.constant_(param, 0)

        elif isinstance(m, nn.TransformerEncoderLayer):
            # Applies to self_attn, linear1, linear2
            init_weights(m.self_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)

        elif isinstance(m, nn.TransformerDecoderLayer):
            init_weights(m.self_attn, weight_init)
            init_weights(m.multihead_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)
            init_weights(m.norm3, weight_init)
    return network


def fix_random_seed(seed, include_cuda=True):
    """
    Fix random seed for reproducibility.
    
    Args:
        seed: int
            Random seed to fix.
        include_cuda: bool
            Whether to include CUDA in the random seed fixing.

    Returns:
        rng_generator: torch.Generator
            Random number generator.
    """
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    """
    Save model checkpoint to file
    
    Args:
        state: dict
            State dictionary containing model and optimizer state.
        is_best: bool
            Whether this is the best model so far.
        file_folder: str
            Folder to save the checkpoint.
        file_name: str
            Name of the checkpoint file.    
    """
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.
    
    Args:
        data: numpy array
            Input data
        win_len: int
            Window length in samples
        overlap_ratio: int, optional
            Overlap ratio in percent (default is None)
            
    Returns:
        windows: numpy array
            Sliding windows
        indices: numpy array
            Indices of the sliding windows
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)


def apply_sliding_window(data, window_size, window_overlap, no_context_windows=0):
    """
    Apply a sliding window to the data.
    
    Args:
        data: numpy array
            Input data
        window_size: int
            Window size
        window_overlap: int
            Window overlap
        no_context_windows: int, optional
            Number of context windows to apply (default is 0)
            
    Returns:
        output_sbj: numpy array
            Subject IDs
        output_x: numpy array
            Sliding windows
        output_y: numpy array
            Labels
    """
    output_x = None
    output_y = None
    output_sbj = []
    if no_context_windows > 0:
        look_up_windows = None
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)
        if no_context_windows > 0:
            tmp_lookup = tmp_x
            tmp_x = tmp_x[no_context_windows - 1:]
            tmp_y = tmp_y[no_context_windows - 1:]
        
        if output_x is None:
            if no_context_windows > 0:
                look_up_windows = tmp_lookup
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            if no_context_windows > 0:
                look_up_windows = np.concatenate((look_up_windows, tmp_lookup), axis=0)
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    if no_context_windows > 0:
        return output_sbj, output_x, look_up_windows, np.array(output_y).flatten()
    else:
        return output_sbj, output_x, np.array(output_y).flatten()