# ------------------------------------------------------------------------
# Methods used for training inertial-based models
# ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from contextlib import contextmanager

from src.utils.data_utils import convert_samples_to_segments, unwindow_inertial_data
from src.utils.torch_utils import BoundedCausalInertialDataset, SequentialInertialDataset, init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from src.utils.os_utils import mkdir_if_missing
from src.utils.map_metric import ANETdetection
from src.utils.scalability_profiler import ScalabilityProfiler

from src.DeepConvLSTM import DeepConvLSTM
from src.DeepConvContext import DeepConvContext


def run_inertial_network(train_sbjs, val_sbjs, test_sbjs, dataset_config, training_config, model_config, ckpt_folder, ckpt_freq, resume, rng_generator, sbj_id, run, device, causal=False, causal_type=None, causal_l=100, benchmark=False):
    """
    Function to run the training and validation of the inertial-based models.
    
    Args:
        train_sbjs: list
            List of subject IDs for training.
        val_sbjs: list
            List of subject IDs for validation.
        test_sbjs: list
            List of subject IDs for testing.
        dataset_config: dict
            Configuration dictionary containing dataset parameters.
        training_config: dict
            Configuration dictionary containing training parameters.
        model_config: dict
            Configuration dictionary containing model parameters.
        ckpt_folder: str
            Folder to save checkpoints.
        ckpt_freq: int
            Frequency of saving checkpoints.
        resume: str or None
            Path to resume training from a checkpoint. If None, training starts from scratch.
        rng_generator: torch.Generator
            Random number generator for reproducibility.
        run: neptune.run or None
            Neptune run object for logging. If None, no logging is done.
            
    Returns:
        t_losses: list
            List of training losses for each epoch.
        v_losses: list
            List of validation losses for each epoch.
        v_mAP: list
            List of mean Average Precision (mAP) scores for each epoch.
        v_preds: np.ndarray
            Array of predictions for the validation dataset.
        v_gt: np.ndarray
            Array of ground truth labels for the validation dataset.
        net: nn.Module
            The trained neural network model.
    """
    split_name = dataset_config['dataset']['json_anno'].split('/')[-1].split('.')[0]
    # load train and val inertial data
    train_data, val_data, test_data = np.empty((0, dataset_config['dataset']['input_dim'] + 2)), np.empty((0, dataset_config['dataset']['input_dim'] + 2)), np.empty((0, dataset_config['dataset']['input_dim'] + 2))
    for t_sbj in train_sbjs:
        t_data = pd.read_csv(os.path.join(dataset_config['dataset']['sens_folder'], t_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": dataset_config['label_dict']}).fillna(0).to_numpy()
        train_data = np.append(train_data, t_data, axis=0)
    for v_sbj in val_sbjs:
        v_data = pd.read_csv(os.path.join(dataset_config['dataset']['sens_folder'], v_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": dataset_config['label_dict']}).fillna(0).to_numpy()
        val_data = np.append(val_data, v_data, axis=0)
    for te_sbj in test_sbjs:
        te_data = pd.read_csv(os.path.join(dataset_config['dataset']['sens_folder'], te_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": dataset_config['label_dict']}).fillna(0).to_numpy()
        test_data = np.append(test_data, te_data, axis=0)

    # define inertial datasets
    if causal and causal_type == 'causal_batch':
        train_dataset = BoundedCausalInertialDataset(train_data, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'], training_config['loader']['train_batch_size'], causal_l, model=model_config['model_name'])
    elif causal_type == 'causal_sequence':
        train_dataset = SequentialInertialDataset(train_data, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'], training_config['loader']['train_batch_size'], model=model_config['model_name'])
    else:
        train_dataset = InertialDataset(train_data, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'], model=dataset_config['dataset_name'])
    val_dataset = InertialDataset(val_data, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'], model=dataset_config['dataset_name'])    
    test_dataset = InertialDataset(test_data, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'], model=dataset_config['dataset_name'])

    # define dataloaders
    if causal_type == 'causal_batch':
        train_loader = DataLoader(train_dataset, training_config['loader']['train_batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    elif causal_type == 'causal_sequence':
        train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, training_config['loader']['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    val_loader = DataLoader(val_dataset, training_config['loader']['test_batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    test_loader = DataLoader(test_dataset, training_config['loader']['test_batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    
    # define network
    if model_config['model_name'] == 'deepconvlstm':
        net = DeepConvLSTM(
            train_dataset.channels, train_dataset.classes, train_dataset.window_size,
            model_config['model']['conv_kernels'],dataset_config['dataset']['conv_kernel_size'], 
            model_config['model']['lstm_units'], model_config['model']['lstm_layers'], model_config['model']['dropout']
            )
        print("Number of learnable parameters for DeepConvLSTM: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        criterion = nn.CrossEntropyLoss()
    elif model_config['model_name'] == 'deepconvcontext':
        net = DeepConvContext(
            training_config['loader']['train_batch_size'], train_dataset.channels, train_dataset.classes, train_dataset.window_size,
            model_config['model']['conv_kernels'], dataset_config['dataset']['conv_kernel_size'], 
            model_config['model']['lstm_units'], model_config['model']['lstm_layers'], model_config['model']['dropout'], model_config['model']['bidirectional'], 
            model_config['model']['type'], model_config['model']['attention_num_heads'], model_config['model']['transformer_depth'], model_config['model']['dim_feedforward']
            )
        print("Number of learnable parameters for DeepConvContext: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        criterion = nn.CrossEntropyLoss()

    # print model summary
    print(net)    
        
    # define criterion and optimizer
    opt = torch.optim.Adam(net.parameters(), lr=training_config['train_cfg']['lr'], weight_decay=training_config['train_cfg']['weight_decay'])

    # use lr schedule if selected
    if training_config['train_cfg']['lr_step'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=training_config['train_cfg']['lr_step'], gamma=training_config['train_cfg']['lr_decay'])
    
    # use weighted loss if selected
    if training_config['train_cfg']['weighted_loss']:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels) + 1, y=train_dataset.labels + 1)
        criterion.weight = torch.tensor(class_weights).float().to(device)

    if resume:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda(device))
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return
    else:
        net = init_weights(net, training_config['train_cfg']['weight_init'])
        start_epoch = 0

    net.to(device)
    for epoch in range(start_epoch, training_config['train_cfg']['epochs']):
        if causal and causal_type == 'causal_batch':
            train_dataset.resample()

        # training
        net, t_losses, _, _ = train_one_epoch(train_loader, net, opt, criterion, device, causal=causal, causal_type=causal_type, causal_l=causal_l, benchmark=benchmark)

        # save ckpt once in a while
        if (((ckpt_freq > 0) and ((epoch + 1) % ckpt_freq == 0))):
            save_states = { 
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
            }

            file_name = 'epoch_{:03d}_{}.pth.tar'.format(epoch + 1, split_name)
            save_checkpoint(save_states, False, file_folder=os.path.join(ckpt_folder, 'ckpts'), file_name=file_name)

        # validation
        v_losses, v_preds, v_gt = validate_one_epoch(val_loader, net, criterion, device, causal=causal)

        if training_config['train_cfg']['lr_step'] > 0:
            scheduler.step()
            
        det_eval = ANETdetection(dataset_config['dataset']['json_anno'], 'validation', tiou_thresholds = dataset_config['dataset']['tiou_thresholds'])
        # undwindow inertial data (sample-wise structure instead of windowed) 
        v_preds, v_gt = unwindow_inertial_data(val_data, val_dataset.ids, v_preds, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'])
        # convert to samples (for mAP calculation)
        v_segments = convert_samples_to_segments(val_data[:, 0], v_preds, dataset_config['dataset']['sampling_rate'])

        if epoch == (start_epoch + training_config['train_cfg']['epochs']) - 1:
            # save raw results (for later postprocessing)
            v_results = pd.DataFrame({
                    'video_id' : v_segments['video-id'],
                    't_start' : v_segments['t-start'].tolist(),
                    't_end': v_segments['t-end'].tolist(),
                    'label': v_segments['label'].tolist(),
                    'score': v_segments['score'].tolist()
            })
            mkdir_if_missing(os.path.join(ckpt_folder, 'unprocessed_results'))
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_preds_' + split_name), v_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_gt_' + split_name), v_gt)
            v_results.to_csv(os.path.join(ckpt_folder, 'unprocessed_results', 'v_seg_' + split_name + '.csv'), index=False)

        # calculate validation metrics
        v_mAP, _ = det_eval.evaluate(v_segments)
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1)
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1)
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1)

        # print results to terminal
        block1 = 'Epoch: [{:03d}/{:03d}]'.format(epoch, training_config['train_cfg']['epochs'])
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(dataset_config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block4  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block4  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block4  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block4  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4]))

        # log to wandb
        if run is not None:
            run.log({f"{sbj_id}/train_loss": np.mean(t_losses), 
                           f"{sbj_id}/val_loss": np.mean(v_losses), 
                           f"{sbj_id}/val_acc": np.nanmean(v_acc), 
                           f"{sbj_id}/val_f1": np.nanmean(v_f1),
                           "step": epoch,
                           "learning_rate": opt.param_groups[-1]['lr']})
    # predict test set
    test_losses, test_preds, test_gt = validate_one_epoch(test_loader, net, criterion, device, causal=causal)

    det_eval = ANETdetection(dataset_config['dataset']['json_anno'], 'validation', tiou_thresholds = dataset_config['dataset']['tiou_thresholds'])
    # undwindow inertial data (sample-wise structure instead of windowed) 
    test_preds, test_gt = unwindow_inertial_data(test_data, test_dataset.ids, test_preds, dataset_config['dataset']['window_size'], dataset_config['dataset']['window_overlap'])
    # convert to samples (for mAP calculation)
    test_segments = convert_samples_to_segments(test_data[:, 0], test_preds, dataset_config['dataset']['sampling_rate'])
    # save raw results (for later postprocessing)
    test_results = pd.DataFrame({
                    'video_id' : test_segments['video-id'],
                    't_start' : test_segments['t-start'].tolist(),
                    't_end': test_segments['t-end'].tolist(),
                    'label': test_segments['label'].tolist(),
                    'score': test_segments['score'].tolist()
            })
    mkdir_if_missing(os.path.join(ckpt_folder, 'unprocessed_results'))
    np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'test_preds_' + split_name), test_preds)
    np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'test_gt_' + split_name), test_gt)
    test_results.to_csv(os.path.join(ckpt_folder, 'unprocessed_results', 'test_seg_' + split_name + '.csv'), index=False)

    # calculate validation metrics
    test_mAP, _ = det_eval.evaluate(test_segments)
    return t_losses, test_losses, test_mAP, test_preds, test_gt, net


"""def train_one_epoch(loader, network, opt, criterion, gpu=None, causal=False, causal_type=None, causal_l=None, benchmark=False):
    """"""
    Trains the network for one epoch.
    
    Args:
        loader: DataLoader
            DataLoader for the training dataset.
        network: nn.Module
            The neural network model to be trained.
        opt: torch.optim.Optimizer
            The optimizer for the model.
        criterion: nn.Module
            The loss function.
        gpu: int, optional
            GPU device ID. If None, CPU is used.
            
    Returns:
        network: nn.Module
            The trained neural network model.
        losses: list
            List of training losses for each batch.
        preds: np.ndarray
            Array of predictions for the training dataset.
        gt: np.ndarray
            Array of ground truth labels for the training dataset.
    """"""
    losses, preds, gt = [], [], []
    network.train()
    hidden = None
    for i, (inputs, targets) in enumerate(loader):
        if causal_type == 'causal_sequence':
            inputs = inputs.squeeze(0)  # remove batch dimension of data loader
            targets = targets.squeeze(0)
            if causal:
                network.set_batch_first(False)

        if gpu is not None:
            inputs, targets = inputs.to(gpu), targets.to(gpu)
        
        if hidden is not None:
            # detach hidden state to prevent backpropagating through entire training history
            if isinstance(hidden, tuple):  # LSTM
                hidden = tuple([h.detach() for h in hidden])
            else:  # GRU or other RNNs
                hidden = hidden.detach()
        
        if causal and causal_type == 'causal_batch':
            if i % causal_l == 0:
                # reset hidden state after causal_l batches
                hidden = None
            if benchmark:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_flops=True, record_shapes=True) as prof:
                    output, hidden = network(inputs, hidden)
            else:
                output, hidden = network(inputs, hidden)
            
        else:
            if type(network).__name__.lower() == 'deepconvcontext':
                if benchmark:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_flops=True, record_shapes=True) as prof:
                        output, _, _ = network(inputs)
                else:
                    output, _, _ = network(inputs)
            else:
                if benchmark:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_flops=True, record_shapes=True) as prof:
                        output, _ = network(inputs)
                else:
                    output, _ = network(inputs)

        batch_loss = criterion(output, targets)
        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        # append train loss to list
        losses.append(batch_loss.item())

        # create predictions and append them to final list
        batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
        batch_gt = targets.cpu().numpy().flatten()
        preds = np.concatenate((preds, batch_preds))
        gt = np.concatenate((gt, batch_gt))

        if benchmark:
            # uncomment to print profiling results
            total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
            total_memory_bytes = torch.cuda.max_memory_allocated()
            print(f"Total FLOPs: {total_flops / 1e6:.2f} MFLOPs")
            print(f"Peak CUDA Memory Usage: {total_memory_bytes / (1024 ** 2):.2f} MB")
    if causal_type == 'causal_sequence' and causal:
        network.set_batch_first(True)
    return network, losses, preds, gt"""


"""
"""

def train_one_epoch(loader, network, opt, criterion, gpu=None, causal=False, causal_type=None, causal_l=None, benchmark=False):
    """
    Trains the network for one epoch.
    
    Args:
        loader: DataLoader
            DataLoader for the training dataset.
        network: nn.Module
            The neural network model to be trained.
        opt: torch.optim.Optimizer
            The optimizer for the model.
        criterion: nn.Module
            The loss function.
        gpu: int, optional
            GPU device ID. If None, CPU is used.
            
    Returns:
        network: nn.Module
            The trained neural network model.
        losses: list
            List of training losses for each batch.
        preds: np.ndarray
            Array of predictions for the training dataset.
        gt: np.ndarray
            Array of ground truth labels for the training dataset.
    """
    losses, preds, gt = [], [], []
    network.train()
    hidden = None

    model_name = type(network).__name__
    profiler = ScalabilityProfiler(model_name) if benchmark else None

    with (profiler or _noop_ctx()):
        for i, (inputs, targets) in enumerate(loader):

            if causal_type == "causal_sequence":
                inputs  = inputs.squeeze(0)
                targets = targets.squeeze(0)
                if causal:
                    network.set_batch_first(False)

            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)

            if hidden is not None:
                if isinstance(hidden, tuple):       # LSTM
                    hidden = tuple(h.detach() for h in hidden)
                else:                               # GRU / other RNN
                    hidden = hidden.detach()

            ctx = profiler.profile_forward(i, inputs) if profiler else _noop_ctx()
            with ctx:
                if causal and causal_type == "causal_batch":
                    if i % causal_l == 0:
                        hidden = None
                    output, hidden = network(inputs, hidden)

                elif model_name.lower() == "deepconvcontext":
                    output, _, _ = network(inputs)

                else:
                    output, _ = network(inputs)

            batch_loss = criterion(output, targets)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            losses.append(batch_loss.item())

            batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
            batch_gt    = targets.cpu().numpy().flatten()
            preds       = np.concatenate((preds, batch_preds))
            gt          = np.concatenate((gt, batch_gt))

    if causal_type == "causal_sequence" and causal:
        network.set_batch_first(True)

    return network, losses, preds, gt


@contextmanager
def _noop_ctx():
    yield


def validate_one_epoch(loader, network, criterion, gpu=None, causal=False):
    """
    Validates the network for one epoch.
    
    Args:
        loader: DataLoader
            DataLoader for the validation dataset.
        network: nn.Module
            The neural network model to be validated.
        criterion: nn.Module
            The loss function.
        gpu: int, optional
            GPU device ID. If None, CPU is used.
    
    Returns:
        losses: list
            List of validation losses for each batch.
        preds: np.ndarray
            Array of predictions for the validation dataset.
        gt: np.ndarray
            Array of ground truth labels for the validation dataset.
    """
    losses, preds, gt = [], [], []
    hidden = None
    window_hidden = None
    network.eval()
    if causal:
        network.set_batch_first(False)
    with torch.no_grad():
        # iterate over validation dataset
        for i, (inputs, targets) in enumerate(loader):            
            # send x and y to GPU
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)

            if hidden is not None:
                # detach hidden state to prevent backpropagating through entire training history
                if isinstance(hidden, tuple):  # LSTM
                    hidden = tuple([h.detach() for h in hidden])
                else:  # GRU or other RNNs
                    hidden = hidden.detach()
            
            if window_hidden is not None:
                if isinstance(window_hidden, tuple):  # LSTM
                    window_hidden = tuple([h.detach() for h in window_hidden])
                else:  # GRU or other RNNs
                    window_hidden = window_hidden.detach()

            if type(network).__name__.lower() == 'deepconvcontext':
                output, hidden, window_hidden = network(inputs, hidden, window_hidden)
                # for bidirectional LSTM, zero out backward hidden state before next batch
                if hidden is not None and isinstance(hidden, tuple) and network.bidirectional:
                    num_layers = hidden[0].shape[0] // 2
                    h_n, c_n = hidden
                    # keep forward slice, zero backward slice
                    h_n = torch.cat([h_n[:num_layers], torch.zeros_like(h_n[num_layers:])], dim=0)
                    c_n = torch.cat([c_n[:num_layers], torch.zeros_like(c_n[num_layers:])], dim=0)
                    hidden = (h_n, c_n)
            else:
                output, hidden = network(inputs, hidden)
            batch_loss = criterion(output, targets.long())
            losses.append(batch_loss.item())

            # create predictions and append them to final list
            batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()
            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))
                
    if causal:
        network.set_batch_first(True)
    return losses, preds, gt