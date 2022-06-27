import os
import sys
import numpy as np
import torch
import argparse
import tqdm
import pickle
import random

_TOTAL_ACTIONS = 4

# Mapping from 1-base of NTU to vibe 49 joints
# hip, thorax, 
_MAJOR_JOINTS = [39, 41, 37, 43, 34, 35, 36, 33, 32, 31, 28, 29, 30, 27, 26, 25, 40]
#                1,   2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21

_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
_MIN_STD = 1e-4
_SPINE_ROOT = 0 # after only taking major joints (ie index in _MAJOR_JOINTS)

def collate_fn(batch):
  """Collate function for data loaders."""
  e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
  d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))
  d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
  action_id = torch.from_numpy(np.stack([e['action_id'] for e in batch]))
  action = [e['action_str'] for e in batch]

  batch_ = {
      'encoder_inputs': e_inp,
      'decoder_inputs': d_inp,
      'decoder_outputs': d_out,
      'action_str': action,
      'action_ids': action_id
  }

  return batch_

class GaitJointsDataset(torch.utils.data.Dataset):
  def __init__(self, params=None, mode='train', fold=1):
    super(GaitJointsDataset, self).__init__()
    self._params = params
    self._mode = mode
    thisname = self.__class__.__name__
    self._monitor_action = 'normal'

    self._action_str = ['normal', 'slight', 'moderate', 'severe']
    self.data_dir = self._params['data_path']
    self.fold = fold

    self.load_data()

  def load_data(self):
    train_data = pickle.load(open(self.data_dir+"EPG_train_" + str(self.fold) + ".pkl", "rb"))
    test_data = pickle.load(open(self.data_dir+"EPG_test_" + str(self.fold) + ".pkl", "rb"))

    if self._mode == 'train':
      X_1, Y = self.data_generator(train_data, mode='train', fold_number=self.fold) 
    else:
      X_1, Y = self.data_generator(test_data) 
    self.X_1 = X_1
    self.Y = Y
    self._action_str = ['none', 'mild', 'moderate', 'severe']

    self._pose_dim = 3 * _NMAJOR_JOINTS
    self._data_dim = self._pose_dim


  def data_generator(self, T, mode='test', fold_number=1):
    X_1 = []
    Y = []

    # bootstrap_number = 3
    # num_samples = 39

    total_num_clips = 0
    for i in range(len(T['pose'])): 
      total_num_clips += 1
      p = np.copy(T['pose'][i])
      # print(np.shape(p))
      y_label_index = T['label'][i]
      label = y_label_index
      X_1.append(p)
      Y.append(label)
    # can't stack X_1 because not all have equal frames
    Y = np.stack(Y)

    # For using a subset of the dataset (few-shot)
    # if mode == 'train':
    #   sampling_dir = 'PATH/TO/BOOTSTRAP_SAMPLING_DIR'
    #   all_clip_video_names = pickle.load(open(sampling_dir + "all_clip_video_names.pkl", "rb"))
    #   clip_video_names = all_clip_video_names[fold_number - 1]

    #   all_bootstrap_samples = pickle.load(open(sampling_dir + f'{num_samples}_samples/bootstrap_{bootstrap_number}_samples.pkl', 'rb'))
    #   bootstrap_samples = all_bootstrap_samples[fold_number - 1]

    #   mask_list = [1 if video_name in bootstrap_samples else 0 for video_name in clip_video_names]
    #   train_indices = [train_idx for train_idx, mask_value in enumerate(mask_list) if mask_value == 1]

    #   X_1 = [X_1[train_idx] for train_idx in train_indices]
    #   Y = Y[train_indices]


    return X_1, Y

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, idx):
    return self._get_item_train(idx)

  def _get_item_train(self, idx):
    """Get item for the training mode."""
    x = self.X_1[idx]
    y = self.Y[idx]

    # adjust for mapping/subset of joints from vibe to ntu
    x = x[:,_MAJOR_JOINTS,:]

    action_id = y
    source_seq_len = self._params['source_seq_len']
    target_seq_len = self._params['target_seq_len']
    input_size = 3 * _NMAJOR_JOINTS # not sure if this is right
    pose_size = 3 * _NMAJOR_JOINTS # note sure if thiis is right
    total_frames = source_seq_len + target_seq_len
    src_seq_len = source_seq_len - 1

    encoder_inputs = np.zeros((src_seq_len, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((target_seq_len, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)

    # total_framesxn_joints*joint_dim

    N = np.shape(x)[0]

    x = x.reshape(N, -1)
    start_frame = np.random.randint(0, N - total_frames + 1)

    # original code did not change start frame between epochs
    start_frame = random.randint(0, N - total_frames) # high inclusive

    data_sel = x[start_frame:(start_frame + total_frames), :]

    encoder_inputs[:, 0:input_size] = data_sel[0:src_seq_len,:]
    decoder_inputs[:, 0:input_size] = \
        data_sel[src_seq_len:src_seq_len+target_seq_len, :]
    decoder_outputs[:, 0:pose_size] = data_sel[source_seq_len:, 0:pose_size]

    if self._params['pad_decoder_inputs']:
      query = decoder_inputs[0:1, :]
      decoder_inputs = np.repeat(query, target_seq_len, axis=0)
    

    return {
        'encoder_inputs': encoder_inputs, 
        'decoder_inputs': decoder_inputs, 
        'decoder_outputs': decoder_outputs,
        'action_id': action_id,
        'action_str': self._action_str[action_id],
    }

def dataset_factory(params, fold):
  """Defines the datasets that will be used for training and validation."""
  params['num_activities'] = _TOTAL_ACTIONS
  params['virtual_dataset_size'] = params['steps_per_epoch']*params['batch_size']
  params['n_joints'] = _NMAJOR_JOINTS

  eval_mode = 'test' if 'test_phase' in params.keys() else 'eval'
  if eval_mode == 'test':
    train_dataset_fn = None
  else:
    train_dataset = GaitJointsDataset(params, mode='train', fold=fold)
    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    eval_dataset = GaitJointsDataset(
      params, 
      mode=eval_mode,
      fold=fold,
    )
    eval_dataset_fn = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    ) 


    return train_dataset_fn, eval_dataset_fn