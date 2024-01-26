import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
import h5py
import glob
import time
from pose_embedding.common import keypoint_utils, keypoint_profiles, constants, data_utils

def preprocess_keypoints_3d(keypoints_3d, keypoint_profile_3d, normalize_keypoints_3d=True):
    side_outputs = {}
    if normalize_keypoints_3d:
        (keypoints_3d, side_outputs[constants.KEY_OFFSET_POINTS_3D],
        side_outputs[constants.KEY_SCALE_DISTANCES_3D]) = (
            keypoint_profile_3d.normalize(keypoints_3d))

    side_outputs[constants.KEY_PREPROCESSED_KEYPOINTS_3D] = keypoints_3d
    return keypoints_3d, side_outputs      



class PosePairsDataset(Dataset):
    def __init__(self, hdf5_folder, keypoint_profile_2d_name='LEGACY_2DCOCO13',
                            keypoint_profile_3d_name="EXTRACTED_3DH36M17",
                            azimuth_range=(-math.pi, math.pi),
                            elevation_range=(-math.pi / 6.0, math.pi / 6.0),
                            roll_range=(-math.pi / 6.0, math.pi / 6.0),
                            normalized_camera_depth_range=(),
                            projection_mix_batch_assignment=None,
                            sequential_inputs=False, seed=None, is_train=True,
                            keypoint_dropout_rate=[0.4, 0.2]):
        super(PosePairsDataset, self).__init__()

        self.hdf5_filepaths = glob.glob(hdf5_folder)
        self.file_handles = [h5py.File(filepath, 'r') for filepath in self.hdf5_filepaths]

        # Since all the HDF5 files have nearly the same number of samples, we select the minimal number
        h5len_per_file = [len(f_handle) for f_handle in self.file_handles]
        self.samples_per_file = min(h5len_per_file)
        min_idx = h5len_per_file.index(self.samples_per_file)
        self.chunk_keys = list(self.file_handles[min_idx].keys())

        self.keypoint_profile_3d = keypoint_profiles.create_keypoint_profile(keypoint_profile_3d_name)
        self.keypoint_profile_2d = keypoint_profiles.create_keypoint_profile(keypoint_profile_2d_name)
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.roll_range = roll_range
        if not normalized_camera_depth_range:   # default ()
            self.normalized_camera_depth_range = (1.0 / self.keypoint_profile_2d.scale_unit,
                                             1.0 / self.keypoint_profile_2d.scale_unit)        
        self.projection_mix_batch_assignment = projection_mix_batch_assignment
        self.sequential_inputs = sequential_inputs
        self.seed = seed
        self.is_train = is_train
        self.keypoint_dropout_rate = keypoint_dropout_rate
        
    def __len__(self):
        return len(self.hdf5_filepaths) * self.samples_per_file

    def preprocess_keypoints_3d(self, keypoints_3d, keypoint_profile_3d, normalize_keypoints_3d=True):
        side_outputs = {}
        if normalize_keypoints_3d:
            keypoints_3d, side_outputs[constants.KEY_OFFSET_POINTS_3D],side_outputs[constants.KEY_SCALE_DISTANCES_3D] = keypoint_profile_3d.normalize(keypoints_3d)

        side_outputs[constants.KEY_PREPROCESSED_KEYPOINTS_3D] = keypoints_3d
        return keypoints_3d, side_outputs      

    def preprocess_keypoints_2d(self, keypoints_2d, keypoint_masks_2d, keypoints_3d):
                    
        projected_keypoints_2d, _ = (
            keypoint_utils.randomly_project_and_select_keypoints(
                keypoints_3d,
                keypoint_profile_3d=self.keypoint_profile_3d,
                output_keypoint_names=(
                    self.keypoint_profile_2d.compatible_keypoint_name_dict[
                        self.keypoint_profile_3d.name]),
                azimuth_range=self.azimuth_range,
                elevation_range=self.elevation_range,
                roll_range=self.roll_range,
                normalized_camera_depth_range=self.normalized_camera_depth_range,
                sequential_inputs=self.sequential_inputs,
                seed=self.seed))
        projected_keypoint_masks_2d = torch.ones(
            projected_keypoints_2d.shape[:-1], dtype=torch.float32)        
        keypoints_2d, keypoint_masks_2d = data_utils.mix_batch(
            [keypoints_2d, keypoint_masks_2d],
            [projected_keypoints_2d, projected_keypoint_masks_2d],
            axis=1,
            assignment=self.projection_mix_batch_assignment,
            seed=self.seed)
        return keypoints_2d, keypoint_masks_2d

    def __getitem__(self, index):
        def squeeze_dict(input):
            for k, v in input.items():
                if v.shape[0] == 1:
                    v.squeeze_(0)
        # Determine which HDF5 file the index belongs to
        file_idx, sample_idx = self._get_file_and_sample_index(index)
        hf = self.file_handles[file_idx]
        data = hf[self.chunk_keys[sample_idx]]
        # # logic to process the data...
        kpt3d = torch.tensor(np.array(data['joint3d']))
        kpt2d_cam1 = torch.tensor(np.array(data['joint2d_camera_1']))
        kpt2d_cam2 = torch.tensor(np.array(data['joint2d_camera_2']))
        kpt2d = torch.stack([kpt2d_cam1, kpt2d_cam2], axis=0)
        kpt2d, _ = keypoint_utils.select_keypoints_by_name(kpt2d, input_keypoint_names=self.keypoint_profile_3d.keypoint_names,
                                                        output_keypoint_names=self.keypoint_profile_2d.compatible_keypoint_name_dict[self.keypoint_profile_3d.name])
        kpt2d = kpt2d[:,:,:2].unsqueeze(0)
        keypoint_masks_2d = torch.ones(kpt2d.shape[:-1], dtype=torch.float32)

        kpt3d = kpt3d.unsqueeze(0).repeat(2,1,1).unsqueeze(0)

        side_outputs = {}

        kpt3d_normal, side_output = self.preprocess_keypoints_3d(kpt3d, self.keypoint_profile_3d)
        side_outputs.update(side_output)

        if self.is_train:
            keypoints_2d, keypoint_masks_2d = self.preprocess_keypoints_2d(kpt2d, keypoint_masks_2d, kpt3d_normal)
        else:
            keypoints_2d = kpt2d

        keypoints_2d, offset_points, scale_distances = (
            self.keypoint_profile_2d.normalize(keypoints_2d, keypoint_masks_2d))        
        
        # occlution augmentation by manipulating keypoint_masks_2d
        if self.keypoint_dropout_rate[0] > 0:
            keypoint_masks_2d = keypoint_utils.apply_stratified_instance_keypoint_dropout(
                keypoint_masks=keypoint_masks_2d,
                probability_to_apply=self.keypoint_dropout_rate[0],
                probability_to_drop=self.keypoint_dropout_rate[1]
            )
        # Create a mask tensor where keypoint_masks_2d equals 1.0
        # keypoint_masks_2d[:,1,:] = 1    # keypoint dropout only applies to anchors
        mask = (keypoint_masks_2d == 1.0).unsqueeze(-1)

        # Tile the mask using the tile_last_dims function
        mask = data_utils.tile_last_dims(mask, last_dim_multiples=[keypoints_2d.shape[-1]])

        # Apply the mask, setting values to zero where the mask is False
        keypoints_2d = torch.where(mask, keypoints_2d, torch.zeros_like(keypoints_2d))

        side_outputs.update({
            constants.KEY_KEYPOINTS_3D: kpt3d_normal, #.squeeze(0),
            constants.KEY_OFFSET_POINTS_2D: offset_points, #.squeeze(0),
            constants.KEY_SCALE_DISTANCES_2D: scale_distances, #.squeeze(0),
            constants.KEY_PREPROCESSED_KEYPOINTS_2D: keypoints_2d, #.squeeze(0),
            constants.KEY_PREPROCESSED_KEYPOINT_MASKS_2D: keypoint_masks_2d #.squeeze(0)
        })
        features = torch.cat([keypoints_2d, keypoint_masks_2d.unsqueeze(-1)], dim=-1)
        features = data_utils.flatten_last_dims(features, num_last_dims=2)
        side_outputs.update(dict(model_inputs=features)) #.squeeze(0)))
        squeeze_dict(side_outputs)
        return side_outputs

    def _get_file_and_sample_index(self, global_index):
        file_idx = global_index // self.samples_per_file
        sample_idx = global_index % self.samples_per_file
        return file_idx, sample_idx

    def close(self):
        for hf in self.file_handles:
            hf.close()

    def __del__(self):
        self.close()

