import torch
import numpy as np
import unittest
import math
from pose_embedding.common import data_utils

class DataTestOperations(unittest.TestCase):

    def assertTensorClose(self, a, b, atol=1e-5, rtol=1e-4):
        """
        Custom assert function to check if two tensors are close
        """
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol),
                        f"Tensors are not close: \n{a} \n{b}")

    def test_flatten_last_dims(self):
        # Shape = [2, 3, 4].
        x = torch.tensor([[[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]],
                         [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]]])
        flattened_x = data_utils.flatten_last_dims(x, num_last_dims=2)
        self.assertTensorClose(flattened_x,
                            torch.tensor([[1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24],
                             [31, 32, 33, 34, 41, 42, 43, 44, 51, 52, 53, 54]]))

    def test_reshape_by_last_dims(self):
        # Shape = [2, 4, 1].
        x = torch.tensor([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])
        # Shape = [2, 2, 2]
        reshaped_x = data_utils.reshape_by_last_dims(x, last_dim_shape=[2, 2])
        self.assertTensorClose(reshaped_x, torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_recursively_expand_dims(self):
        # Shape = [2, 3].
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        # Shape = [2, 1, 3, 1]
        expanded_x = data_utils.recursively_expand_dims(x, axes=[-1, 1])
        self.assertTensorClose(expanded_x, torch.tensor([[[[1], [2], [3]]], [[[4], [5], [6]]]]))

    def test_flatten_first_dims(self):
        # Shape = [1, 2, 3, 4, 1].
        x = torch.tensor([[[[[1], [2], [3], [4]], [[11], [12], [13], [14]],
                           [[21], [22], [23], [24]]],
                          [[[31], [32], [33], [34]], [[41], [42], [43], [44]],
                           [[51], [52], [53], [54]]]]])
        flattened_x = data_utils.flatten_first_dims(x, num_last_dims_to_keep=2)
        self.assertTensorClose(flattened_x,
                            torch.tensor([[[1], [2], [3], [4]], [[11], [12], [13], [14]],
                             [[21], [22], [23], [24]], [
                                 [31], [32], [33], [34]],
                             [[41], [42], [43], [44]], [[51], [52], [53], [54]]]))
        
    def test_unflatten_first_dim(self):
        # Shape = [6, 2].
        x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        unflattened_x = data_utils.unflatten_first_dim(
            x, shape_to_unflatten=torch.tensor([2, 3]))
        self.assertTensorClose(unflattened_x,
                            torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))

    def test_tile_last_dims(self):
        # Shape = [2, 1, 2, 1].
        x = torch.tensor([[[[1], [2]]], [[[3], [4]]]])
        tiled_x = data_utils.tile_last_dims(x, last_dim_multiples=[2, 2])
        self.assertTensorClose(tiled_x, torch.tensor([[[[1, 1], [2, 2], [1, 1], [2, 2]]],
                                      [[[3, 3], [4, 4], [3, 3], [4, 4]]]]))                
        
    def test_sample_gaussians(self):
        means = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stddevs = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        samples = data_utils.sample_gaussians(
            means, stddevs, num_samples=10000, seed=1)
        self.assertTensorClose(
            torch.mean(samples,dim=1), means, atol=0.1, rtol=0.1
            )
        self.assertTensorClose(
            torch.std(samples,dim=1), stddevs, atol=0.1, rtol=0.1
            )        
                

    def test_mix_batch_singletons(self):
        # Shape = [8, 1, 2].
        lhs_batch = torch.tensor([
            [[[1.0, 1.01]]],
            [[[1.1, 1.11]]],
            [[[3.0, 3.01]]],
            [[[3.1, 3.11]]],
            [[[5.0, 5.01]]],
            [[[5.1, 5.11]]],
            [[[7.0, 7.01]]],
            [[[7.1, 7.11]]],
        ])
        rhs_batch = torch.tensor([
            [[[11.0, 11.01]]],
            [[[11.1, 11.11]]],
            [[[13.0, 13.01]]],
            [[[13.1, 13.11]]],
            [[[15.0, 15.01]]],
            [[[15.1, 15.11]]],
            [[[17.0, 17.01]]],
            [[[17.1, 17.11]]],
        ])
        # Shape = [8, 1].
        assignment = torch.tensor([[True], [True], [False], [False], [True], [True],
                                  [False], [False]])
        mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch],
                                           axis=1,
                                           assignment=assignment)[0]
        self.assertTensorClose(
            mixed_batch,
            torch.tensor([
                [[[1.0, 1.01]]],
                [[[1.1, 1.11]]],
                [[[13.0, 13.01]]],
                [[[13.1, 13.11]]],
                [[[5.0, 5.01]]],
                [[[5.1, 5.11]]],
                [[[17.0, 17.01]]],
                [[[17.1, 17.11]]],
            ],
                dtype=torch.float32))

    def test_mix_batch_pairs(self):
        # Shape = [8, 2, 2].
        lhs_batch = torch.tensor([
            [[[1.0, 1.01]], [[2.0, 2.01]]],
            [[[1.1, 1.11]], [[2.1, 2.11]]],
            [[[3.0, 3.01]], [[4.0, 4.01]]],
            [[[3.1, 3.11]], [[4.1, 4.11]]],
            [[[5.0, 5.01]], [[6.0, 6.01]]],
            [[[5.1, 5.11]], [[6.1, 6.11]]],
            [[[7.0, 7.01]], [[8.0, 8.01]]],
            [[[7.1, 7.11]], [[8.1, 8.11]]],
        ])
        rhs_batch = torch.tensor([
            [[[11.0, 11.01]], [[12.0, 12.01]]],
            [[[11.1, 11.11]], [[12.1, 12.11]]],
            [[[13.0, 13.01]], [[14.0, 14.01]]],
            [[[13.1, 13.11]], [[14.1, 14.11]]],
            [[[15.0, 15.01]], [[16.0, 16.01]]],
            [[[15.1, 15.11]], [[16.1, 16.11]]],
            [[[17.0, 17.01]], [[18.0, 18.01]]],
            [[[17.1, 17.11]], [[18.1, 18.11]]],
        ])
        # Shape = [8, 2].
        assignment = torch.tensor([[True, True], [True, True], [False, False],
                                  [False, False], [True, False], [True, False],
                                  [False, True], [False, True]])
        mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch],
                                           axis=1,
                                           assignment=assignment)[0]
        self.assertTensorClose(
            mixed_batch,
            torch.tensor([
                [[[1.0, 1.01]], [[2.0, 2.01]]],
                [[[1.1, 1.11]], [[2.1, 2.11]]],
                [[[13.0, 13.01]], [[14.0, 14.01]]],
                [[[13.1, 13.11]], [[14.1, 14.11]]],
                [[[5.0, 5.01]], [[16.0, 16.01]]],
                [[[5.1, 5.11]], [[16.1, 16.11]]],
                [[[17.0, 17.01]], [[8.0, 8.01]]],
                [[[17.1, 17.11]], [[8.1, 8.11]]],
            ],
                dtype=torch.float32))

    def test_mix_batch_pair_lists(self):
        lhs_batches, rhs_batches = [None, None], [None, None]

        # Shape = [4, 3, 2, 1].
        lhs_batches[0] = torch.tensor([
            [[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
            [[[2.0], [2.1]], [[2.2], [2.3]], [[2.4], [2.5]]],
            [[[3.0], [3.1]], [[3.2], [3.3]], [[3.4], [3.5]]],
            [[[4.0], [4.1]], [[4.2], [4.3]], [[4.4], [4.5]]],
        ])
        rhs_batches[0] = torch.tensor([
            [[[11.0], [11.1]], [[11.2], [11.3]], [[11.4], [11.5]]],
            [[[12.0], [12.1]], [[12.2], [12.3]], [[12.4], [12.5]]],
            [[[13.0], [13.1]], [[13.2], [13.3]], [[13.4], [13.5]]],
            [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]],
        ])

        # Shape = [4, 3, 2, 2, 1].
        lhs_batches[1] = torch.tensor([[[[[1.0], [10.0]], [[1.1], [10.1]]],
                                       [[[1.2], [10.2]], [[1.3], [10.3]]],
                                       [[[1.4], [10.4]], [[1.5], [10.5]]]],
                                      [[[[2.0], [20.0]], [[2.1], [20.1]]],
                                       [[[2.2], [20.2]], [[2.3], [20.3]]],
                                       [[[2.4], [20.4]], [[2.5], [20.5]]]],
                                      [[[[3.0], [30.0]], [[3.1], [30.1]]],
                                       [[[3.2], [30.2]], [[3.3], [30.3]]],
                                       [[[3.4], [30.4]], [[3.5], [30.5]]]],
                                      [[[[4.0], [40.0]], [[4.1], [40.1]]],
                                       [[[4.2], [40.2]], [[4.3], [40.3]]],
                                       [[[4.4], [40.4]], [[4.5], [40.5]]]]])
        rhs_batches[1] = torch.tensor([[[[[11.0], [110.0]], [[11.1], [110.1]]],
                                       [[[11.2], [110.2]], [[11.3], [110.3]]],
                                       [[[11.4], [110.4]], [[11.5], [110.5]]]],
                                      [[[[12.0], [120.0]], [[12.1], [120.1]]],
                                       [[[12.2], [120.2]], [[12.3], [120.3]]],
                                       [[[12.4], [120.4]], [[12.5], [120.5]]]],
                                      [[[[13.0], [130.0]], [[13.1], [130.1]]],
                                       [[[13.2], [130.2]], [[13.3], [130.3]]],
                                       [[[13.4], [130.4]], [[13.5], [130.5]]]],
                                      [[[[14.0], [140.0]], [[14.1], [140.1]]],
                                       [[[14.2], [140.2]], [[14.3], [140.3]]],
                                       [[[14.4], [140.4]], [[14.5], [140.5]]]]])

        # Shape = [4, 1, 2].
        assignment = torch.tensor([[[True, True]], [[True, False]], [[False, True]],
                                  [[False, False]]])

        mixed_batches = data_utils.mix_batch(
            lhs_batches, rhs_batches, axis=2, assignment=assignment)
        self.assertEqual(len(mixed_batches), 2)
        self.assertTensorClose(
            mixed_batches[0],
            # Shape = [4, 3, 2, 1].
            torch.tensor([[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                      [[[2.0], [12.1]], [[2.2], [12.3]], [[2.4], [12.5]]],
                      [[[13.0], [3.1]], [[13.2], [3.3]], [[13.4], [3.5]]],
                      [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]],
                     dtype=torch.float32))
        self.assertTensorClose(
            mixed_batches[1],
            # Shape = [4, 3, 2, 2, 1].
            torch.tensor([[[[[1.0], [10.0]], [[1.1], [10.1]]],
                       [[[1.2], [10.2]], [[1.3], [10.3]]],
                       [[[1.4], [10.4]], [[1.5], [10.5]]]],
                      [[[[2.0], [20.0]], [[12.1], [120.1]]],
                       [[[2.2], [20.2]], [[12.3], [120.3]]],
                       [[[2.4], [20.4]], [[12.5], [120.5]]]],
                      [[[[13.0], [130.0]], [[3.1], [30.1]]],
                       [[[13.2], [130.2]], [[3.3], [30.3]]],
                       [[[13.4], [130.4]], [[3.5], [30.5]]]],
                      [[[[14.0], [140.0]], [[14.1], [140.1]]],
                       [[[14.2], [140.2]], [[14.3], [140.3]]],
                       [[[14.4], [140.4]], [[14.5], [140.5]]]]],
                     dtype=torch.float32))

# Run the tests
if __name__ == '__main__':
    unittest.main()        