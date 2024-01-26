import torch
import numpy as np
import unittest
import math
from pose_embedding.common import distance_utils

class DistanceTestOperations(unittest.TestCase):

    def assertTensorClose(self, a, b, atol=1e-5, rtol=1e-4):
        """
        Custom assert function to check if two tensors are close
        """
        self.assertTrue(torch.allclose(a, b, atol=atol, rtol=rtol),
                        f"Tensors are not close: \n{a} \n{b}")

    def test_compute_l2_distances(self):
        # Shape = [2, 1, 2, 2]
        lhs_points = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]],
                      [[[10.0, 11.0], [12.0, 13.0]]]])
        rhs_points = torch.tensor([[[[0.0, 1.1], [2.3, 3.4]]],
                      [[[10.4, 11.0], [12.4, 13.3]]]])
        # Shape = [2, 1, 2]
        distances = distance_utils.compute_l2_distances(lhs_points, rhs_points)
        self.assertTensorClose(distances, torch.tensor([[[0.1, 0.5]], [[0.4, 0.5]]]))

    def test_compute_l2_distances_keepdims(self):
        # Shape = [2, 1, 2, 2]
        lhs = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]], [[[10.0, 11.0], [12.0, 13.0]]]])
        rhs = torch.tensor([[[[0.0, 1.1], [2.3, 3.4]]], [[[10.4, 11.0], [12.4, 13.3]]]])
        # Shape = [2, 1, 2]
        distances = distance_utils.compute_l2_distances(
            lhs, rhs, keepdims=True)
        self.assertTensorClose(distances, torch.tensor([[[[0.1], [0.5]]], [[[0.4], [0.5]]]]))

    def test_compute_squared_l2_distances(self):
        # Shape = [2, 1, 2, 2]
        lhs_points = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]],
                      [[[10.0, 11.0], [12.0, 13.0]]]])
        rhs_points = torch.tensor([[[[0.0, 1.1], [2.3, 3.4]]],
                      [[[10.4, 11.0], [12.4, 13.3]]]])
        # Shape = [2, 1, 2]
        distances = distance_utils.compute_l2_distances(
            lhs_points, rhs_points, squared=True)
        self.assertTensorClose(distances, torch.tensor([[[0.01, 0.25]], [[0.16, 0.25]]]))


    def test_compute_sigmoid_matching_probabilities(self):
        inner_distances = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        matching_probabilities = (
            distance_utils.compute_sigmoid_matching_probabilities(
                inner_distances, a=0.01, b=1.0))

        self.assertTensorClose(matching_probabilities,torch.tensor(
                            [[0.70617913, 0.704397395, 0.702607548],
                             [0.700809625, 0.69900366, 0.697189692]])
        )
        
    def test_compute_all_pair_squared_l2_distances(self):
        # Shape = [2, 2, 2].
        lhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]])
        # Shape = [2, 3, 2].
        rhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
        # Shape = [2, 2, 3].
        distances = distance_utils.compute_all_pair_l2_distances(
            lhs, rhs, squared=True)
        self.assertTensorClose(distances, torch.tensor([[[0.0, 8.0, 32.0], [8.0, 0.0, 8.0]],
                                        [[8.0, 0.0, 8.0], [32.0, 8.0, 0.0]]]))

    def test_compute_all_pair_l2_distances(self):
        # Shape = [2, 2, 2].
        lhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]])
        # Shape = [2, 3, 2].
        rhs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]])
        # Shape = [2, 2, 3].
        distances = distance_utils.compute_all_pair_l2_distances(lhs, rhs)
        self.assertTensorClose(
            distances, torch.tensor(
            [[[0.0, 2.828427125, 5.656854249], [2.828427125, 0.0, 2.828427125]],
             [[2.828427125, 0.0, 2.828427125], [5.656854249, 2.828427125, 0.0]]])
        )

    def test_compute_gaussian_likelihoods(self):
        # Shape = [2, 1, 1].
        means = torch.tensor([[[1.0]], [[2.0]]])
        # Shape = [2, 1, 1].
        stddevs = torch.tensor([[[1.0]], [[2.0]]])
        # Shape = [2, 3, 1].
        samples = torch.tensor([[[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]]])
        # Shape = [2, 3, 1].
        likelihoods = distance_utils.compute_gaussian_likelihoods(
            means, stddevs, samples)
        self.assertTensorClose(
            likelihoods, torch.tensor([[[1.0], [0.3173], [0.0455]],
                          [[1.0], [0.3173], [0.0455]]]),
            atol=1e-4)
    
    def test_compute_distance_matrix(self):
        # Shape = [2, 1]
        start_points = torch.tensor([[1], [2]])
        # Shape = [3, 1]
        end_points = torch.tensor([[3], [4], [5]])
        distance_matrix = distance_utils.compute_distance_matrix(
            start_points, end_points, distance_fn=torch.subtract)
        self.assertTensorClose(distance_matrix,
                            torch.tensor([[[-2], [-3], [-4]], [[-1], [-2], [-3]]]))

    def test_compute_gaussian_kl_divergence_unit_univariate(self):
        lhs_means = torch.tensor([[0.0]])
        lhs_stddevs = torch.tensor([[1.0]])
        kl_divergence = distance_utils.compute_gaussian_kl_divergence(
            lhs_means, lhs_stddevs, rhs_means=torch.tensor(0.0), rhs_stddevs=torch.tensor(1.0))

        self.assertTensorClose(kl_divergence, torch.tensor([0.0]))

    def test_compute_gaussian_kl_divergence_unit_multivariate_to_univariate(self):
        lhs_means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        lhs_stddevs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        kl_divergence = distance_utils.compute_gaussian_kl_divergence(
            lhs_means, lhs_stddevs, rhs_means=torch.tensor(0.0), rhs_stddevs=torch.tensor(1.0))

        self.assertTensorClose(kl_divergence, torch.tensor([0.0, 0.0]))

    def test_compute_gaussian_kl_divergence_multivariate_to_multivariate(self):
        lhs_means = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        lhs_stddevs = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        rhs_means = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        rhs_stddevs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        kl_divergence = distance_utils.compute_gaussian_kl_divergence(
            lhs_means, lhs_stddevs, rhs_means=rhs_means, rhs_stddevs=rhs_stddevs)

        self.assertTensorClose(kl_divergence, torch.tensor([31.198712171, 2.429343385]))


# Run the tests
if __name__ == '__main__':
    unittest.main()
