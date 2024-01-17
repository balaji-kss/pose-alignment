import torch
import numpy as np
import unittest
import math
import functools
from pose_embedding.common import constants, loss_utils, models


class ModelTestOperations(unittest.TestCase):
    def test_fully_connected_shapes(self):
        # Create a dummy input tensor
        input_features = torch.ones([4, 2, 8])

        # Initialize the FullyConnected module and compute the output
        model = models.FullyConnected(input_dim=input_features.shape[-1], num_hidden_nodes=32)
        output_features = model(input_features)

        # Verify that the output shape matches the expected shape
        self.assertEqual(output_features.shape, torch.Size([4, 2, 32]))

    def test_fully_connected_shapes_2(self):
        # Create a dummy input tensor
        input_features = torch.ones([4, 8])

        # Initialize the FullyConnected module and compute the output
        model = models.FullyConnected(input_dim=input_features.shape[-1], num_hidden_nodes=32)
        output_features = model(input_features)

        # Verify that the output shape matches the expected shape
        self.assertEqual(output_features.shape, torch.Size([4, 32]))

    def test_fully_connected_block_shapes(self):
        # Create a dummy input tensor
        input_features = torch.ones([4, 8])
        feat_dim = 32

        # Initialize the FullyConnected module and compute the output

        model = torch.nn.Sequential(
                                    models.FullyConnected(input_dim=input_features.shape[-1], num_hidden_nodes=feat_dim),
                                    models.FullyConnectedBlock(input_dim=feat_dim, num_hidden_nodes=feat_dim, num_fcs_per_block=3)
        )
        output_features = model(input_features)

        # Verify that the output shape matches the expected shape
        self.assertEqual(output_features.shape, torch.Size([4, 32]))

    def test_multi_head_logits_shapes(self):
        N, T = 5, 10
        input_features = torch.ones([N, T, 32])
        output_sizes = {'a': 8, 'b': [4, 3]}
        model = models.MultiHeadLogits(input_features.shape[-1], output_sizes=output_sizes, num_hidden_nodes=5)
        output_features = model(input_features)
        self.assertCountEqual(output_features.keys(), ['a', 'b'])
        self.assertEqual(output_features['a'].shape, torch.Size([N, T, 8]))
        self.assertEqual(output_features['b'].shape, torch.Size([N, T, 4, 3]))

    def test_get_simple_model(self):
        input_features = torch.Tensor([[1.0, 2.0, 3.0]])

        output_sizes = {'a': 4}
        
        model = models.get_model(
            base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
            feature_dim=input_features.shape[-1],
            num_hidden_nodes=2,
            output_sizes=output_sizes,
            weight_initializer=torch.nn.init.ones_,
            bias_initializer=torch.nn.init.zeros_,
            weight_max_norm=0.0,
            use_batch_norm=False,
            dropout_rate=0.0,
            num_fcs_per_block=2,
            num_fc_blocks=3
        )

        # Switch to eval mode
        model.eval()

        # Forward pass
        outputs, activations = model(input_features)

        # Check outputs
        self.assertSetEqual(set(outputs.keys()), {'a'})
        self.assertTrue(torch.allclose(outputs['a'], torch.Tensor([[1500.0, 1500.0, 1500.0, 1500.0]])))
        
        # Check activations
        self.assertSetEqual(set(activations.keys()), {'base_activations'})
        self.assertTrue(torch.allclose(activations['base_activations'], torch.Tensor([[750.0, 750.0]])))

    def test_get_simple_point_embedder(self):
        # Shape = [4, 2, 3].
        input_features = torch.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                      [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                                      [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]])
        embedder_fn = models.get_embedder(
            base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
            embedding_type=constants.EMBEDDING_TYPE_POINT,
            num_embedding_components=3,
            embedding_size=16,
            feature_dim=input_features.shape[-1])
        outputs, activations = embedder_fn(input_features)

        self.assertCountEqual(outputs.keys(), [constants.KEY_EMBEDDING_MEANS])
        self.assertEqual(outputs[constants.KEY_EMBEDDING_MEANS].shape,
                            torch.Size([4, 2, 3, 16]))
        self.assertCountEqual(activations.keys(), ['base_activations'])
        self.assertEqual(activations['base_activations'].shape,
                            torch.Size([4, 2, 512]))

    def test_get_simple_gaussian_embedder(self):
        # Shape = [4, 2, 3].
        input_features = torch.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                      [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                                      [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]])
        embedder_fn = models.get_embedder(
            base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
            embedding_type=constants.EMBEDDING_TYPE_GAUSSIAN,
            num_embedding_components=5,
            embedding_size=16,
            num_embedding_samples=20,
            is_training=True,
            weight_max_norm=0.0,
            feature_dim=input_features.shape[-1])
        outputs, activations = embedder_fn(input_features)

        self.assertCountEqual(outputs.keys(), [
            constants.KEY_EMBEDDING_MEANS,
            constants.KEY_EMBEDDING_STDDEVS,
            constants.KEY_EMBEDDING_SAMPLES,
        ])
        self.assertEqual(outputs[constants.KEY_EMBEDDING_MEANS].shape,
                            torch.Size([4, 2, 3, 16]))
        self.assertEqual(outputs[constants.KEY_EMBEDDING_STDDEVS].shape,
                            torch.Size([4, 2, 3, 16]))
        self.assertEqual(outputs[constants.KEY_EMBEDDING_SAMPLES].shape,
                            torch.Size([4, 2, 3, 32, 16]))
        self.assertCountEqual(activations.keys(), ['base_activations'])
        self.assertEqual(activations['base_activations'].shape,
                            torch.Size([4, 2, 512]))

# Run the tests
if __name__ == '__main__':
    unittest.main()