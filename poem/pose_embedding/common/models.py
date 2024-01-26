import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import functools
import operator
import os


from pose_embedding.common import constants, data_utils

import torch.nn as nn
import torch

def _add_prefix(key, c): return 'C%d/' % c + key

def _stddev_activation(x):
    """Activation function for standard deviation logits.

    Args:
      x: A tensor for standard deviation logits.

    Returns:
      A tensor for non-negative standard deviations.
    """
    return F.elu(x) + 1.0

class LinearWithClipping(nn.Linear):
    def __init__(self, in_features, num_hidden_nodes, bias=True, weight_max_norm=0.0,
                #  linear_weight_initializer=torch.nn.init.ones_, linear_bias_initializer=init.zeros_):
                 weight_initializer=init.kaiming_normal_, bias_initializer=init.zeros_, **kwargs):
        super(LinearWithClipping, self).__init__(in_features, num_hidden_nodes, bias)
        self.weight_max_norm = weight_max_norm
        
        # Initialize with He's initialization
        # linear_bias_initializer(self.weight, mode='fan_in', nonlinearity='relu')
        weight_initializer(self.weight) #, mode='fan_in', nonlinearity='relu')
        bias_initializer(self.bias)

    def forward(self, x):
        def clip_weight_norm(weight, max_norm):
            norm = torch.norm(weight)
            if norm > max_norm:
                weight = (weight * max_norm) / norm
            return weight        
        
        # Optionally clip weights
        if self.weight_max_norm > 0.0:
            weight_clipped = clip_weight_norm(self.weight, self.weight_max_norm)
        else:
            weight_clipped = self.weight    

        return nn.functional.linear(x, weight_clipped, self.bias)



class FullyConnected(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(FullyConnected, self).__init__()
        
        self.linear = LinearWithClipping(input_dim, **kwargs) #, kwargs.get('num_hidden_nodes', 1024), weight_max_norm=kwargs.get('weight_max_norm', 0.0))
        
        self.use_batch_norm = kwargs.get('use_batch_norm', True)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(kwargs.get('num_hidden_nodes', 1024))
            
        self.dropout = nn.Dropout(kwargs.get('dropout_rate', 0.0))
        
        # For the sake of simplicity, I am assuming only ReLU is used here as it's the only one mentioned before.
        self.activation = nn.ReLU()

    def forward(self, x, is_training=True):
        # Store original shape for later
        original_shape = x.shape

        # Flatten tensor if it's 3D
        if len(original_shape) == 3:
            x = x.view(-1, original_shape[-1])

        x = self.linear(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        
        if is_training:
            x = self.dropout(x)

        # Reshape back to 3D tensor if original input was 3D
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)
        
        return x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(FullyConnectedBlock, self).__init__()
        
        # Extracting key arguments
        self.num_fcs_per_block = kwargs.get('num_fcs_per_block', 2)
        
        # Assuming the size of all intermediate FC layers is the same (which matches with `num_hidden_nodes`)
        self.fc_layers = nn.ModuleList([FullyConnected(input_dim, **kwargs) for _ in range(self.num_fcs_per_block)])
        
    def forward(self, x, is_training=True):
        residual = x
        for layer in self.fc_layers:
            x = layer(x, is_training=is_training)
        x = x + residual
        return x

class MultiHeadLogits(nn.Module):
    def __init__(self, input_dim, output_sizes, **kwargs):
        super(MultiHeadLogits, self).__init__()

        self.output_layers = nn.ModuleDict()
        self.output_shapes = {}

        ########### 
        ## Fix required!!
        num_hidden_nodes = None
        if 'num_hidden_nodes' in kwargs:
            num_hidden_nodes = kwargs.pop('num_hidden_nodes')
        
        for output_name, output_size in output_sizes.items():
            if isinstance(output_size, int):
                self.output_layers[output_name] = LinearWithClipping(input_dim, num_hidden_nodes=output_size, **kwargs)
            else:
                self.output_layers[output_name] = LinearWithClipping(input_dim, num_hidden_nodes=functools.reduce(operator.mul, output_size), **kwargs) #, **kwargs
                self.output_shapes[output_name] = output_size  # Store the desired shape
        if num_hidden_nodes is not None:
            kwargs["num_hidden_nodes"] = num_hidden_nodes
        ########### 
        ## Fix required!!


    def forward(self, input_features):
        outputs = {}
        
        for output_name, layer in self.output_layers.items():
            out = layer(input_features)
            
            # If the expected output is not just a flat vector, reshape it
            if output_name in self.output_shapes:
                output_size = self.output_shapes[output_name]
                
                out = data_utils.recursively_expand_dims(out, axes=[-1] * (len(output_size) - 1))
                out = data_utils.reshape_by_last_dims(out, last_dim_shape=output_size)

            outputs[output_name] = out

        return outputs


class SimpleBaseline(nn.Module):
    def __init__(self, feature_dim, sequence_length=1, sequential_inputs=False, **kwargs):
        super(SimpleBaseline, self).__init__()

        self.sequential_inputs = sequential_inputs

        # Adjust the input feature dimension if sequential inputs are used
        adjusted_feature_dim = feature_dim * sequence_length if sequential_inputs else feature_dim

        # Initial fully-connected layer
        if 'num_hidden_nodes' not in kwargs:
            kwargs['num_hidden_nodes'] = 1024
        self.input_fc = LinearWithClipping(adjusted_feature_dim, **kwargs)

        # Fully connected blocks
        self.fc_blocks = nn.ModuleList([
            FullyConnectedBlock(input_dim=kwargs.get('num_hidden_nodes', 1024), **kwargs) for _ in range(kwargs.get('num_fc_blocks', 2))
        ])

    def forward(self, input_features, is_training=True):
        if self.sequential_inputs:
            # Flatten last two dimensions
            input_features = input_features.view(input_features.shape[:-2] + (-1,))

        net = self.input_fc(input_features)

        for block in self.fc_blocks:
            net = block(net, is_training=is_training)

        return net


class SimpleModel(nn.Module):
    def __init__(self, feature_dim, output_sizes, sequence_length=1, sequential_inputs=False, num_bottleneck_nodes=0, **kwargs):
        super(SimpleModel, self).__init__()

        # Base layers
        self.simple_base = SimpleBaseline(feature_dim, sequence_length=sequence_length, sequential_inputs=sequential_inputs, **kwargs)
        
        # Optional bottleneck layer
        self.use_bottleneck = num_bottleneck_nodes > 0
        if self.use_bottleneck:
            self.bottleneck = LinearWithClipping(
                kwargs.get('num_hidden_nodes', 1024),  # This assumes that the output of the base model is of size num_hidden_nodes.
                num_bottleneck_nodes,
                weight_max_norm=kwargs.get('weight_max_norm', 0.0)
            )
        
        # Multi-head logits layer
        self.multi_head_logits = MultiHeadLogits(
            num_bottleneck_nodes if self.use_bottleneck else kwargs.get('num_hidden_nodes', 1024),
            output_sizes,
            **kwargs
        )

    def forward(self, input_features):
        base_activations = self.simple_base(input_features)

        # Pass through bottleneck layer if it exists
        if self.use_bottleneck:
            bottleneck_activations = self.bottleneck(base_activations)
        else:
            bottleneck_activations = base_activations

        outputs = self.multi_head_logits(bottleneck_activations)
        activations = {'base_activations': base_activations}

        # Store bottleneck activations if they exist
        if self.use_bottleneck:
            activations['bottleneck_activations'] = bottleneck_activations

        return outputs, activations


def create_model_helper(base_model_fn, sequential_inputs):
    """Helper function for creating model function given base model function.

    Args:
      base_model_fn: A function handle for base model.
      sequential_inputs: A boolean for whether the model input features are sequential.

    Returns:
      model_fn: A function handle for model.
    """
    class ModelWrapper(torch.nn.Module):
        def __init__(self,  **kwargs): #output_sizes):
            super(ModelWrapper, self).__init__()
            # prepare feature_dim according to sequential_inputs flag

            self.model = base_model_fn(**kwargs)
            self.sequential_inputs = sequential_inputs

        def forward(self, input_features):
            """Applies model to input features."""
            if self.training:
                # Flatten all the model-irrelevant dimensions, i.e., dimensions that
                # precede the sequence / feature channel dimensions.
                num_last_dims_to_keep = 2 if self.sequential_inputs else 1
                flattened_input_features = data_utils.flatten_first_dims(
                    input_features, num_last_dims_to_keep=num_last_dims_to_keep)
                flattened_shape = data_utils.get_shape_by_first_dims(
                    input_features, num_last_dims=num_last_dims_to_keep)                

                outputs, activations = self.model(flattened_input_features)

                # Unflatten back all the model-irrelevant dimensions.
                for key, output in outputs.items():
                    outputs[key] = data_utils.unflatten_first_dim(
                        output, shape_to_unflatten=flattened_shape)
                for key, activation in activations.items():
                    activations[key] = data_utils.unflatten_first_dim(
                        activation, shape_to_unflatten=flattened_shape)
            else:
                outputs, activations = self.model(input_features)

            return outputs, activations

    return ModelWrapper


def get_model(base_model_type, **kwargs):
    """Gets a base model builder function handle.
    
    Args:
      base_model_type: An enum for base model type.
      **kwargs: A dictionary of additional arguments.
      
    Returns:
      A PyTorch model instance.

    Raises:
      ValueError: If base model type is not supported.
    """
    if base_model_type == constants.BASE_MODEL_TYPE_SIMPLE:
        model_cls = create_model_helper(SimpleModel, sequential_inputs=False)
        return model_cls(**kwargs)
    elif base_model_type == constants.BASE_MODEL_TYPE_TEMPORAL_SIMPLE:
        model_cls = create_model_helper(SimpleModel, sequential_inputs=True)
        return model_cls(**kwargs)
    else:
        raise ValueError(f'Unsupported base model type: `{base_model_type}`.')


def _point_embedder(input_features, base_model_fn, num_embedding_components):
    """Implements a point embedder.

    Output tensor shapes:
      KEY_EMBEDDING_MEANS: Shape = [..., num_embedding_components, embedding_dim].

    Args:
      input_features: A tensor for input features. Shape = [..., feature_dim].
      base_model_fn: A function handle for base model.
      num_embedding_components: An integer for the number of embedding components.

    Returns:
      outputs: A dictionary for output tensors See comment above for details.
      activations: A dictionary of addition activation tensors for pre-output
        model activations. Keys include 'base_activations' and optionally
        'bottleneck_activations'.
    """


    component_outputs, activations = base_model_fn(input_features)

    outputs = {
        constants.KEY_EMBEDDING_MEANS:
            torch.stack(
                [
                component_outputs[_add_prefix(constants.KEY_EMBEDDING_MEANS, c)]
                for c in range(num_embedding_components)
                ], 
                axis=-2
                )
    }
    return outputs, activations


def _gaussian_embedder(input_features,
                       base_model_fn,
                       num_embedding_components,
                       num_embedding_samples,
                       seed=None):
    """Implements a Gaussian (mixture) embedder.

    Output tensor shapes:
      KEY_EMBEDDING_MEANS: Shape = [..., num_embedding_components,
        embedding_dim].
      KEY_EMBEDDING_STDDEVS: Shape = [..., num_embedding_components,
        embedding_dim].
      KEY_EMBEDDING_SAMPLES: Shape = [..., num_embedding_components, num_samples,
        embedding_dim].

    Args:
      input_features: A tensor for input features. Shape = [..., feature_dim].
      base_model_fn: A function handle for base model.
      num_embedding_components: An integer for the number of Gaussian mixture
        components.
      num_embedding_samples: An integer for number of samples drawn Gaussian
        distributions. If non-positive, skips the sampling step.
      seed: An integer for random seed.

    Returns:
      outputs: A dictionary for output tensors See comment above for details.
      activations: A dictionary of addition activation tensors for pre-output
        model activations. Keys include 'base_activations' and optionally
        'bottleneck_activations'.
    """

    component_outputs, activations = base_model_fn(
        input_features)

    for c in range(num_embedding_components):
        embedding_stddev_key = _add_prefix(constants.KEY_EMBEDDING_STDDEVS, c)
        component_outputs[embedding_stddev_key] = _stddev_activation(
            component_outputs[embedding_stddev_key])

        if num_embedding_samples > 0:
            component_outputs[_add_prefix(constants.KEY_EMBEDDING_SAMPLES, c)] = (
                data_utils.sample_gaussians(
                    means=component_outputs[_add_prefix(constants.KEY_EMBEDDING_MEANS,
                                                        c)],
                    stddevs=component_outputs[embedding_stddev_key],
                    num_samples=num_embedding_samples,
                    seed=seed))

    outputs = {
        constants.KEY_EMBEDDING_MEANS:
            torch.stack([
                component_outputs[_add_prefix(constants.KEY_EMBEDDING_MEANS, c)]
                for c in range(num_embedding_components)
            ],
                axis=-2),
        constants.KEY_EMBEDDING_STDDEVS:
            torch.stack([
                component_outputs[_add_prefix(constants.KEY_EMBEDDING_STDDEVS, c)]
                for c in range(num_embedding_components)
            ],
                axis=-2),
    }
    if num_embedding_samples > 0:
        outputs[constants.KEY_EMBEDDING_SAMPLES] = torch.stack([
            component_outputs[_add_prefix(constants.KEY_EMBEDDING_SAMPLES, c)]
            for c in range(num_embedding_components)
        ],
            axis=-3)

    return outputs, activations


def create_embedder_helper(base_model_type, embedding_type,
                           num_embedding_components, embedding_size, **kwargs):
    """Helper function for creating an embedding model builder function handle.

    Args:
      base_model_type: An enum string for base model type. See supported base model
        types in the `constants` module.
      embedding_type: An enum string for embedding type. See supported embedding
        types in the `constants` module.
      num_embedding_components: An integer for the number of embedding components.
      embedding_size: An integer for embedding dimensionality.
      **kwargs: A dictionary of additional arguments to embedder.

    Returns:
      A function handle for embedding model builder.

    Raises:
      ValueError: If base model type or embedding type is not supported.
    """
    if embedding_type == constants.EMBEDDING_TYPE_POINT:
        output_sizes = {
            _add_prefix(constants.KEY_EMBEDDING_MEANS, c): embedding_size
            for c in range(num_embedding_components)
        }  
        base_model_fn = get_model(base_model_type, output_sizes=output_sizes, **kwargs)
        return functools.partial(
            _point_embedder,
            base_model_fn=base_model_fn,
            num_embedding_components=num_embedding_components
            )

    if embedding_type == constants.EMBEDDING_TYPE_GAUSSIAN:
        output_sizes = {}
        scalar_stddev = False
        """
        scalar_stddev: A boolean for whether to predict scalar standard deviations.
        """
        for c in range(num_embedding_components):
            output_sizes.update({
                _add_prefix(constants.KEY_EMBEDDING_MEANS, c):
                    embedding_size,
                _add_prefix(constants.KEY_EMBEDDING_STDDEVS, c):
                    1 if scalar_stddev else embedding_size,
            })       
        base_model_fn = get_model(base_model_type, output_sizes=output_sizes, **kwargs)            
        return functools.partial(
            _gaussian_embedder,
            base_model_fn=base_model_fn,
            num_embedding_components=num_embedding_components,
            num_embedding_samples=kwargs.get('num_embedding_samples'),
            seed=kwargs.get('seed', None))

    if embedding_type == constants.EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR:
        output_sizes = {}
        scalar_stddev = True
        """
        scalar_stddev: A boolean for whether to predict scalar standard deviations.
        """
        for c in range(num_embedding_components):
            output_sizes.update({
                _add_prefix(constants.KEY_EMBEDDING_MEANS, c):
                    embedding_size,
                _add_prefix(constants.KEY_EMBEDDING_STDDEVS, c):
                    1 if scalar_stddev else embedding_size,
            })       
        base_model_fn = get_model(base_model_type, output_sizes=output_sizes, **kwargs)            
        return functools.partial(
            _gaussian_embedder,
            base_model_fn=base_model_fn,
            num_embedding_components=num_embedding_components,
            embedding_size=embedding_size,
            scalar_stddev=scalar_stddev,
            num_embedding_samples=kwargs.get('num_embedding_samples'),
            seed=kwargs.get('seed', None))

    raise ValueError('Unsupported embedding type: `%s`.' % str(embedding_type))


def get_embedder(base_model_type, embedding_type, num_embedding_components,
                 embedding_size, **kwargs):
    """Gets an embedding model builder function handle.

    Args:
      base_model_type: An enum string for base model type. See supported base
        model types in the `constants` module.
      embedding_type: An enum string for embedding type. See supported embedding
        types in the `constants` module.
      num_embedding_components: An integer for the number of embedding components.
      embedding_size: An integer for embedding dimensionality.
      **kwargs: A dictionary of additional arguments to pass to base model and
        embedder.

    Returns:
      A function handle for embedding model builder.
    """
    return create_embedder_helper(
        base_model_type,
        embedding_type,
        num_embedding_components=num_embedding_components,
        embedding_size=embedding_size,
        **kwargs)

def save_embedder(embedder_fn, pth_path):
    # Check if a file with the same name exists
    if os.path.exists(pth_path):
        print(f"Warning: {pth_path} already exists and will be overwritten.")

    try:
        base_model_fn = embedder_fn.keywords.get('base_model_fn', None)
        if base_model_fn is None:
            print("Keyword argument 'base_model_fn' is missing.")
            return

        if not hasattr(base_model_fn, 'model'):
            print("'base_model_fn' does not have a 'model' attribute.")
            return
        
        torch.save(base_model_fn.model.state_dict(), pth_path)
    except Exception as e:
        print(f"Failed to save embedder function: {e}")


def load_embedder(embedder_fn, pth_path):
    try:
        state_dict = torch.load(pth_path, map_location='cpu')  # Load to CPU
    except Exception as e:
        print(f"Failed to load state dict from {pth_path}: {e}")
        return None
    
    try:
        base_model_fn = embedder_fn.keywords.get('base_model_fn', None)
        if base_model_fn is None:
            print("Keyword argument 'base_model_fn' is missing.")
            return None
        
        if not hasattr(base_model_fn, 'model'):
            print("'base_model_fn' does not have a 'model' attribute.")
            return None

        base_model_fn.model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to update embedder function: {e}")
        return None
    
    return embedder_fn

def save_training_state(embedder_fn, optimizer, scalar_parameters, ckpt_path):
    epoch = scalar_parameters["epoch"]
    epoch_step = scalar_parameters["epoch_step"]
    training_state = {
        'model_state_dict': embedder_fn.keywords['base_model_fn'].model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'epoch_step': epoch_step,
        'raw_a': scalar_parameters["raw_a"],
        'raw_b': scalar_parameters["raw_b"]
    }
    torch.save(training_state, os.path.join(ckpt_path, f"training_model_{epoch}_{epoch_step}.pth"))

def load_training_state(embedder_fn, optimizer, scalar_parameters, ckpt_resume_path):
    training_state = torch.load(ckpt_resume_path, map_location='cpu')

    try:
        base_model_fn = embedder_fn.keywords.get('base_model_fn', None)
        if base_model_fn is None:
            print("Keyword argument 'base_model_fn' is missing.")
            return False
        
        if not hasattr(base_model_fn, 'model'):
            print("'base_model_fn' does not have a 'model' attribute.")
            return False

        base_model_fn.model.load_state_dict(training_state['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
        if scalar_parameters is not None:
            scalar_parameters['epoch_step'] = training_state.get('epoch_step', 0)
            scalar_parameters['epoch'] = training_state.get('epoch', 0)
            scalar_parameters['raw_a'] = training_state['raw_a']
            scalar_parameters['raw_b'] = training_state['raw_b']
    except Exception as e:
        print(f"Failed to update embedder function: {e}")
        return False
    return True


def get_learning_rate(schedule_type, init_learning_rate, decay_steps, global_step=None, num_warmup_steps=None, **kwargs):
    """Creates learning rate with schedules.

    Args:
        schedule_type (str): The type of learning rate schedule to choose.
        init_learning_rate (float): The initial learning rate.
        decay_steps (int): The number of decay steps.
        global_step (int, optional): The global step. Defaults to None.
        num_warmup_steps (int, optional): The number of linear warmup training steps.
                                          Defaults to None.
        **kwargs: Additional keyword arguments for learning rate schedulers.
    
    Returns:
        float: The learning rate according to the schedule.

    Raises:
        ValueError: If the schedule type is not supported.
    """

    if global_step is None:
        global_step = 0  # Assuming a default value of 0 for global_step

    learning_rate = init_learning_rate

    if schedule_type:
        if schedule_type == 'EXP_DECAY':
            decay_rate = kwargs.get('EXP_DECAY_decay_rate', 0.9)
            staircase = kwargs.get('EXP_DECAY_staircase', False)
            if staircase:
                global_step = global_step // decay_steps
            learning_rate *= decay_rate ** (global_step / decay_steps)
        elif schedule_type == 'LINEAR_DECAY':
            end_learning_rate = kwargs.get('LINEAR_DECAY_end_learning_rate', 0.0)
            cycle = kwargs.get('LINEAR_DECAY_cycle', False)
            if cycle:
                global_step = global_step % decay_steps
            learning_rate = end_learning_rate + (init_learning_rate - end_learning_rate) * (1 - global_step / decay_steps)
        else:
            raise ValueError(f'Unsupported learning rate schedule type: {schedule_type}')

    # Implement linear warmup
    if num_warmup_steps:
        if global_step < num_warmup_steps:
            warmup_percent_done = global_step / num_warmup_steps
            learning_rate = init_learning_rate * warmup_percent_done

    return learning_rate


def get_optimizer(optimizer_type, learning_rate, model_parameters, **kwargs):
    """Creates PyTorch optimizer with learning rate.

    Args:
      optimizer_type (str): The type of optimizer to choose.
      learning_rate (float): The learning rate.
      model_parameters (iterable): Model parameters to optimize.
      **kwargs: Additional keyword arguments for optimizers.
    
    Returns:
      torch.optim.Optimizer: An optimizer instance.

    Raises:
      ValueError: If the optimizer type is not supported.
    """

    if optimizer_type == 'ADAGRAD':
        optimizer = optim.Adagrad(
            model_parameters,
            lr=learning_rate,
            initial_accumulator_value=kwargs.get('ADAGRAD_initial_accumulator_value', 0.1)
        )
    elif optimizer_type == 'ADAM':
        optimizer = optim.Adam(
            model_parameters,
            lr=learning_rate,
            betas=(kwargs.get('ADAM_beta1', 0.9), kwargs.get('ADAM_beta2', 0.999)),
            eps=kwargs.get('ADAM_epsilon', 1e-8)
        )
    elif optimizer_type == 'ADAMW':
        optimizer = optim.AdamW(
            model_parameters,
            lr=learning_rate,
            betas=(kwargs.get('ADAMW_beta1', 0.9), kwargs.get('ADAMW_beta2', 0.999)),
            eps=kwargs.get('ADAMW_epsilon', 1e-8),
            weight_decay=kwargs.get('ADAMW_weight_decay_rate', 0.01)
        )
    elif optimizer_type == 'RMSPROP':
        optimizer = optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            alpha=kwargs.get('RMSPROP_decay', 0.9),
            momentum=kwargs.get('RMSPROP_momentum', 0.9),
            eps=kwargs.get('RMSPROP_epsilon', 1e-10)
        )
    else:
        raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

    return optimizer

# Example usage:
# model_parameters = model.parameters()
# optimizer = get_optimizer('ADAM', 0.001, model_parameters, ADAM_beta1=0.9)

# Test cases
# print(get_learning_rate('', 0.01, 100, global_step=50, EXP_DECAY_decay_rate=0.9))
# print(get_learning_rate('LINEAR_DECAY', 0.01, 100, global_step=50, LINEAR_DECAY_end_learning_rate=0.001))
