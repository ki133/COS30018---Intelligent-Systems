"""
model_builder.py
Task C.4: Flexible Deep Learning Model Construction

Provides a generic function to build sequence models (LSTM / GRU / SimpleRNN)
with configurable depth, widths, dropout, bidirectionality, and optimizer setup.

Inspired by P1 reference `create_model` but extended with:
- Per-layer unit sizes
- Choice of recurrent cell type via string
- Optional bidirectional wrapper per-layer or global
- Optional layer normalization
- Configurable output activation
- Adjustable optimizer with learning rate
- Return model plus parameter summary dict (useful for experiment logging)

Author: Auto-generated assistant (explanatory comments in Vietnamese + English)
Date: 2025-09-14
"""
from __future__ import annotations
from typing import List, Union, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dropout, Dense, Bidirectional, Layer, InputLayer
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# Mapping layer name to actual Keras class
RECURRENT_LAYER_MAP = {
    'lstm': LSTM,
    'gru': GRU,
    'rnn': SimpleRNN,
    'simplernn': SimpleRNN,
}

OPTIMIZER_MAP = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'sgd': SGD,
}

def build_sequence_model(
    sequence_length: int,
    n_features: int,
    layer_type: str = 'lstm',
    layer_units: Union[int, List[int]] = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    bidirectional: bool = False,
    last_layer_units: int = 1,
    output_activation: Optional[str] = 'linear',
    optimizer: str = 'adam',
    learning_rate: Optional[float] = None,
    loss: str = 'mean_squared_error',
    metrics: Optional[List[str]] = None,
    return_sequences_strategy: str = 'auto',  # 'auto' or explicit list[bool]
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a configurable recurrent sequence model.

    Parameters
    ----------
    sequence_length : int
        Number of timesteps in each input sample.
    n_features : int
        Number of features per timestep.
    layer_type : str
        One of {'lstm','gru','rnn'}. Case-insensitive.
    layer_units : int | List[int]
        Units for each recurrent layer; if int -> repeated for all layers.
    dropout : float
        Dropout applied after each recurrent layer.
    recurrent_dropout : float
        Recurrent dropout (inside recurrent cell). May slow training.
    bidirectional : bool
        Wrap each recurrent layer inside Bidirectional if True.
    last_layer_units : int
        Units of final Dense output layer (default 1 for regression).
    output_activation : str | None
        Activation for output Dense. For regression typically 'linear'.
    optimizer : str
        Optimizer key (adam, rmsprop, sgd).
    learning_rate : float | None
        Custom learning rate. If None uses framework default.
    loss : str
        Loss function (e.g., 'mse', 'mae', 'huber').
    metrics : list[str] | None
        Additional metrics. Defaults to ['mae'] if None.
    return_sequences_strategy : str
        'auto' sets return_sequences=True for all but last recurrent layer.
        You can alternatively pass an explicit list of booleans (same len as layers).
    name : str | None
        Optional name for the model.

    Returns
    -------
    dict with keys:
        model: compiled tf.keras.Model
        config: parameter dictionary used to build model
        summary_str: textual model summary (for logging)
    """
    layer_type_key = layer_type.lower().strip()
    assert layer_type_key in RECURRENT_LAYER_MAP, f"Unsupported layer_type: {layer_type}"
    CellClass = RECURRENT_LAYER_MAP[layer_type_key]

    # Normalize layer_units to list
    if isinstance(layer_units, int):
        layer_units_list = [layer_units]
    else:
        assert len(layer_units) > 0, "layer_units list must not be empty"
        layer_units_list = list(layer_units)

    n_layers = len(layer_units_list)

    # Determine return_sequences flags
    if return_sequences_strategy == 'auto':
        return_sequences_flags = [True]*(n_layers-1) + [False]
    elif isinstance(return_sequences_strategy, list):
        assert len(return_sequences_strategy) == n_layers, "return_sequences list length mismatch"
        return_sequences_flags = return_sequences_strategy
    else:
        raise ValueError("return_sequences_strategy must be 'auto' or list[bool]")

    # Optimizer setup
    opt_key = optimizer.lower().strip()
    assert opt_key in OPTIMIZER_MAP, f"Unsupported optimizer: {optimizer}"
    if learning_rate is None:
        optimizer_instance = OPTIMIZER_MAP[opt_key]()
    else:
        optimizer_instance = OPTIMIZER_MAP[opt_key](learning_rate=learning_rate)

    if metrics is None:
        metrics = ['mae']

    model = Sequential(name=name)
    model.add(InputLayer(input_shape=(sequence_length, n_features)))

    # Build recurrent stack
    for i, (units, ret_seq) in enumerate(zip(layer_units_list, return_sequences_flags)):
        layer_args = dict(units=units, return_sequences=ret_seq)
        # Only pass recurrent_dropout if > 0 (some layers may not support otherwise)
        if recurrent_dropout > 0:
            layer_args['recurrent_dropout'] = recurrent_dropout
        recurrent_layer: Layer = CellClass(**layer_args)
        if bidirectional:
            recurrent_layer = Bidirectional(recurrent_layer)
        model.add(recurrent_layer)
        if dropout > 0:
            model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(last_layer_units, activation=output_activation))

    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics)

    # Capture summary into string
    lines: List[str] = []
    model.summary(print_fn=lambda x: lines.append(x))
    summary_str = "\n".join(lines)

    config = {
        'sequence_length': sequence_length,
        'n_features': n_features,
        'layer_type': layer_type_key,
        'layer_units': layer_units_list,
        'dropout': dropout,
        'recurrent_dropout': recurrent_dropout,
        'bidirectional': bidirectional,
        'last_layer_units': last_layer_units,
        'output_activation': output_activation,
        'optimizer': opt_key,
        'learning_rate': learning_rate,
        'loss': loss,
        'metrics': metrics,
        'return_sequences_flags': return_sequences_flags,
        'param_count': model.count_params(),
    }

    return {
        'model': model,
        'config': config,
        'summary_str': summary_str,
    }

if __name__ == '__main__':
    # Quick self-test with small dummy input
    build_info = build_sequence_model(
        sequence_length=30,
        n_features=5,
        layer_type='GRU',
        layer_units=[64, 32],
        dropout=0.1,
        bidirectional=True,
        optimizer='adam',
        learning_rate=0.001,
        loss='mse'
    )
    print(build_info['summary_str'])
