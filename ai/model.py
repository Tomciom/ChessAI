# ai/model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters=64):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def create_chess_model(input_shape=(8, 8, 12), num_actions=8000, num_filters=64, num_res_blocks=8):
    """
    Przykładowy model w stylu AlphaZero.
    Argumenty:
      - num_actions -> ustalamy według move_mapping.NUM_ACTIONS
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)

    for _ in range(num_res_blocks):
        x = residual_block(x, filters=num_filters)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    policy_logits = layers.Dense(num_actions, activation='linear')(x)
    policy_out = layers.Softmax(name='policy')(policy_logits)

    value_out = layers.Dense(1, activation='tanh', name='value')(x)

    model = models.Model(inputs=inputs, outputs=[policy_out, value_out])
    model.compile(
        optimizer='adam',
        loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
        loss_weights={'policy': 1.0, 'value': 1.0}
    )
    return model

def save_model(model, path):
    model.save_weights(path)

def load_model(path, num_actions):
    """
    Wczytanie wag do modelu o zadanym num_actions.
    """
    model = create_chess_model(num_actions=num_actions)
    model.load_weights(path)
    return model
