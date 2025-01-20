import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras import layers, models

########################
# 1. Model rezydualny  #
########################

def residual_block(x, filters=64):
    """
    Prosty blok rezydualny:
     - 2x Conv2D(filters=filters, kernel_size=3, padding='same')
     - shortcut + ReLU
    """
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def create_chess_model(input_shape=(8, 8, 12), num_actions=4672, num_filters=64, num_res_blocks=8):
    """
    Przykładowa sieć wzorowana na architekturze rezydualnej, głębsza niż minimalne przykłady.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)

    for _ in range(num_res_blocks):
        x = residual_block(x, filters=num_filters)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Policy logits (zamiast od razu Softmax – można z tego wyciągać priorytety)
    policy_logits = layers.Dense(num_actions, activation='linear')(x)
    policy_out = layers.Softmax(name='policy')(policy_logits)

    # Value output
    value_out = layers.Dense(1, activation='tanh', name='value')(x)

    model = models.Model(inputs=inputs, outputs=[policy_out, value_out])
    model.compile(
        optimizer='adam',
        loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
        loss_weights={'policy': 1.0, 'value': 1.0}
    )
    return model
