import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters=64):
    fx = layers.Conv2D(filters, 3, padding='same')(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Activation('relu')(fx)
    fx = layers.Conv2D(filters, 3, padding='same')(fx)
    fx = layers.BatchNormalization()(fx)
    
    out = layers.Add()([x, fx])
    out = layers.Activation('relu')(out)
    return out

def create_chess_model(input_shape=(8, 8, 12), num_actions=4672, num_filters=64, num_res_blocks=8):
    num_move_planes = 73 

    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, filters=num_filters)

    policy_x = layers.Conv2D(filters=num_move_planes, kernel_size=1, padding='same')(x)
    policy_x = layers.BatchNormalization()(policy_x)
    policy_x = layers.Activation('relu')(policy_x)
    policy_x = layers.Reshape((num_actions,))(policy_x) 
    policy_out = layers.Softmax(name='policy')(policy_x)

    value_x = layers.Conv2D(filters=1, kernel_size=1, padding='same')(x)
    value_x = layers.BatchNormalization()(value_x)
    value_x = layers.Activation('relu')(value_x)
    value_x = layers.Flatten()(value_x)
    value_x = layers.Dense(256, activation='relu')(value_x)
    value_out = layers.Dense(1, activation='tanh', name='value')(value_x)

    model = models.Model(inputs=inputs, outputs=[policy_out, value_out])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            'policy': 'categorical_crossentropy', 
            'value': 'mean_squared_error'
        },
        loss_weights={'policy': 1.0, 'value': 1.0}
    )
    return model

def save_model(model, path):
    model.save_weights(path)

def load_model(path, num_actions):
    model = create_chess_model(num_actions=num_actions)
    model.load_weights(path)
    return model
