
import tensorflow as tf

def build_model(input_shape):

    # hyperparams
    l2_reg = tf.keras.regularizers.l2(1e-4)
    use_bias = True
    out_dim = 13

    inpt = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=4,
        padding='same',
        use_bias=use_bias,
        kernel_regularizer=l2_reg
    )(inpt)

    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    for f in [32, 16]:

        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
            
        x = tf.keras.layers.Conv2D(
            filters=f, 
            kernel_size=2, 
            padding='same',
            use_bias=use_bias, 
            kernel_regularizer=l2_reg
        )(x)
        
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        
    x = tf.keras.layers.Flatten()(x)
    
    # top right y
    x = tf.keras.layers.Dense(out_dim)(x)
    out = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.Model(
        inputs=[inpt], 
        outputs=[out]
    )

    return model
