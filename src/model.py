import tensorflow as tf


def build_model(num_classes=6, image_size=(150,150), dropout=0.3, kernel_size=(3,3)):
  seq = tf.keras.Sequential([
    tf.keras.Input(shape=(*image_size, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(
      32,
      kernel_size=kernel_size,
      strides=(1,1),
      padding='valid',
      activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(
          64,
          kernel_size=kernel_size,
          strides=(1,1),
          padding='same',
          activation='relu'
        ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(
              128,
              kernel_size=kernel_size,
              strides=(1,1),
              padding='same',
              activation='relu'
            ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
  ])
  seq.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          loss="sparse_categorical_crossentropy",
          metrics=["accuracy"]
          )
  return seq





# model = build_model()
# model.summary()