#this code is being used for the GPU detection by the tensorflow-metal 
# we should use this sample code in every GPU and tensor based code
# without the right detection every heavy computation will need long time

import tensorflow as tf
import platform

print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Platform: {platform.platform()}")
print(f"Python Version: {platform.python_version()}")

print("Is Metal GPU available:")
print(tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    try:
        print("TensorFlow is configured to use Metal GPU.")
    except RuntimeError as e:
        print(e)
else:
    print("No Metal GPU found or TensorFlow is not configured to use it.")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

dummy_data = tf.random.normal((100, 784))
dummy_labels = tf.random.uniform((100,), minval=0, maxval=10, dtype=tf.int32)
print("\nRunning a dummy training step to confirm GPU usage (check console output for Metal/MPS logs):")
model.fit(dummy_data, dummy_labels, epochs=1, verbose=1)
