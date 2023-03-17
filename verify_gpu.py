import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f'GPU available: {gpu}')
else:
    print('No GPU available')