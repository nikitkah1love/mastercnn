import tensorflow as tf
import gc

# Reset Keras Session
def reset_keras():
    """Clear TensorFlow/Keras session and free memory"""
    tf.keras.backend.clear_session()
    print(gc.collect())  # if it's done something you should see a number being outputted
    
    # Configure GPU memory growth if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")