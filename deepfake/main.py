import tensorflow as tf
rnn_model = tf.keras.models.load_model('D:\Pratyakshai\deepfakedetection\ml_models\rnn_lstm.h5')
print(rnn_model.input_shape)
