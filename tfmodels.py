### TENSORFLOW MODELS

"""
  This is where tensorflow deep neural nets are created, and transformation occur on their results
"""


## imports
import numpy as np
import pandas as pd
import tensorflow as tf
import math

def autoencoder(input_size:int, code_ratio:int, dropout_rate=0.4, myactivation="relu"):
    """
    Returns the classic autoencoder model.
    3 hidden layers in encoder and decoder each.
    """

    ratios = np.geomspace(1,code_ratio,5)
    code_size = math.floor(input_size/code_ratio)
    hidden_1_size = math.floor(input_size/ratios[1])
    hidden_2_size = math.floor(input_size/ratios[2])
    hidden_3_size = math.floor(input_size/ratios[3])

    input_data = tf.keras.layers.Input(shape=input_size, name='input_data')
    hidden_1 = tf.keras.layers.Dense(hidden_1_size, activation=myactivation, name='hidden_1')(input_data)
    e_drop_1 = tf.keras.layers.Dropout(dropout_rate, name='e_drop_1')(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_2_size, activation=myactivation, name='hidden_2')(e_drop_1)
    e_drop_2 = tf.keras.layers.Dropout(dropout_rate, name='e_drop_2')(hidden_2)
    hidden_3 = tf.keras.layers.Dense(hidden_3_size, activation=myactivation, name='hidden_3')(e_drop_2)

    code = tf.keras.layers.Dense(code_size, activation=myactivation, name='code')(hidden_3)

    hidden_3_rev = tf.keras.layers.Dense(hidden_3_size, activation=myactivation, name='hidden_3_rev')(code)
    d_drop_1 = tf.keras.layers.Dropout(dropout_rate, name='d_drop_1')(hidden_3_rev)
    hidden_2_rev = tf.keras.layers.Dense(hidden_2_size, activation=myactivation, name='hidden_2_rev')(d_drop_1)
    d_drop_2 = tf.keras.layers.Dropout(dropout_rate, name='d_drop_2')(hidden_2_rev)
    hidden_1_rev = tf.keras.layers.Dense(hidden_1_size, activation=myactivation, name='hidden_1_rev')(d_drop_2)
    output_data = tf.keras.layers.Dense(input_size, activation=myactivation, name='output_data')(hidden_1_rev) 

    return tf.keras.models.Model(input_data, output_data)

class Sampling(tf.keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    Used in VAE encoding layers.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def nll(y_true, y_pred):
    """
    Negative log likelihood (Bernoulli).
    Used in VAE training.
    """
    ## From Louis Tiao's post
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return tf.keras.backend.sum(tf.keras.backend.binary_crossentropy(y_true, y_pred), axis=-1)

class VAE(tf.keras.Model):
    """
    Model class used for variational autoencoder
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            ## ELBO RECONSTRUCTION LOSS: 
            reconstruction_loss = tf.reduce_mean( nll(data, reconstruction) )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            ## BASE TOTAL LOSS:
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        ## ELBO RECONSTRUCTION LOSS: 
        reconstruction_loss = tf.reduce_mean( nll(data, reconstruction) )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        ## BASE TOTAL LOSS:
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    


def variational_autoencoder(input_size:int, code_ratio:int, dropout_rate=0.4, myactivation="sigmoid"):
    """
    Returns the variational autoencoder used for most of the project.
    3 hidden layers in the encoder and decoder each.
    """

    # define size fo each layer
    ratios = np.geomspace(1,code_ratio,5)
    code_size = math.floor(input_size/code_ratio)
    hidden_1_size = math.floor(input_size/ratios[1])
    hidden_2_size = math.floor(input_size/ratios[2])
    hidden_3_size = math.floor(input_size/ratios[3])

    # encoder structure
    input_data = tf.keras.layers.Input(shape=input_size)
    hidden_1 = tf.keras.layers.Dense(hidden_1_size, activation=myactivation)(input_data)
    e_drop_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_2_size, activation=myactivation)(e_drop_1)
    e_drop_2 = tf.keras.layers.Dropout(dropout_rate)(hidden_2)
    hidden_3 = tf.keras.layers.Dense(hidden_3_size, activation=myactivation)(e_drop_2)

    # latent space, defined as a gaussian distribution for each sample
    code_mean = tf.keras.layers.Dense(code_size, name="code_mean")(hidden_3)
    code_log_var = tf.keras.layers.Dense(code_size, name="code_log_var")(hidden_3)
    code = Sampling()([code_mean, code_log_var])

    # decoder structure
    latent_inputs = tf.keras.layers.Input(shape=(code_size,))
    hidden_3_rev = tf.keras.layers.Dense(hidden_3_size, activation=myactivation)(latent_inputs)
    d_drop_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_3_rev)
    hidden_2_rev = tf.keras.layers.Dense(hidden_2_size, activation=myactivation)(d_drop_1)
    d_drop_2 = tf.keras.layers.Dropout(dropout_rate)(hidden_2_rev)
    hidden_1_rev = tf.keras.layers.Dense(hidden_1_size, activation=myactivation)(d_drop_2)
    output_data = tf.keras.layers.Dense(input_size, activation=myactivation)(hidden_1_rev)

    encoder = tf.keras.models.Model(input_data, [code_mean, code_log_var, code], name="encoder")
    decoder = tf.keras.models.Model(latent_inputs, output_data, name="decoder")
    return VAE(encoder, decoder)



def decode_sampling(decoder_f:str, sampling_f:str, out_f:str):
    """
    Load the decoder model with keras, then decode the sampling data and save it as csv
    """
    decoder = tf.keras.models.load_model(decoder_f)
    sampling_data = pd.read_csv(sampling_f, sep=";", header=0, dtype=np.float64)
    IDs=list(sampling_data.columns)
    sampling_data = sampling_data.to_numpy()
    decoded_data = np.array(decoder(sampling_data.transpose())).transpose()
    np.savetxt(out_f, decoded_data, delimiter=";", header=";".join(IDs))
    return "Decoded data written to " + out_f

