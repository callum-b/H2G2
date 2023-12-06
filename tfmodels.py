### TENSORFLOW MODELS

"""
  This is where tensorflow deep neural nets are created, and transformation occur on their results
"""


## imports
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
from pathlib import Path

def autoencoder(input_size:int, code_ratio:int, dropout_rate=0.4, my_activation="relu", ls_activation="linear"):
    """
    Returns the classic autoencoder model, as well as the encoder and decoder.
    3 hidden layers in encoder and decoder each.
    """

    ratios = np.geomspace(1,code_ratio,5)
    code_size = math.floor(input_size/code_ratio)
    hidden_1_size = math.floor(input_size/ratios[1])
    hidden_2_size = math.floor(input_size/ratios[2])
    hidden_3_size = math.floor(input_size/ratios[3])

    input_data = tf.keras.layers.Input(shape=input_size, name='input_data')
    hidden_1 = tf.keras.layers.Dense(hidden_1_size, activation=my_activation, name='hidden_1')(input_data)
    e_drop_1 = tf.keras.layers.Dropout(dropout_rate, name='e_drop_1')(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_2_size, activation=my_activation, name='hidden_2')(e_drop_1)
    e_drop_2 = tf.keras.layers.Dropout(dropout_rate, name='e_drop_2')(hidden_2)
    hidden_3 = tf.keras.layers.Dense(hidden_3_size, activation=my_activation, name='hidden_3')(e_drop_2)

    code = tf.keras.layers.Dense(code_size, activation=ls_activation, name='code')(hidden_3)
    encoded_input = tf.keras.layers.Input(shape=(None, code_size))

    hidden_3_rev = tf.keras.layers.Dense(hidden_3_size, activation=my_activation, name='hidden_3_rev')(code)
    d_drop_1 = tf.keras.layers.Dropout(dropout_rate, name='d_drop_1')(hidden_3_rev)
    hidden_2_rev = tf.keras.layers.Dense(hidden_2_size, activation=my_activation, name='hidden_2_rev')(d_drop_1)
    d_drop_2 = tf.keras.layers.Dropout(dropout_rate, name='d_drop_2')(hidden_2_rev)
    hidden_1_rev = tf.keras.layers.Dense(hidden_1_size, activation=my_activation, name='hidden_1_rev')(d_drop_2)
    output_data = tf.keras.layers.Dense(input_size, activation=my_activation, name='output_data')(hidden_1_rev) 

    
    autoencoder = tf.keras.models.Model(input_data, output_data)
    encoder = tf.keras.models.Model(input_data, code)
    decoder = tf.keras.models.Model(encoded_input, autoencoder.layers[-1](autoencoder.layers[-2](autoencoder.layers[-3](
        autoencoder.layers[-4](autoencoder.layers[-5](autoencoder.layers[-6](encoded_input)))))) )

    return autoencoder, encoder, decoder

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
    


def variational_autoencoder(input_size:int, code_ratio:int, dropout_rate=0.4, my_activation="sigmoid", ls_activation="linear"):
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
    hidden_1 = tf.keras.layers.Dense(hidden_1_size, activation=my_activation)(input_data)
    e_drop_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_2_size, activation=my_activation)(e_drop_1)
    e_drop_2 = tf.keras.layers.Dropout(dropout_rate)(hidden_2)
    hidden_3 = tf.keras.layers.Dense(hidden_3_size, activation=my_activation)(e_drop_2)

    # latent space, defined as a gaussian distribution for each sample
    code_mean = tf.keras.layers.Dense(code_size, activation=ls_activation, name="code_mean")(hidden_3)
    code_log_var = tf.keras.layers.Dense(code_size, name="code_log_var")(hidden_3)
    code = Sampling()([code_mean, code_log_var])

    # decoder structure
    latent_inputs = tf.keras.layers.Input(shape=(code_size,))
    hidden_3_rev = tf.keras.layers.Dense(hidden_3_size, activation=my_activation)(latent_inputs)
    d_drop_1 = tf.keras.layers.Dropout(dropout_rate)(hidden_3_rev)
    hidden_2_rev = tf.keras.layers.Dense(hidden_2_size, activation=my_activation)(d_drop_1)
    d_drop_2 = tf.keras.layers.Dropout(dropout_rate)(hidden_2_rev)
    hidden_1_rev = tf.keras.layers.Dense(hidden_1_size, activation=my_activation)(d_drop_2)
    output_data = tf.keras.layers.Dense(input_size, activation=my_activation)(hidden_1_rev)

    encoder = tf.keras.models.Model(input_data, [code_mean, code_log_var, code], name="encoder")
    decoder = tf.keras.models.Model(latent_inputs, output_data, name="decoder")
    return VAE(encoder, decoder), encoder , decoder



def decode_latent(decoder_f:str, latent_f:str, out_f:str):
    """
    Load the decoder model with keras, then decode the latent data and save it as csv
    """
    decoder = tf.keras.models.load_model(decoder_f)
    latent_data = pd.read_csv(latent_f, sep=";", header=0, dtype=np.float64)
    IDs=list(latent_data.columns)
    latent_data = latent_data.to_numpy()
    decoded_data = np.array(decoder(latent_data.transpose())).transpose()
    np.savetxt(out_f, decoded_data, delimiter=";", header=";".join(IDs))
    return "Decoded data written to " + out_f


def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_real_samples(data, n) :
    idx = np.random.choice(data.shape[0], n, replace=False)
    return data[idx, :], -np.ones((n,1))

def generate_fake_samples(g_model, latent, n) :
    return g_model(generate_latent_points(latent, n)), np.ones((n,1))

def wgan(input_data, ae_act="linear", latent_dim_size=10000):
    """
    Create unconstrained WGAN model, and return it and its two components 
    input_data is numerical, each row contains a sample, each column a variable
    ae_act is the activation function of the autoencoder, generator needs to use the same for data distribution purposes 
    """

    input_size = input_data.shape[1]

    # create critic 
    hidden_c_1_size = math.ceil(input_size/10)
    hidden_c_2_size = math.ceil(input_size/50)
    hidden_c_3_size = math.ceil(input_size/100)

    critic_input = tf.keras.layers.Input(shape=(input_size, ))
    c_hidden_1 = tf.keras.layers.Dense(hidden_c_1_size, activation='relu')(critic_input)
    c_drop_1 = tf.keras.layers.Dropout(0.4)(c_hidden_1)
    c_hidden_2 = tf.keras.layers.Dense(hidden_c_2_size, activation='relu')(c_drop_1)
    c_drop_2 = tf.keras.layers.Dropout(0.4)(c_hidden_2)
    c_hidden_3 = tf.keras.layers.Dense(hidden_c_3_size, activation='relu')(c_drop_2)
    critic_output = tf.keras.layers.Dense(1, activation='linear')(c_hidden_3)

    critic_model = tf.keras.models.Model(critic_input, critic_output)

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    critic_model.compile(optimizer=opt, loss=wasserstein_loss)
    critic_model.trainable = False


    # create generator
    hidden_g_1_size = math.ceil(input_size/2)
    hidden_g_2_size = math.ceil(input_size/5)
    hidden_g_3_size = math.ceil(input_size/10)

    generator_input = tf.keras.layers.Input(latent_dim_size)
    g_hidden_1 = tf.keras.layers.Dense(hidden_g_3_size, activation='relu')(generator_input)
    g_drop_1 = tf.keras.layers.Dropout(0.4)(g_hidden_1)
    g_hidden_2 = tf.keras.layers.Dense(hidden_g_2_size, activation='relu')(g_drop_1)
    g_drop_2 = tf.keras.layers.Dropout(0.4)(g_hidden_2)
    g_hidden_3 = tf.keras.layers.Dense(hidden_g_1_size, activation='relu')(g_drop_2)
    generator_output = tf.keras.layers.Dense(input_size, activation=ae_act)(g_hidden_3)

    generator_model = tf.keras.models.Model(generator_input, generator_output)


    wgan_model = tf.keras.models.Sequential()
    wgan_model.add(generator_model)
    wgan_model.add(critic_model)

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    wgan_model.compile(optimizer=opt, loss=wasserstein_loss)

    return wgan_model, critic_model, generator_model


def train_wgan(wgan_model, critic_model, generator_model, input_data, out_dir:str, n_steps:int, n_samples_batch=1000, n_crit_steps=5, save=100, check=10000, load="", latent_dim_size=10000, verbose=True):
    """
    Take WGAN and components created using wgan() and apply the custom WGAN training loop to them.
    save is the rate at which the model will be saved (in case of crashes etc), check is how often a separate checkpoint will be saved (to measure performance over the training process)
    load is the path to a directory containing "GAN.h5", the weights you want to load into the model
    """
    c_history = []
    g_history = []
    if load:
        wgan_model.load_weights(load+"GAN.h5")
        if os.path.isfile(load+"c_history.txt") :
            with open(load+"c_history.txt") as f:
                c_history = [float(line) for line in f]
        if os.path.isfile(load+"g_history.txt") :
            with open(load+"g_history.txt") as f:
                g_history = [float(line) for line in f]


    for i in range(n_steps):
        # train critic (more than generator)
        c_loss = 0
        critic_model.trainable = True
        for j in range(n_crit_steps):
            x_real, y_real = generate_real_samples(input_data, n_samples_batch)
            x_fake, y_fake = generate_fake_samples(generator_model, latent_dim_size, n_samples_batch)
            x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            c_loss += critic_model.train_on_batch(x, y)
        c_history.append(c_loss)

        # train generator
        critic_model.trainable = False
        x_gen = generate_latent_points(latent_dim_size, n_samples_batch)
        y_gen = -np.ones((n_samples_batch,1))
        g_loss = wgan_model.train_on_batch(x_gen, y_gen)
        g_history.append(g_loss)

        if verbose and (i+1)%10==0 :
            print(">"+str(i+1)+": c_loss= "+str(c_loss)+"; g_loss= "+str(g_loss))

        # saving models if needed
        if (len(c_history)+1)%save == 0:
            generator_model.save(out_dir+"generator")
            critic_model.save(out_dir+"critic")
            wgan_model.save(out_dir+"WGAN")
            wgan_model.save(out_dir+"WGAN/GAN.h5")
            with open(out_dir + "GAN_chr1/c_history.txt", 'w') as f:
                for item in c_history:
                    f.write("%s\n" % item)
            with open(out_dir + "GAN_chr1/g_history.txt", 'w') as f:
                for item in g_history:
                    f.write("%s\n" % item)
            if verbose:
                print("Model saved.")
                    
        if (len(c_history)+1)%check == 0:
            check_dir = out_dir + "checkpoint_" + str(len(c_history)+1) + "/"
            Path(check_dir).mkdir(parents=True, exist_ok=True)
            generator_model.save(check_dir+"generator")
            critic_model.save(check_dir+"critic")
            wgan_model.save(check_dir+"WGAN")
            wgan_model.save(check_dir+"WGAN/GAN.h5")
            with open(check_dir + "GAN_chr1/c_history.txt", 'w') as f:
                for item in c_history:
                    f.write("%s\n" % item)
            with open(check_dir + "GAN_chr1/g_history.txt", 'w') as f:
                for item in g_history:
                    f.write("%s\n" % item)
            if verbose:
                print("Checkpoint saved.")

    # save model before exiting in case n_steps % save != 0
    generator_model.save(out_dir+"generator")
    critic_model.save(out_dir+"critic")
    wgan_model.save(out_dir+"WGAN")
    wgan_model.save(out_dir+"WGAN/GAN.h5")
    with open(out_dir + "GAN_chr1/c_history.txt", 'w') as f:
        for item in c_history:
            f.write("%s\n" % item)
    with open(out_dir + "GAN_chr1/g_history.txt", 'w') as f:
        for item in g_history:
            f.write("%s\n" % item)
    if verbose:
        print("Model saved.")
