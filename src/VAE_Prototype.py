# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
database = input_data.read_data_sets('/content/data', one_hot = True)

# Parameters for neural networks
learning_param = 0.001
epochs = 30000
batch_size = 32

# Network parameters
image_dimension = 784 # 28 * 28
neural_network_dimension = 512

# Change this value for latent space effect on visualizations of reconstructed data
latent_variable_dimension = 2

def xavier (in_shape):
    val = tf.random_normal(shape = in_shape , stddev = 1./tf.sqrt(in_shape[0]/2.)) # We want to initialize with values that are neither too small nor too large 
    return val

# Weight and bias dictionaries
Weight = {
    "weight_matrix_encoder_hidden" : tf.Variable( xavier([image_dimension, neural_network_dimension])),
    "weight_mean_hidden" : tf.Variable( xavier([neural_network_dimension, latent_variable_dimension])),
    "weight_std_hidden" : tf.Variable( xavier([neural_network_dimension, latent_variable_dimension])),
    "weight_matrix_decoder_hidden" : tf.Variable( xavier([latent_variable_dimension, neural_network_dimension])),
    "weight_decoder" : tf.Variable( xavier([neural_network_dimension, image_dimension]))
}
Bias = {
    "bias_matrix_encoder_hidden" : tf.Variable( xavier([neural_network_dimension])),
    "bias_mean_hidden" : tf.Variable( xavier([latent_variable_dimension])),
    "bias_std_hidden" : tf.Variable( xavier([latent_variable_dimension])),
    "bias_matrix_decoder_hidden" : tf.Variable( xavier([neural_network_dimension])),
    "bias_decoder" : tf.Variable( xavier([image_dimension]))
}

# Building the Variational Autoencoder (a computational graph)

# Encoder Section

image_X = tf.placeholder(tf.float32, shape = [None, image_dimension])

Encoder_layer = tf.add(tf.matmul(image_X, Weight["weight_matrix_encoder_hidden"]), Bias["bias_matrix_encoder_hidden"])
Encoder_layer = tf.nn.tanh(Encoder_layer)

Mean_layer = tf.add(tf.matmul(Encoder_layer, Weight["weight_mean_hidden"]), Bias["bias_mean_hidden"])
Standard_deviation_layer = tf.add( tf.matmul(Encoder_layer, Weight["weight_std_hidden"]), Bias["bias_std_hidden"])

# Reparametrization Trick
epsilon = tf.random_normal(tf.shape(Standard_deviation_layer), dtype=tf.float32, mean = 0.0, stddev=1.0)
latent_layer = Mean_layer + tf.exp(0.5 * Standard_deviation_layer) * epsilon

# Decoder Section
Decoder_hidden = tf.add(tf.matmul(latent_layer, Weight["weight_matrix_decoder_hidden"]), Bias["bias_matrix_decoder_hidden"])
Decoder_hidden = tf.nn.tanh(Decoder_hidden)

Decoder_output_layer = tf.add(tf.matmul(Decoder_hidden, Weight["weight_decoder"]), Bias["bias_decoder"])
Decoder_output_layer = tf.nn.sigmoid(Decoder_output_layer)

# Defining Variational Autoencoder Loss

def loss_function(original_image, reconstructed_image):

    # Reconstruction loss
    data_fidelity_loss = original_image * tf.log(1e-10 + reconstructed_image) + (1 - original_image) * tf.log(1e-10 + 1 - reconstructed_image)
    data_fidelity_loss = -tf.reduce_sum(data_fidelity_loss, 1)

    # KL Divergence 
    KL_div_loss = 1 + Standard_deviation_layer - tf.square(Mean_layer) - tf.exp(Standard_deviation_layer)
    KL_div_loss = -0.5 * tf.reduce_sum(KL_div_loss, 1)

    # Depending on which loss you have to take, select values of alpha and beta
    # If only KL loss is to be evaluated, put alpha = 0. If fidelity loss, put beta = 0. For VAE, both should be 1.
    alpha = 1
    beta = 1
    network_loss = tf.reduce_mean(alpha * data_fidelity_loss + beta * KL_div_loss)
    return network_loss


loss_value = loss_function(image_X , Decoder_output_layer)
optimizer = tf.train.RMSPropOptimizer(learning_param).minimize(loss_value)

# Initialize all the variables
init = tf.global_variables_initializer()

# Executing the computational graph

# Start the session
sess = tf.Session()

# Run the initializer
sess.run(init)

for i in range(epochs):
    x_batch, _ = database.train.next_batch(batch_size)
    _, loss = sess.run([optimizer, loss_value], feed_dict = {image_X : x_batch})
    if i % 5000 == 0:
        print("Loss is {0} at iteration {1}".format(loss, i))


# Testing Phase

# Noise Input Handle
noise_X = tf.placeholder(tf.float32, shape = [None, latent_variable_dimension])

# Rebuild the decoder to create output image from noise

# Decoder Section (Repeat)
Decoder_hidden = tf.add(tf.matmul(noise_X, Weight["weight_matrix_decoder_hidden"]), Bias["bias_matrix_decoder_hidden"])
Decoder_hidden = tf.nn.tanh(Decoder_hidden)

Decoder_output_layer = tf.add(tf.matmul(Decoder_hidden, Weight["weight_decoder"]), Bias["bias_decoder"])
Decoder_output_layer = tf.nn.sigmoid(Decoder_output_layer)

# Output visualizations

n = 20
x_limit = np.linspace(-2, 2, n)
y_limit = np.linspace(-2, 2, n)

empty_image = np.empty((28*n, 28*n))

for i, zi in enumerate(x_limit):
    for j, pi in enumerate(y_limit):
        generated_latent_layer = np.array([[zi, pi]] * batch_size)
        # generated_latent_layer = np.random.normal(0, 1, size=[batch_size, latent_variable_dimension])
        generated_image = sess.run(Decoder_output_layer, feed_dict= {noise_X : generated_latent_layer})
        empty_image[ (n-i-1) * 28 : (n-i) * 28, j * 28 : (j+1)*28] = generated_image[0].reshape(28,28)

plt.figure(figsize = (8,10))

X,Y = np.meshgrid(x_limit, y_limit)
plt.imshow(empty_image, origin = "upper", cmap="gray")
plt.grid(False)
plt.show()

x_sample, y_sample = database.test.next_batch(batch_size + 15000)
print(x_sample.shape)

# Run of the latent layer. This graph is only possible if latent dimensions are 2. Graph shows the gaussian distribution profile of the latent vectors as discussed in theory
interim = sess.run(latent_layer, feed_dict = {image_X : x_sample})
print(interim.shape)

colors = np.argmax(y_sample, 1)

plt.figure(figsize = (8, 6))
plt.scatter(interim[:, 0], interim[:, 1], c = colors, cmap = 'viridis')
plt.colorbar()
plt.grid();

sess.close()
