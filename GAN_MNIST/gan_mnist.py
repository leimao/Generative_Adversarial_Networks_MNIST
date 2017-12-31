'''
Handwritten Digit Generator
Generative Adversarial Network for MNIST Dataset

Version GAN

Lei Mao

Department of Computer Science
University of Chicago

Reference: 
https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN
'''

import os
import itertools
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
plt.switch_backend('agg')



def discriminator(x, dropout_kp):

    # Variable initializer
    weight_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.01) 
    bias_initializer = tf.constant_initializer(0.01)

    # FC hidden layer 1
    w1 = tf.get_variable(name = 'D_w1', 
        shape = [x.get_shape()[1], 1024], 
        initializer = weight_initializer)
    b1 = tf.get_variable(name = 'D_b1', 
        shape = [1024], 
        initializer = bias_initializer)
    h1 = tf.nn.bias_add(tf.matmul(x, w1), b1)
    h1 = tf.nn.relu(h1)
    h1 = tf.nn.dropout(x = h1, keep_prob = dropout_kp)

    # FC hidden layer 2
    w2 = tf.get_variable(name = 'D_w2', 
        shape = [h1.get_shape()[1], 512], 
        initializer = weight_initializer)
    b2 = tf.get_variable(name = 'D_b2', 
        shape = [512], 
        initializer = bias_initializer)
    h2 = tf.nn.bias_add(tf.matmul(h1, w2), b2)
    h2 = tf.nn.relu(h2)
    h2 = tf.nn.dropout(x = h2, keep_prob = dropout_kp)

    # FC hidden layer 3
    w3 = tf.get_variable(name = 'D_w3', 
        shape = [h2.get_shape()[1], 256], 
        initializer = weight_initializer)
    b3 = tf.get_variable(name = 'D_b3', 
        shape = [256], 
        initializer = bias_initializer)
    h3 = tf.nn.bias_add(tf.matmul(h2, w3), b3)
    h3 = tf.nn.relu(h3)
    h3 = tf.nn.dropout(x = h3, keep_prob = dropout_kp)

    # FC output layer
    w4 = tf.get_variable(name = 'D_w4', 
        shape = [h3.get_shape()[1], 1], 
        initializer = weight_initializer)
    b4 = tf.get_variable(name = 'D_b4', 
        shape = [1], 
        initializer = bias_initializer)
    output = tf.nn.bias_add(tf.matmul(h3, w4), b4)
    output = tf.sigmoid(output)

    return output


def generator(z):

    # Variable initializer
    weight_initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.01) 
    bias_initializer = tf.constant_initializer(0.01)

    # FC hidden layer 1
    w1 = tf.get_variable(name = 'G_w1', 
        shape = [z.get_shape()[1], 256], 
        initializer = weight_initializer)
    b1 = tf.get_variable(name = 'G_b1', 
        shape = [256], 
        initializer = bias_initializer)
    h1 = tf.nn.bias_add(tf.matmul(z, w1), b1)
    h1 = tf.nn.relu(h1)

    # FC hidden layer 2
    w2 = tf.get_variable(name = 'G_w2', 
        shape = [h1.get_shape()[1], 512], 
        initializer = weight_initializer)
    b2 = tf.get_variable(name = 'G_b2', 
        shape = [512], 
        initializer = bias_initializer)
    h2 = tf.nn.bias_add(tf.matmul(h1, w2), b2)
    h2 = tf.nn.relu(h2)

    # FC hidden layer 3
    w3 = tf.get_variable(name = 'G_w3', 
        shape = [h2.get_shape()[1], 1024], 
        initializer = weight_initializer)
    b3 = tf.get_variable(name = 'G_b3', 
        shape = [1024], 
        initializer = bias_initializer)
    h3 = tf.nn.bias_add(tf.matmul(h2, w3), b3)
    h3 = tf.nn.relu(h3)

    # FC output layer
    # MNIST dataset image size 28 x 28 = 784
    w4 = tf.get_variable(name = 'G_w4', 
        shape = [h3.get_shape()[1], 784], 
        initializer = weight_initializer)
    b4 = tf.get_variable(name = 'G_b4', 
        shape = [784], 
        initializer = bias_initializer)
    output = tf.nn.bias_add(tf.matmul(h3, w4), b4)
    # output = tf.sigmoid(output)

    output = tf.nn.tanh(output)

    return output


def save_images(images, label, path):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize = (5,5))

    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(images[k], (28,28)), cmap = 'gray')

    fig.text(0.5, 0.04, label, ha = 'center')
    plt.savefig(path)
    plt.close()


def save_outputs(sess, epoch, latent_code, path):
    if not os.path.exists(path):
        os.makedirs(path)
    images = sess.run(g_z, {z: latent_code})
    label =  "Epoch " + str(epoch)
    save_images(images = images, label = label, path = path + str(epoch) + '.png')


def save_model(sess):
    
    saver = tf.train.Saver()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    saver.save(sess, MODEL_DIR + MODEL_FILENAME)


def save_learning_curve(t, d_losses, g_losses, path):

    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(t, d_losses, 'r-o', label = 'Discriminator Loss')
    ax.plot(t, g_losses, 'b-o', label = 'Generator Loss')
    ax.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(path + 'learning_curve.png', format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()



BATCH_SIZE = 100
LEARNING_RATE = 0.0002
PRE_TRAIN_EPOCH = 0
PRE_TRAIN_TIMESTEP = 100
TRAIN_EPOCH = 100
DROPOUT_KEEP_PROB = 0.3
NP_RANDSEED = 0
TF_RANDSEED  = 0
LATENT_SIZE = 100


MODEL_DIR = 'model/'
MODEL_FILENAME = 'gan_mnist.ckpt'
SAVED_MODEL_DIR = 'pretrained_model/'
PRETRAINED_MODEL_FILENAME = 'pretrained_gan_mnist.ckpt'
IMAGES_DIR = 'generated_images/'
TRAINING_STATS_DIR =  'training_stats/'


# Set random seed for reproducibility
tf.set_random_seed(TF_RANDSEED)
np.random.seed(NP_RANDSEED)

# Read MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
mnist_size = mnist.train.images.shape[0]
# Normalization; range: -1 ~ 1
mnist_normalized = (mnist.train.images - 0.5) / 0.5

# Generator
with tf.variable_scope('G'):
    z= tf.placeholder(tf.float32, shape = [None, LATENT_SIZE])
    g_z = generator(z = z)

# Discriminator
# Share variables
with tf.variable_scope('D') as scope:

    dropout_kp = tf.placeholder(tf.float32, name = 'dropout_kp')
    x = tf.placeholder(tf.float32, shape = [None, 784])
    d_real = discriminator(x = x, dropout_kp = dropout_kp)
    scope.reuse_variables()
    d_fake = discriminator(x = g_z, dropout_kp = dropout_kp)

# Loss function
# The theoretical loss function should be:
# d_loss = tf.reduce_mean(-tf.log(d_real) - tf.log(1 - d_fake))
# g_loss = tf.reduce_mean(-tf.log(d_fake))
# To prevent the scenario when d_real, 1-d_fake, or d_fake is always 0
# We add a small number eps to the loss function to stablize the loss function
eps = 1e-2
d_loss = tf.reduce_mean(-tf.log(d_real + eps) - tf.log(1 - d_fake + eps))
g_loss = tf.reduce_mean(-tf.log(d_fake + eps))


# Trainable variables for generator and discriminator
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'D_' in var.name]
g_vars = [var for var in t_vars if 'G_' in var.name]

# Optimizer for generator and discriminator
d_optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(d_loss, var_list = d_vars)
g_optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(g_loss, var_list = g_vars)


# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Record losses
g_losses_train = list()
d_losses_train = list()
d_losses_pre_train = list()
i_epoch_train = list()
i_epoch_pre_train = list()

'''
# Pre-train discriminator
print("Start pre-training discriminator.")
for i in range(PRE_TRAIN_TIMESTEP):

    z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, LATENT_SIZE))
    x_batch = mnist.train.next_batch(BATCH_SIZE)[0]
    _, d_loss_pre_train = sess.run([d_optimizer, d_loss], {x: x_batch, z: z_batch, dropout_kp: DROPOUT_KEEP_PROB})

    if (i % 100 == 0):
        print("Pre-training Timestep: %d, D_Loss: %f" %(i, d_loss_pre_train))
        i_epoch_pre_train.append(i)
        d_losses_pre_train.append(d_loss_pre_train)
print("Pre-training discriminator finished.")
'''

# Fixed latent code to to track the training
z_fixed = np.random.normal(0, 1, size = (5*5, LATENT_SIZE))
save_outputs(sess = sess, epoch = 0, latent_code = z_fixed, path = IMAGES_DIR)

print("Start training discriminator and generator.")
# Train discriminator and generator simutaneously
for i in range(TRAIN_EPOCH):
    for j in range(mnist_size // BATCH_SIZE):
        # Train discriminator
        z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, LATENT_SIZE))
        # x_batch = mnist.train.next_batch(BATCH_SIZE)[0]
        x_batch = mnist_normalized[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        _, d_loss_train = sess.run([d_optimizer, d_loss], {x: x_batch, z: z_batch, dropout_kp: DROPOUT_KEEP_PROB})

        # Train generator
        # Test  here dropout = 0?
        z_batch = np.random.normal(0, 1, size = (BATCH_SIZE, LATENT_SIZE))
        _, g_loss_train = sess.run([g_optimizer, g_loss],  {z: z_batch, dropout_kp: DROPOUT_KEEP_PROB})

    print("Training Epoch: %d, D_Loss: %f, G_Loss: %f" %(i, d_loss_train, g_loss_train))

    i_epoch_train.append(i)
    d_losses_train.append(d_loss_train)
    g_losses_train.append(g_loss_train)

    save_outputs(sess = sess, epoch = i+1, latent_code = z_fixed, path = IMAGES_DIR)

    save_model(sess = sess)

print("Training discriminator and generator finished.")

sess.close()

save_learning_curve(t = list(range(TRAIN_EPOCH)), d_losses = d_losses_train, g_losses = g_losses_train, path = TRAINING_STATS_DIR)

images = list()
for i in range(TRAIN_EPOCH + 1):
    img_filename = IMAGES_DIR + str(i) + '.png'
    images.append(imageio.imread(img_filename))
imageio.mimsave(IMAGES_DIR + 'training_animation.gif', images, fps = 5)



'''
# Use pre-trained model to generate handwritten digits

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, MODEL_DIR + MODEL_FILENAME)
    z_batch = np.random.normal(0, 1, size = (25, LATENT_SIZE))
    images = sess.run(g_z, {z: z_batch})
    label =  "Generated Handwritten Digits"
    save_images(images = images, label = label, path = IMAGES_DIR + 'sample_handwritten_digits.png')
'''