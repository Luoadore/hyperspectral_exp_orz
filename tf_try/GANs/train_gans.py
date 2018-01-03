# coding: utf-8
import tensorflow as tf
import math
import numpy as np
import scipy.io as sio
import preprocess_data as pd
import gans_config as gc


def next_batch(batch_size, num_step, data_set):
    """Return the next 'batch_size' examples from the data set.

    Args:
        batch_size: The batch size
        num_step: The step of iteration
        data_set: The data set

    Return:
        batch_data: Next batch size data
    """
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]
    #print('取出的数据大小： ' + str(end_index - start_index))
    return batch_data

def run_training(dataset):
    """
    Train GANs, reserve generator.

    Args:
        dataset: Data set of real world, index means the label and value means all the samples.
    Return:
        generator_container: The generator of every class.
    """

    # Build Network
    # Network input
    gen_input = tf.placeholder(tf.float32, shape = [None, gc.noise_dim], name = 'input_noise')
    disc_input = tf.placeholder(tf.float32, shape = [None, gc.data_dim], name = 'disc_input')

    # Build Generator Network
    gen_sample = gc.generator(gen_input)

    # Build 2 discriminator Network (one from noise input and one from generated samples)
    disc_real = gc.discriminator(disc_input)
    disc_fake = gc.discriminator(gen_sample)

    # Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # Opitimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate= gc.learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate= gc.learning_rate)

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [gc.weights['gen_hidden1'], gc.weights['gen_out'],
            gc.biases['gen_hidden1'], gc.biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [gc.weights['disc_hidden1'], gc.weights['disc_out'],
                gc.biases['disc_hidden1'], gc.biases['disc_out']]

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # Initialize the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        generator_container = []

        for index, eachclass in enumerate(dataset):
            print('The ' + str(index) + '-class generator training......')
            for i in range(gc.iterations):
                # Prepare Data
                # Get the next batch data (only images are needed, not labels)
                eachclass = np.array(eachclass)
                batch_x = next_batch(gc.batch_size, i, eachclass)
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[len(batch_x), gc.noise_dim])

                # Train
                feed_dict = {disc_input: batch_x, gen_input: z, }
                _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                        feed_dict=feed_dict)
                if i % 999 == 0 or i == 0:
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

            generator_container.append(gen_vars)
            print(generator_container.shape)

    print('Train done.')

    return generator_container

if __name__ == '__main__':
    data_file = 'F:\hsi_data\Kennedy Space Center (KSC)\KSCData.mat'
    label_file = 'F:\hsi_data\Kennedy Space Center (KSC)\KSCGt'
    hsi_data = pd.extract_data(data_file, label_file)
    generator_param = run_training(hsi_data)

    sio.savemat('F:\hsi_result\gan\gan_data.mat', {'gen_params': generator_param, 'data_set': hsi_data})
    print('Done.')