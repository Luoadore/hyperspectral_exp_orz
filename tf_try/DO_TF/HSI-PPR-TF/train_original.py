# coding: utf-8
"""Train the original network using a feed dictionary."""

import tensorflow as tf
import time
import os.path
import original_cnn as oc
import cnn_deconv as cd
import data_preprocess_pos as dp
import numpy as np
import math
import scipy.io as sio
from data_config import *

# Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1_uints', 20, 'Number of uints in convolutional layer.')
flags.DEFINE_integer('conv1_kernel', 19 * 1, 'Length of kernel in conv1.')
flags.DEFINE_integer('conv1_stride', 9, 'Stride of conv1.')
flags.DEFINE_integer('fc_uints', 100, 'Number of uints in fully connection layer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('neighbor', 8, 'Neighbor of data option, including 0, 4 and 8.')
flags.DEFINE_integer('ratio', 80, 'Ratio of the train set in the whole data.')
flags.DEFINE_string('data_dir', '/media/luo/result/hsi_transfer/ksc/', 'Directory of data file.')
flags.DEFINE_string('data_name', 'data_normalize.mat', 'Name of data file.')
flags.DEFINE_string('label_dir', '/media/luo/result/hsi/KSC/KSCGt.mat', 'Directory of label file.')
flags.DEFINE_string('train_dir', '/media/luo/result/hsi_transfer/ksc/results/0316/', 'The train result save file.')
flags.DEFINE_string('PPR_block', 'cd', 'ppr block method of model.')
flags.DEFINE_boolean('is_training', True, 'Whether the parameters of source net training.')
flags.DEFINE_string('save_name', 'ksc_0517_s20', 'Save name of dataset.')
flags.DEFINE_string('S_or_T', 's', 'Source or target for training.')
flags.DEFINE_string('train_ratio', '20', 'source data ratio for training.')
flags.DEFINE_string('test_ratio', '80', 'source data ratio for testing.')

def placeholder_inputs(batch_size):
    """Generate palcehold variables to represent the input tensors.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        data_placeholder: Data placeholder
        labels_placeholder: Labels placeholder
    """
    data_placeholder = tf.placeholder(tf.float32,
                                      shape=(None, oc.BANDS_SIZE * FLAGS.conv1_stride))  # 记得这里修改BAND_SIZE的值, * 1, 5, 9
    label_placeholder = tf.placeholder(tf.float32, shape=(None, oc.NUM_CLASSES))

    return data_placeholder, label_placeholder


def next_batch(batch_size, num_step, data_set, label_set):
    """Return the next 'batch_size' examples from the data set.

    Args:
        batch_size: The batch size
        num_step: The step of iteration
        data_set: The data set
        label_set: The correspoding label set

    Return:
        batch_data: Next batch size data
        batch_label: Next batch size correspoding label
    """
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch
    # if remainder == 0:
    # data_set, label_set = dp.shuffling2(data_set, label_set)

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index: end_index]
    batch_label = label_set[start_index: end_index]
    # print('取出的数据大小： ' + str(end_index - start_index))
    return batch_data, batch_label


def fill_feed_dict(num_step, data_set, label_set, data_pl, label_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
        num_step: The step of iteration
        data_set: The set of data, from dp.load_data()
        label_set: The set of label, from dp.load_data()
        data_pl: The data placeholder, from placeholder_inputs()
        label_pl: The label placeholder, from placeholder_inputs()

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    data_feed, label_feed = next_batch(FLAGS.batch_size, num_step, data_set, label_set)
    feed_dict = {
        data_pl: data_feed,
        label_pl: label_feed,
    }

    return feed_dict


def do_eval(sess, eval_correct, data_placeholder, label_placeholder, data_set, label_set, softmax):
    """Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained
        eval_correct: The tensor that returns the number of correct predictions
        data_placeholder: The data placeholder
        label_placeholder: The label placeholder
        data_set: The set of data to evaluate, from dp.load_data()
        label_set: The set of label to evaluate, from dp.load_data()
        softmax: Softmax layer output, use for calculating each sample's predicting label, from oc.inference()

    Return:
        precision: The accuray of the data set
        prediction: The prediction of the data set

    """

    # And run one apoch of eval
    true_count = 0
    num_examples = len(label_set)
    predicition = []
    steps_per_epoch = math.ceil(num_examples / FLAGS.batch_size)
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(step, data_set, label_set, data_placeholder, label_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        softmax_value = sess.run(softmax, feed_dict=feed_dict)
        predicition.extend(np.argmax(softmax_value, axis=1))
    precision = true_count / num_examples
    print('Num examples: %d Num correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))

    return precision, predicition


def get_feature(sess, data_placeholder, label_placeholder, data_set, label_set, conv1):
    """Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained
        data_placeholder: The data placeholder
        label_placeholder: The label placeholder
        data_set: The set of data to evaluate, from dp.load_data()
        label_set: The set of label to evaluate, from dp.load_data()
        fc: fc layer output, from oc.inference()

    Return:
        fc_values: The first full-connection layer's output

    """

    # And run one apoch of data
    num_examples = len(label_set)
    feature_values = []
    steps_per_epoch = math.ceil(num_examples / FLAGS.batch_size)
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(step, data_set, label_set, data_placeholder, label_placeholder)
        fea_v = sess.run(conv1, feed_dict=feed_dict)
        feature_values.extend(fea_v)

    print('All the feature values extract done.')

    return feature_values


def run_training():
    # Train net model.
    # Second method to get data
    data_set = ksc
    num_class = data_set['num_classes']
    bands = data_set['bands']
    data = sio.loadmat(data_set['data_dir'])
    train_data = data[FLAGS.S_or_T + '_' + FLAGS.train_ratio]
    train_label = np.transpose(data[FLAGS.S_or_T + '_' + FLAGS.train_ratio + '_l'])
    source_test_data = data[FLAGS.S_or_T + '_'+ FLAGS.test_ratio]
    source_test_label = np.transpose(data[FLAGS.S_or_T + '_' + FLAGS.test_ratio +'_l'])
    #target_test_data = data['target_test_data']
    #target_test_label = np.transpose(data['target_test_label'])
    train_label = dp.onehot_label(train_label, num_class)
    source_test_label = dp.onehot_label(source_test_label, num_class)
    #target_test_label = dp.onehot_label(target_test_label, num_class)
    # print(np.shape(train_data))
    # print(np.shape(train_label))
    # print(np.shape(test_data))
    # print(np.shape(test_label))

    train_acc_steps = []
    test_loss_steps = []
    s_test_acc_steps = []
    t_test_acc_steps = []
    test_steps = []
    train_prediction = []
    s_test_prediction = []
    t_test_prediction = []

    with tf.Graph().as_default():
        # Generate placeholders
        data_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)
        # Build a Graph that computes predictions from the inference model
        conv1_kernel = data_set['conv1_kernel']
        softmax = cd.inference(data_placeholder, conv1_kernel, FLAGS.conv1_stride,
                               FLAGS.fc_uints, num_class, bands, FLAGS.is_training)
        # Add to the Graph the Ops for loss calculation
        loss_entroy = cd.loss(softmax, label_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients
        train_op = cd.training(loss_entroy, FLAGS.learning_rate)
        # Add thp Op to compare the loss to the labels
        pred, correct = cd.acc(softmax, label_placeholder)
        tf.summary.scalar('accuracy', pred)
        # Build the summary operation based on the TF collection of Summaries
        summary_op = tf.summary.merge_all()
        # Add the variable initalizer Op
        init = tf.global_variables_initializer()
        # Create a saver for writing traing checkpoints
        saver = tf.train.Saver()

        # Create a session for training
        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph
        train_dir = data_set['train_dir'] + FLAGS.save_name
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # Run the Op to initialize the variables
        sess.run(init)

        time_sum = 0

        # Start the training loop
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(step, train_data, train_label, data_placeholder, label_placeholder)

            # Run one step of the model
            _, loss_value = sess.run([train_op, loss_entroy], feed_dict=feed_dict)


            duration = time.time() - start_time

            # Write the summaries and print an overview farily often
            if step % 100 == 0:
                # Print status to stdout
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                time_sum = time_sum + duration
                # Update the events file
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically
            if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                train_dir = data_set['train_dir'] + FLAGS.save_name
                checkpoint_file = os.path.join(train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)
                # data_train_placeholder, label_train_placeholder = placeholder_inputs(len(train_label))
                # data_test_placeholder, label_test_placeholder = placeholder_inputs(len(test_label))
                # feed_dict_test = {data_test_placeholder: test_data, label_test_placeholder: test_label,}
                # feed_dict_test = fill_feed_dict(step, test_data, test_label, data_placeholder, label_placeholder)
                # Evaluate against the data set
                print('Training Source Data Eval:')
                train_acc, train_prediction = do_eval(sess, correct, data_placeholder, label_placeholder, train_data,
                                                      train_label, softmax)
                train_acc_steps.append(train_acc)
                print('Source test Data Eval:')
                s_test_acc, s_test_prediction = do_eval(sess, correct, data_placeholder, label_placeholder, source_test_data,
                                                    source_test_label, softmax)
                #print('Target Test Data Eval:')
                #t_test_acc, t_test_prediction = do_eval(sess, correct, data_placeholder, label_placeholder, target_test_data,
                #                                    target_test_label, softmax)
                # test_loss = sess.run(loss_entroy, feed_dict=feed_dict_test)
                test_steps.append(step)
                s_test_acc_steps.append(s_test_acc)
                #t_test_acc_steps.append(t_test_acc)
                # test_loss_steps.append(test_loss)

                # train_fea_values = get_feature(sess, data_placeholder, label_placeholder, train_data, train_label, fc)
                # test_fea_values = get_feature(sess, data_placeholder, label_placeholder, test_data, test_label, fc)


    sio.savemat(data_set['train_dir'] + FLAGS.save_name + '/original_data.mat', {
    # 'train_data': train_data, 'train_label': dp.decode_onehot_label(train_label, oc.NUM_CLASSES), 'train_pos': train_pos,
        # 'test_data': test_data, 'test_label': dp.decode_onehot_label(test_label, oc.NUM_CLASSES), 'test_pos': test_pos,
        # 'test_loss': test_loss_steps,
        'source_test_acc': s_test_acc_steps, 'test_step': test_steps,
        #'target_test_acc': t_test_acc_steps,
        'train_acc': train_acc_steps,  # 'train_fea': train_fea_values, 'test_fea': test_fea_values,
        'train_prediction': train_prediction,
        'source_test_prediction': s_test_prediction#, 'target_test_prediction': t_test_prediction
        })


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()