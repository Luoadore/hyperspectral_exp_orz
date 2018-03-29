# coding: utf-8
"""Train the original network using a feed dictionary of one dataset PC and test using the other dataset PU."""

import tensorflow as tf
import time
import os.path
import deep_cnn as dc
import data_preprocess_pos as dp
import numpy as np
import math
import scipy.io as sio

#Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1_uints', 153, 'Number of uints in convolutional layer.')
flags.DEFINE_integer('conv1_kernel', 24 * 9, 'Length of kernel in conv1.')
flags.DEFINE_integer('conv1_stride', 9, 'Stride of conv1.')
flags.DEFINE_integer('conv2_uints', 32, 'Number of uints in convolutiona2 layer.')
flags.DEFINE_integer('conv3_uints', 64, 'Number of uints in convolutiona3 layer.')
flags.DEFINE_integer('conv4_uints', 64, 'Number of uints in convolutiona4 layer.')
flags.DEFINE_integer('fc1_uints', 1024, 'Number of uints in fully connection layer one.')
flags.DEFINE_integer('fc2_uints', 100, 'Number of uints in fully connection layer two.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('neighbor', 8, 'Neighbor of data option, including 0, 4 and 8.')
flags.DEFINE_integer('ratio', 100, 'Ratio of the train set in the whole data.')
flags.DEFINE_string('pc_data_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCData.mat', 'Directory of data file.')
flags.DEFINE_string('pc_label_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCGt.mat', 'Directory of label file.')
flags.DEFINE_string('pu_data_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCData.mat', 'Directory of data file.')
flags.DEFINE_string('pu_label_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCGt.mat', 'Directory of label file.')
flags.DEFINE_string('train_dir', 'F:\hsi_result\deep\KSC', 'The train result save file.')

def placeholder_inputs(batch_size):
    """Generate palcehold variables to represent the input tensors.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        data_placeholder: Data placeholder
        labels_placeholder: Labels placeholder
    """
    data_placeholder = tf.placeholder(tf.float32, shape = (None, dc.BANDS_SIZE * FLAGS.conv1_stride))
    label_placeholder = tf.placeholder(tf.float32, shape = (None, dc.NUM_CLASSES))

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
    #if remainder == 0:
    #data_set, label_set = dp.shuffling2(data_set, label_set)

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]
    batch_label = label_set[start_index : end_index]
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

    Return:
        precision: The accuray of the test data set
        prediction: The prediction of the data set

    """

    #And run one apoch of eval
    true_count = 0
    num_examples = len(label_set)
    prediction = []
    steps_per_epoch = math.ceil(num_examples / FLAGS.batch_size)
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(step, data_set, label_set, data_placeholder, label_placeholder)
        true_count += sess.run(eval_correct, feed_dict = feed_dict)
        softmax_value = sess.run(softmax, feed_dict = feed_dict)
        prediction.extend(np.argmax(softmax_value, axis = 1))
    precision = true_count / num_examples
    print('Num examples: %d Num correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))

    return precision, prediction

def run_training():
    """Train net model."""
    # Get the sets of data
    # First method to get data
    pc_data_set, pc_data_pos = dp.extract_data(FLAGS.pc_data_dir, FLAGS.pc_label_dir, FLAGS.neighbor)
    pu_data_set, pu_data_pos = dp.extract_data(FLAGS.pu_data_dir, FLAGS.pu_label_dir, FLAGS.neighbor)
    pc_train_data, pc_train_label, pc_train_pos, _, _, _ = dp.load_data(pc_data_set, pc_data_pos, FLAGS.ratio)
    _, _, _, pu_test_data, pu_test_label, pu_test_pos = dp.load_data(pu_data_set, pu_data_pos, FLAGS.ratio)
    print('train label length: ' + str(len(pc_train_label)) + ', train data length: ' + str(len(pc_train_data)))
    print('test label length:' + str(len(pu_test_label)) + ', test data length: ' + str(len(pu_test_data)))
    #transform int label into one-hot values
    print('train: ')
    train_label = dp.onehot_label(pc_train_label, dc.NUM_CLASSES)
    print('test: ')
    test_label = dp.onehot_label(pu_test_label, dc.NUM_CLASSES)

    test_loss_steps = []
    test_acc_steps = []
    test_steps = []
    train_prediction = []
    test_prediction = []

    with tf.Graph().as_default():
        #Generate placeholders
        data_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)
        #Build a Graph that computes predictions from the inference model
        softmax = dc.inference(data_placeholder, FLAGS.conv1_uints, FLAGS.conv1_kernel, FLAGS.conv1_stride,
                               FLAGS.conv2_uints, FLAGS.conv3_uints, FLAGS.conv4_uints, FLAGS.fc1_uints, FLAGS.fc2_uints)
        #Add to the Graph the Ops for loss calculation
        loss_entroy = dc.loss(softmax, label_placeholder)
        #Add to the Graph the Ops that calculate and apply gradients
        train_op = dc.training(loss_entroy, FLAGS.learning_rate)
        #Add thp Op to compare the loss to the labels
        pred, correct = dc.acc(softmax, label_placeholder)
        tf.summary.scalar('accuracy', pred)
        #Build the summary operation based on the TF collection of Summaries
        summary_op = tf.summary.merge_all()
        #Add the variable initalizer Op
        init = tf.global_variables_initializer()
        #Create a saver for writing traing checkpoints
        saver = tf.train.Saver()

        #Create a session for training
        sess = tf.Session()
        #Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        #Run the Op to initialize the variables
        sess.run(init)

        time_sum = 0

        #Start the training loop
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(step, pc_train_data, pc_train_label, data_placeholder, label_placeholder)

            #Run one step of the model
            _, loss_value, softmax_value = sess.run([train_op, loss_entroy, softmax], feed_dict = feed_dict)

            duration = time.time() - start_time

            #Write the summaries and print an overview farily often
            if step % 100 == 0:
                #Print status to stdout
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                time_sum = time_sum +  duration
                #Update the events file
                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            #Save a checkpoint and evaluate the model periodically
            if(step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step = step)
                #data_train_placeholder, label_train_placeholder = placeholder_inputs(len(train_label))
                #data_test_placeholder, label_test_placeholder = placeholder_inputs(len(test_label))
                #feed_dict_test = {data_test_placeholder: test_data, label_test_placeholder: test_label,}
                feed_dict_test = fill_feed_dict(step, test_data, test_label, data_placeholder, label_placeholder)
                #Evaluate against the data set
                print('Training Data Eval:')
                _, train_prediction = do_eval(sess, correct, data_placeholder, label_placeholder, train_data, train_label, softmax)
                print('Test Data Eval:')
                test_acc, test_prediction = do_eval(sess, correct, data_placeholder, label_placeholder, test_data, test_label, softmax)
                test_loss = sess.run(loss_entroy, feed_dict = feed_dict_test)
                test_steps.append(step)
                test_acc_steps.append(test_acc)
                test_loss_steps.append(test_loss)

    print('test loss: ' + str(test_loss_steps))
    print('test acc: ' + str(test_acc_steps))
    print('test step: ' + str(test_steps))
    print('总用时： ' + str(time_sum))

    sio.savemat(FLAGS.train_dir + '\data.mat', {'train_data': train_data, 'train_label': dp.decode_onehot_label(train_label, dc.NUM_CLASSES), 'train_pos': train_pos,
                                                'test_data': test_data, 'test_label': dp.decode_onehot_label(test_label, dc.NUM_CLASSES), 'test_pos': test_pos,
                                                'test_loss': test_loss_steps, 'test_acc': test_acc_steps, 'test_step': test_steps,
                                                'train_prediction': train_prediction, 'test_prediction': test_prediction})

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()