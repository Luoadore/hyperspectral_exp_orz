# coding: utf-8
"""Train the original network using a feed dictionary."""

import tensorflow as tf
import time
import os.path
import original_cnn as oc
import data_preprocessing as dp

#Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1_uints', 20, 'Number of uints in convolutional layer.')
flags.DEFINE_integer('conv1_kernel', 24, 'Length of kernel in conv1.')
flags.DEFINE_integer('conv1_stride', 1, 'Stride of conv1.')
flags.DEFINE_integer('fc_uints', 100, 'Number of uints in fully connection layer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('neighbor', 0, 'Neighbor of data option, including 0, 4 and 8.')
flags.DEFINE_integer('ratio', 80, 'Ratio of the train set in the whole data.')
flags.DEFINE_string('data_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCData.mat', 'Directory of data file.')
flags.DEFINE_string('label_dir', 'F:\hsi_data\Kennedy Space Center (KSC)\KSCGt.mat', 'Directory of label file.')
flags.DEFINE_string('train_dir', 'F:\hsi_result', 'The train result save file.')

def placeholder_inputs(batch_size):
    """Generate palcehold variables to represent the input tensors.
    
    Args:
        batch_size: The batch size will be baked into both placeholders.
    
    Returns:
        data_placeholder: Data placeholder
        labels_placeholder: Labels placeholder
    """
    data_placeholder = tf.placeholder(tf.float32, shape = (batch_size, oc.BANDS_SIZE))
    label_placeholder = tf.placeholder(tf.int32, shape = (batch_size, oc.NUM_CLASSES))
    
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
    num_per_epoch = data_size // batch_size
    remainder = num_step % num_per_epoch
    if remainder == 0:
        data_set, label_set = dp.shuffling2(data_set, label_set)
    
    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index, end_index]
    batch_label = label_set[start_index, end_index]
    yield batch_data, batch_label
    
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
    
def do_eval(sess, eval_correct, data_placeholder, label_placeholder, data_set, label_set):
    """Runs one evaluation against the full epoch of data.
    
    Args:
        sess: The session in which the model has been trained
        eval_correct: The tensor that returns the number of correct predictions
        data_placeholder: The data placeholder
        label_placeholder: The label placeholder
        data_set: The set of data to evaluate, from dp.load_data()
        label_set: The set of label to evaluate, from dp.load_data()
        
    """
    
    #And run one apoch of eval
    true_count = 0
    num_examples = len(label_set)
    steps_per_epoch = num_examples // FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, label_set, data_placeholder, label_placeholder)
        true_count += sess.run(eval_correct, feed_dict = feed_dict)
    precision = true_count / num_examples
    print('Num examples: %d Num correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    
def run_training():
    """Train net model."""
    #Get the sets of data
    data_set = dp.extract_data(FLAGS.data_dir, FLAGS.label_dir, FLAGS.neighbor)
    train_data, train_label, test_data, test_label = dp.load_data(data_set, FLAGS.ratio)
    #transform int label into one-hot values
    train_label = onehot_label(train_label)
    test_label = onehot_label(test_label)
    
    with tf.Graph().as_default():
        #Generate placeholders
        data_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)
        #Build a Graph that computes predictions from the inference model
        softmax = oc.inference(data_placeholder, FLAGS.conv1_uints, FLAGS.conv1_kernel, FLAGS.conv1_stride, FLAGS.fc_uints)
        #Add to the Graph the Ops for loss calculation
        loss_entroy = oc.loss(softmax, label_placeholder)
        #Add to the Graph the Ops that calculate and apply gradients
        train_op = oc.training(loss_entroy, FLAGS.learning_rate)
        #Add thp Op to compare the loss to the labels
        correct = oc.acc(softmax, label_placeholder)
        #Build the summary operation based on the TF collection of Summaries
        summary_op = tf.merge_all_summaries()
        #Add the variable initalizer Op
        init = tf.initialize_all_variables()
        #Create a saver for writing traing checkpoints
        saver = tf.train.Saver()
        
        #Create a session for training
        sess = tf.Session()
        #Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        
        #Run the Op to initialize the variables
        sess.run(init)
        
        #Start the training loop
        for step in range(FLAGS.max_step):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(step, train_data, train_label, data_placeholder, label_placeholder)
            
            #Run one step of the model
            _, loss_value = sess.run([train_op, loss_entroy], feed_dict = feed_dict)
            
            duration = time.time() - start_time

            #Write the summaries and print an overview farily often
            if step % 100 == 0:
                #Print status to stdout
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                #Update the events file
                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                
            #Save a checkpoint and evaluate the model periodically
            if(step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step = step)
                #Evaluate against the data set
                print('Training Data Eval:')
                do_eval(sess, correct, data_placeholder, label_placeholder, train_data, train_label)
                print('Test Data Eval:')
                do_eval(sess, correct, data_placeholder, label_placeholder, test_data, test_label)
                
def main(_):
    run_training()
    

if __name__ == '__main__':
    tf.app.run()