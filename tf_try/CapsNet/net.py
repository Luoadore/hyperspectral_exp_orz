# -*- coding: utf-8 -*-
import os
import time

import tensorflow as tf
from tensorflow.python.framework import ops

from data_file import DataManager
from caps import get_graph
from params import param


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def initializer_variable(sess, saver, model_dir):
    with sess.as_default():
        check = tf.train.get_checkpoint_state(model_dir)
        if check and check.model_checkpoint_path:
            saver.restore(sess, check.model_checkpoint_path)
            print('checkpoint file has found')
        else:
            tf.global_variables_initializer().run()
            print('No checkpoint file found')
        step = tf.train.get_global_step().eval()

    return step


def train_model(train_dir, model_dir):
    graph = tf.Graph()
    element = get_graph(graph)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        step = initializer_variable(sess, saver, model_dir)
        summary_writer = tf.summary.FileWriter(train_dir, graph)

        def do_train():
            _, loss, accuracy, margin, summary = sess.run(
                [element['train'], element['loss'], element['accuracy'], element['margin'], element['summary']],
                feed_dict={element['data']: data, element['label']: label})
            return loss, accuracy, margin, summary

        def save_model():
            saver.save(sess, os.path.join(model_dir, 'model'), global_step=step)

        datas = DataManager(
            file=os.path.join(param.pre, param.hsi.data_file),
            class_number=param.hsi.class_number)
        train_set = datas.train.next_batch(param.batch_size, True)
        start_time = time.time()
        while step < param.generations:
            data, label = next(train_set)
            loss, accuracy, margin, summary = do_train()
            step = tf.train.get_global_step().eval()
            summary_writer.add_summary(summary, step)

            if step % param.output_every == 0:
                next_time = time.time()
                print('accuracy: {:04f}.'.format(accuracy))
                print('total loss:{:04f}, margin loss: {:04f}.'.format(loss, margin))
                print('-- have train {}s step using {} seconds. --'
                      .format(step, next_time - start_time))

            if step % param.save_every == 0:
                save_model()
                print('model saved in step {}'.format(step))

        summary_writer.close()


def test_model(test_dir, model_dir):
    graph = tf.Graph()
    element = get_graph(graph)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        s = initializer_variable(sess, saver, model_dir)
        if s == 0:
            print(' -- Having no model to test. -- ')
            return

        def do_summary(data, label):
            summary_str = sess.run(
                element['summary'],
                feed_dict={element['data']: data, element['label']: label})
            return summary_str

        def test_set(data_set, step, name):
            summary_writer = tf.summary.FileWriter(os.path.join(test_dir, name), graph)
            i = 0
            for data, label in data_set:
                summary_str = do_summary(data, label)
                summary_writer.add_summary(summary_str, i)
                if i >= step:
                    break
                i += 1
            summary_writer.close()

        datas = DataManager(
            file=os.path.join(param.pre, param.hsi.data_file),
            class_number=param.hsi.class_number)

        test_data = datas.test.next_batch(param.batch_size, False)
        test_step = datas.test.length // param.batch_size
        test_set(test_data, test_step, 'test')

        train_data = datas.train.next_batch(param.batch_size, False)
        train_step = datas.train.length // param.batch_size
        test_set(train_data, train_step, 'train')


def pred_model(test_dir, model_dir):
    graph = tf.Graph()
    element = get_graph(graph)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        s = initializer_variable(sess, saver, model_dir)
        if s == 0:
            print(' -- Having no model to pred. -- ')
            return

        def do_pred(data, label):
            pred, real = sess.run(
                element['prediction'],
                feed_dict={element['data']: data, element['label']: label})
            return pred, real

        def write_data(name, data):
            file = os.path.join(test_dir, name+'.csv')
            pred, real = data
            if isinstance(pred, str):
                pout = open(file, encoding='utf-8', mode='a')
                pout.writelines([pred, ',', real, '\n'])
            else:
                with open(file, encoding='utf-8', mode='a') as pout:
                    for p, r in zip(pred, real):
                        pout.writelines([str(p), ',', str(r), '\n'])

        def pred_set(data_set, step, name):
            write_data(name, ['prediction', 'label'])
            i = 0
            for data, label in data_set:
                pred, real = do_pred(data, label)
                write_data(name, [pred, real])
                if i >= step:
                    break
                i += 1

        datas = DataManager(
            file=os.path.join(param.pre, param.hsi.data_file),
            class_number=param.hsi.class_number)

        test_data = datas.test.next_batch(param.batch_size, False)
        test_step = datas.test.length // param.batch_size
        pred_set(test_data, test_step, 'test')

        train_data = datas.train.next_batch(param.batch_size, False)
        train_step = datas.train.length // param.batch_size
        pred_set(train_data, train_step, 'train')


def make_path(pre, path, i):
    k = os.path.join(pre, path, str(i))
    if not os.path.exists(k):
        os.makedirs(k)
    return k


def main(pre, after):
    train_dir = 'train'
    model_dir = 'model'
    test_dir = 'test'
    pred_dir = 'pred'
    for i in after:
        training_dir = make_path(pre, train_dir, i)
        modeling_dir = make_path(pre, model_dir, i)
        testing_dir = make_path(pre, test_dir, i)
        predict_dir = make_path(pre, pred_dir, i)
        ops.reset_default_graph()
        train_model(training_dir, modeling_dir)
        # time.sleep(10)
        # ops.reset_default_graph()
        # test_model(testing_dir, modeling_dir)
        # time.sleep(10)
        # pred_model(predict_dir, modeling_dir)


if __name__ == '__main__':
    param.hsi = param.data['pu']
    param.vector = 8
    param.channel = 12
    param.length = 32
    pre = 'pu' + '--' + 'vector(' + str(param.vector) + ')channel(' + str(param.channel) + ')length(' + str(param.length) + ')'
    main(pre, [1, 2, 3])
