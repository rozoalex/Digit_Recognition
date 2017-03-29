from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from os import listdir
from os.path import isfile, join
import os
import skimage.io
import skimage.transform
import skimage.util
import numpy
import PIL
numpy.set_printoptions(threshold=numpy.nan)

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def dilate(img, radius):
    from skimage.morphology import square
    from skimage.morphology import dilation
    return dilation(img, selem=square(radius))


def make_square_img(img_arr):
    (vertical_pixel, horizontal_pixel) = img_arr.shape
    if vertical_pixel > horizontal_pixel:
        vertical_padding = int(round(vertical_pixel*0.05))
        horizontal_padding = int(round((vertical_pixel*1.1 - horizontal_pixel) / 2))
        padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
        return skimage.util.pad(img_arr, padding, 'constant', constant_values=0)
    else:
        horizontal_padding = int(round(horizontal_pixel*0.05))
        vertical_padding = int(round((horizontal_pixel*1.1 - vertical_pixel) / 2))
        padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
        return skimage.util.pad(img_arr, padding, 'constant', constant_values=0)


def filter_zero_or_one(img_arr):
    [vert, hori] = img_arr.shape
    for i in range(vert):
        for j in range(hori):
            if img_arr[i, j] > 1:
                img_arr[i, j] = 1
    return img_arr


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    do_restore = 1
    files = [f for f in listdir('.')]  # if os.path.isfile(f)
    input_imgs = numpy.array([])

    filenames = []
    ii = 0
    for f in files:
        print('processing file: ' + f)
        if f.endswith(".png") and (not f.startswith("copy")):
            arr = make_square_img(skimage.io.imread(f, as_grey=True))
            f_new = 'copy'+f
            arr = dilate(arr, 3)
            resized_arr = filter_zero_or_one(skimage.transform.resize(arr, (28, 28)))
            #print(resized_arr)
            skimage.io.imsave(f_new, resized_arr)
            print("Image detected, name: " + f)
            print(arr.shape)
            print(resized_arr.shape)
            filenames.append(f)
            if input_imgs.size == 0:

                input_imgs = numpy.reshape(resized_arr, (1, 784))
                #input_imgs = resized_arr.flatten('C')
            else:
                input_imgs = numpy.vstack((input_imgs, numpy.reshape(resized_arr, (1, 784))))
                #input_imgs = numpy.vstack((input_imgs, resized_arr.flatten('C')))
            # print(resized_arr.flatten().shape)
            print(input_imgs.shape)
            ii += 1
            print("\n")
    print(filenames)
    input_result = numpy.zeros((len(filenames), 10))





    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)

    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    results = tf.argmax(y_conv, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()#save training
    if do_restore == 1:
        saver.restore(sess, './model')
    else:
        # Train
        for i in range(20000):  # 20000
            batch = mnist.train.next_batch(50)
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 100 == 0:
                print("step %d, training accuracy %g" % (i, train_accuracy))
        save_path = saver.save(sess, 'model')
        print("Model saved in file: %s" % save_path)
    # Test trained model
    #print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    feed_dict = {x: input_imgs, keep_prob: 1.0}

    test = {}
    correct = 0
    # read in annotation (test)
    with open("annotation_correct.txt", "r") as annotations:
        for line in annotations:
            image, label = line.split()
            test[image] = label

    p = results.eval(feed_dict=feed_dict)
    print(p)
    print(filenames)
    print(len(p))
    print(len(filenames))
    f = open('annotation.txt', 'w')
    for i in range(len(filenames)):
        s = "%s\t%i\n" % (filenames[i], (p[i]))
        f.write(s)
        #if test[filenames[i]] == p[i]:
        #   correct += 1
        print(p[i])
        print(filenames[i])
        print("\n")
    f.close()

    with open("annotation.txt", "r") as predictions:
        for line in predictions:
            image, label = line.split()
            if label == test[image]:
                correct += 1

    print("correct rate %f" % (correct/len(filenames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


