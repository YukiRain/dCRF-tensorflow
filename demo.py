import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

path = {'mask': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\masks\\80.png',
        'org': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\orgs\\80.png',
        'label': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\masks\\80.jpg'}

has_variable = False

img = np.array(Image.open(path['org']).convert('L'), dtype=np.float32).reshape((1, 512, 512, 1))
mask = np.array(Image.open(path['mask']).convert('L'), dtype=np.float32) / 255.0
mask = mask.reshape((1, 512, 512, 1))
arg_mask = 1.0 - mask
if has_variable:
    label_ = np.array(Image.open(path['label']).convert('L'), dtype=np.float32).reshape((1, 512, 512, 1))
    label_[label_ <= 180.0] = 0.0
    label_[label_ > 180.0] = 1.0
    label = np.concatenate([1.0 - label_, label_], axis=3)

img_op = tf.placeholder(tf.float32, shape=[1, 512, 512, 1], name='img_op')
q_i = tf.placeholder(tf.float32, shape=[1, 512, 512, 1], name='q_i')
q_j = tf.placeholder(tf.float32, shape=[1, 512, 512, 1], name='q_j')
crf = utils.message_passing_v2(img_op, q_i, q_j, has_variables=has_variable)

if has_variable:
    label_op = tf.placeholder(tf.float32, shape=[1, 512, 512, 2], name='label_op')
    label_op_ = tf.reshape(tf.clip_by_value(label_op, 1e-10, 1.0), shape=[512*512, 2])
    crf_ = tf.reshape(tf.clip_by_value(crf, 1e-10, 1.0), shape=[512*512, 2])
    loss = tf.reduce_sum(label_op_ * tf.log(crf_))
    optim = tf.train.AdamOptimizer(2e-4).minimize(loss)

print('-----------RUNNING----------')
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    if has_variable:
        print('----------------TRAINING----------------')
        for i in range(300):
            sess.run(optim, feed_dict={img_op: img, q_i: mask, q_j: arg_mask, label_op: label})
    print('----------------TESTING-----------------')
    out = sess.run(crf, feed_dict={img_op: img, q_i: mask, q_j: arg_mask})

    for _ in range(50):
        tq_i = np.expand_dims(out[:, :, :, 0], axis=3)
        tq_j = np.expand_dims(out[:, :, :, 1], axis=3)
        out = sess.run(crf, feed_dict={img_op: img, q_i: tq_i, q_j: tq_j})

    result = np.argmax(out, axis=3)[0, :, :]

    plt.figure()
    plt.subplot(221)
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title('image')
    plt.subplot(222)
    plt.imshow(mask[0, :, :, 0], cmap='gray')
    plt.title('before crf')
    plt.subplot(223)
    plt.imshow(out[0, :, :, 1], cmap='gray')
    plt.title('after crf')
    plt.subplot(224)
    plt.imshow(result, cmap='gray')
    plt.title('result')

    plt.show()