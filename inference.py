from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
import os

import utils
from cnn_read import Reader, dcmReader, multiClassReader

class CRF_RNN(object):
    def __init__(self, input_shape, batch_size=4, learning_rate=0.0002, n_classes=2,
                 model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        # Copy parameters
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.output_shape = [batch_size, input_shape[0], input_shape[1], n_classes]
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.model_dir = model_dir

        self._build_model()
        self.fcn_loss, self.crf_loss = self._loss_function()
        self.fcn_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fcn_loss, var_list=self.fcn_vars)
        self.crf_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.crf_loss, var_list=self.crf_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        self.fcn_summary = tf.summary.scalar('fcn_loss', self.fcn_loss)
        self.crf_summary = tf.summary.scalar('crf_loss', self.crf_loss)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # load pre_trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.input_image_vector = tf.placeholder(tf.float32, shape=(None, self.input_dim), name='input_img')
        self.input_image = tf.reshape(self.input_image_vector, shape=[self.batch_size] + self.input_shape)
        self.label_vector = tf.placeholder(tf.float32, shape=[None, self.n_classes * self.input_dim], name='label')
        self.factor_a = tf.placeholder(tf.float32, shape=self.output_shape, name='factor_a')
        self.factor_b = tf.placeholder(tf.float32, shape=self.output_shape, name='factor_b')

        self.fcnn, self.fcnn_logits = self._build_fcn()
        # applied for training
        crf_a, crf_b = self._build_crf(self.fcnn, self.fcnn, reuse=False)
        self.crf_logits = crf_a * crf_b
        self.crf = tf.nn.softmax(self.crf_logits, dim=3, name='crf')
        # only applied for iteration
        self.crf_iter = self._build_crf(self.factor_a, self.factor_b, reuse=True)

        trainable_variables = tf.trainable_variables()
        self.fcn_vars = [var for var in trainable_variables if 'FCNN' in var.name]
        self.crf_vars = [var for var in trainable_variables if 'CRF' in var.name]

    def _build_fcn(self):
        # deconv size
        row, col = self.input_shape[0], self.input_shape[1]
        row_p1, col_p1 = int(row / 2), int(col / 2)
        row_p2, col_p2 = int(row_p1 / 2), int(col_p1 / 2)

        with tf.variable_scope('FCNN'):
            conv1_1 = utils.inception_block(self.input_image, n_out=32, name='inception1_1')
            conv1_2 = utils.inception_block(conv1_1, n_out=32, name='conv1_2')
            pool_1 = utils.pooling(conv1_2, name='pool_1')

            conv2_1 = utils.inception_block(pool_1, n_out=128, name='conv2_1')
            conv2_2 = utils.inception_block(conv2_1, n_out=128, name='conv2_2')
            pool_2 = utils.pooling(conv2_2, name='pool_2')

            conv3_1 = utils.inception_block(pool_2, n_out=512, name='conv3_1')
            conv3_2 = utils.inception_block(conv3_1, n_out=512, name='conv3_2')
            pool_3 = utils.pooling(conv3_2, name='pool_3')

            conv4_1 = utils.inception_block(pool_3, n_out=512, name='conv4_1')
            conv4_2 = utils.inception_block(conv4_1, n_out=512, name='conv4_2')
            deconv_1 = utils.deconv2d(conv4_2, output_shape=[self.batch_size, row_p2, col_p2, 512], name='deconv_1')

            concat_1 = tf.concat([conv3_2, deconv_1], axis=3, name='concat_1')
            conv5_1 = utils.inception_block(concat_1, n_out=128, name='conv5_1')
            deconv_2 = utils.deconv2d(conv5_1, output_shape=[self.batch_size, row_p1, col_p1, 128], name='deconv_2')

            concat_2 = tf.concat([conv2_2, deconv_2], axis=3, name='concat_2')
            conv6_1 = utils.inception_block(concat_2, n_out=32, name='conv6_1')
            conv6_2 = utils.inception_block(conv6_1, n_out=32, name='conv6_2')
            deconv_3 = utils.deconv2d(conv6_2, output_shape=[self.batch_size, row, col, 32], name='deconv_3')

            concat_3 = tf.concat([conv1_2, deconv_3], axis=3, name='concat_3')
            conv7_1 = utils.inception_block(concat_3, n_out=32, name='conv7_1')
            conv7_2 = utils.conv2d_relu(conv7_1, n_out=2, name='conv7_2')
            return tf.nn.sigmoid(conv7_2, name='sigmoid_fcn'), conv7_2

    def _build_crf(self, map_a, map_b, reuse=False):
        # reuse: iteratively update Q_{i}
        with tf.variable_scope('CRFasRNN', reuse=reuse):
            # First half
            unaries_a = tf.split(map_a, num_or_size_splits=self.n_classes, axis=3, name='unaries_a')
            unaries_b = tf.split(map_b, num_or_size_splits=self.n_classes, axis=3, name='unaries_b')
            factors_a = [[op1, op2] for op1, op2 in zip(unaries_a, unaries_b)]
            # message passing
            kernel_1x1 = tf.get_variable(name='msg_passing_weighted',
                                         shape=(1, 1, 3, 1),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
            inf_a = [utils.message_passing(pair[0], pair[1], self.input_image, kernel_1x1) for pair in factors_a]
            compat_a = tf.concat(inf_a, axis=3, name='compat_a')
            # compatible transform
            local_a = utils.conv2d(compat_a,  n_out=self.n_classes, kh=1, kw=1, name='local_a')
            # local update
            norm_a = tf.nn.softmax(-self.fcnn_logits - local_a, dim=3, name='norm_a')
            # Second half
            unaries_inf = tf.split(norm_a, num_or_size_splits=self.n_classes, axis=3, name='unarie_inf')
            factors_b = [[op1, op2] for op1, op2 in zip(unaries_a, unaries_inf)]
            # message passing
            inf_b = [utils.message_passing(pair[1], pair[0], self.input_image, kernel_1x1) for pair in factors_b]
            compat_b = tf.concat(inf_b, axis=3, name='compat_b')
            # compatible transform
            local_b = utils.conv2d(compat_b,  n_out=self.n_classes, kh=1, kw=1, name='local_b')
            # local update
            norm_b = tf.nn.softmax(-self.fcnn_logits - local_b, dim=3, name='norm_b')
            return norm_a, norm_b

    def _loss_function(self):
        logit_vector = tf.reshape(self.fcnn_logits, shape=(self.batch_size, self.n_classes * self.input_dim))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_vector, labels=self.label_vector)
        crf_vector = tf.reshape(self.crf_logits, shape=(self.batch_size, self.n_classes * self.input_dim))
        crf_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=crf_vector, labels=self.label_vector)
        return tf.reduce_sum(cross_entropy, name='fcn_loss'), tf.reduce_sum(crf_loss, name='crf_loss')

    def train(self, reader, loop=20000, print_iter=100, update_crf=True):
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            self.sess.run(self.fcn_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys})
            if update_crf:
                self.sess.run(self.crf_optim, feed_dict={self.input_image_vector: batch_xs,
                                                         self.label_vector: batch_ys})
            # Log on screen
            if i % print_iter == 5:
                loss, loss_ = self.sess.run([self.fcn_loss, self.crf_loss], feed_dict={self.input_image_vector: batch_xs,
                                                                                       self.label_vector: batch_ys})
                logging = ' --Iteration %d --FCN loss %g --CRF loss %g' % (i, loss, loss_)
                print(str(datetime.now()) + logging)
            # Log on tensorboard
            _, fcn_sum, crf_sum = self.sess.run([self.fcn_optim, self.fcn_summary, self.crf_summary],
                                                feed_dict={self.input_image_vector: batch_xs,
                                                           self.label_vector: batch_ys})
            self.writer.add_summary(fcn_sum, self.counter)
            self.writer.add_summary(crf_sum, self.counter)
            self.counter += 1
        print('Training Finished!!')
        if 'y' in str(input('Save??')):
            self.save()

    def predict(self, imgvec, use_logits=False, as_list=False):
        if use_logits:
            pred = self.sess.run(self.fcnn_logits, feed_dict={self.input_image_vector: imgvec})
        else:
            pred = self.sess.run(self.fcnn, feed_dict={self.input_image_vector: imgvec})
        input_size = pred.shape[0]
        if as_list:
            return [pred[i, :, :, 0] for i in range(input_size)]
        else:
            return pred.reshape((input_size, self.input_dim))

    def predict_n_inference(self, imgvec, as_list=True):
        pred = self.sess.run(self.crf, feed_dict={self.input_image_vector: imgvec})
        input_size = pred.shape[0]
        if as_list:
            return [pred[i, :, :, 0] for i in range(input_size)]
        else:
            return pred.reshape((input_size, self.input_dim))

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + 'CRFasRNN.model', global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    reader = multiClassReader()
    crf_rnn = CRF_RNN(input_shape=[512,512,1], batch_size=1, learning_rate=2e-4, pre_train=False)
    crf_rnn.train(reader, loop=2000)

    iter = 100
    acc_without_crf = 0.0
    acc_with_crf = 0.0
    for _ in range(iter):
        xs, ys = reader.next_batch(1)

        pred = crf_rnn.predict(xs, as_list=False)
        infe = crf_rnn.predict_n_inference(xs, as_list=False)

        tmp_without_crf = utils.get_accuracy(pred, ys)
        tmp_with_crf = utils.get_accuracy(infe, ys)
        print('--Precision without CRF: %g --Precision with CRF: %g' % (tmp_without_crf, tmp_with_crf))
        acc_with_crf += tmp_with_crf
        acc_without_crf += tmp_without_crf

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(pred, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(infe, cmap='gray')
        # plt.show()
        # plt.close()

    print('Total Precision:\n\t--Precision without CRF: %g\n\t--Precision with CRF: %g' %
          (acc_without_crf / float(iter), acc_with_crf / float(iter)))
    print('\nFinish!!!')