from datetime import datetime
from functools import reduce
import os

from utils import *
from cnn_read import Reader, queueReader, testReader


class CRF_RNN(object):
    def __init__(self, input_shape, input_dim, batch_size=4, learning_rate=0.0002,
                 model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.model_dir = model_dir

        self._build_model()
        self.fcn_loss, self.sub_loss = self._loss_function()
        self.fcn_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fcn_loss, var_list=self.fcn_vars)
        self.sub_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.sub_loss, var_list=self.fcn_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        fcn_summary = tf.summary.scalar('fcn_loss_summary', self.fcn_loss)
        img_summary = tf.summary.image('origin images', self.input_image)
        sub_img_summary = tf.summary.image('augmented images', self.sub_img)
        sub_pred_summary = tf.summary.image('augmented prediction',
                                            tf.concat([tf.zeros_like(self.input_image, dtype=tf.float32),
                                                       self.sub_fcnn], axis=3))
        pred_summary = tf.summary.image('predictions',
                                        tf.concat([tf.zeros_like(self.input_image, dtype=tf.float32), self.fcnn],
                                        axis=3))
        self.summary = tf.summary.merge([fcn_summary])
        self.img_summary = tf.summary.merge([img_summary, sub_img_summary,
                                             pred_summary, sub_pred_summary])
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # load pre_trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.input_image_vector = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        self.input_image = tf.reshape(self.input_image_vector, shape=[self.batch_size] + self.input_shape)
        self.label_vector = tf.placeholder(tf.int32, shape=[self.batch_size * self.input_dim])
        label_image = tf.reshape(self.label_vector, shape=[self.batch_size] + self.input_shape)
        # data augmentation
        pair_sample = tf.concat([self.input_image, tf.cast(label_image, tf.float32)], axis=3)
        image_cropped = tf.random_crop(pair_sample, size=[self.batch_size, 200, 200, 2])
        image_resized = tf.image.resize_images(image_cropped, size=[512, 512], method=tf.image.ResizeMethod.BILINEAR)
        image_reshaped = tf.reshape(image_resized, shape=[512, 512, 2])
        image_flip = tf.image.random_flip_up_down(tf.image.random_flip_left_right(image_reshaped))
        sub_img, sub_label = tf.split(image_flip, num_or_size_splits=[1, 1], axis=2)
        self.sub_img = tf.expand_dims(tf.image.random_contrast(sub_img, 0.1, 0.9), dim=0)
        self.sub_label = tf.cast(tf.expand_dims(sub_label, dim=0), tf.int32)

        self.random_image = tf.reshape(tf.image.random_contrast(self.input_image, 0.2, 1.0),
                                       shape=[self.batch_size, self.input_dim], name='random_image')

        self.fcnn, self.fcnn_logits = self._build_fcn(self.input_image, is_training=True, reuse=False)
        self.sub_fcnn, self.sub_logits = self._build_fcn(self.sub_img, is_training=True, reuse=True)
        self.crf = self._build_crf(self.fcnn, self.input_image, self.fcnn_logits)

        trainable_variables = tf.trainable_variables()
        self.fcn_vars = [var for var in trainable_variables if 'FCNN' in var.name]

    def _build_fcn(self, input_op, reuse, is_training):
        row, col = self.input_shape[0], self.input_shape[1]
        row_p1, col_p1 = int(row / 2), int(col / 2)
        with tf.variable_scope('FCNN', reuse=reuse):
            conv1_1 = inception_block(input_op, n_out=64, name='inception1_1')
            conv1_2 = inception_block(conv1_1, n_out=64, name='conv1_2')
            conv1_3 = inception_block(conv1_2, n_out=64, name='conv1_3', activation='None')
            pool_1 = pooling(tf.nn.relu(batch_norm(conv1_3, is_training=is_training, name='bn_1')), name='pool_1')

            conv2_1 = inception_block(pool_1, n_out=128, name='conv2_1')
            conv2_2 = inception_block(conv2_1, n_out=128, name='conv2_2')
            conv2_3 = inception_block(conv2_2, n_out=128, name='conv2_3', activation='None')
            pool_2 = pooling(tf.nn.relu(batch_norm(conv2_3, is_training=is_training, name='bn_2')), name='pool_2')

            conv3_1 = inception_block(pool_2, n_out=256, name='conv3_1')
            conv3_2 = inception_block(conv3_1, n_out=256, name='conv3_2')
            conv3_3 = inception_block(conv3_2, n_out=256, name='conv3_3', activation='None')
            deconv_1 = deconv2d(tf.nn.relu(batch_norm(conv3_3, is_training=is_training, name='bn_3')),
                                output_shape=[self.batch_size, row_p1, col_p1, 128])

            concat_1 = tf.concat([conv2_3, deconv_1], axis=3, name='concat_1')
            conv4_1 = conv2d_relu(concat_1, n_out=128, name='conv4_1')
            conv4_2 = inception_block(conv4_1, n_out=128, name='conv4_2', activation='None')
            deconv_2 = deconv2d(tf.nn.relu(batch_norm(conv4_2, is_training=is_training, name='bn_4')),
                                output_shape=[self.batch_size, row, col, 32], name='deconv_2')

            concat_2 = tf.concat([conv1_3, deconv_2], axis=3, name='concat_2')
            conv5_1 = conv2d_relu(concat_2, n_out=64, name='conv5_1')
            conv5_2 = conv2d_relu(conv5_1, n_out=32, name='conv5_2')
            conv5_3 = conv2d(conv5_2, n_out=2, name='conv5_3')
            return tf.nn.softmax(conv5_3, axis=3, name='softmax'), conv5_3

    @staticmethod
    def _build_crf(q_op, img_op, unaries=None, theta=10.0, has_variables=False):
        with tf.variable_scope('CRF'):
            Q_j, Q_i = tf.split(q_op, [1, 1], axis=3)
            # unary_j, unary_i = tf.split(unaries, [1, 1], axis=3)
            # Q_i must be one channel for axis==3
            n_in = img_op.get_shape()[-1].value
            n_in_ = Q_j.get_shape()[-1].value

            kernel_ = np.array(get_message_kernels(theta), dtype=np.float32)
            kernel_ = np.array([kernel_ for _ in range(n_in)], dtype=np.float32).transpose((2, 3, 0, 1))
            _kernel = np.array([get_message_kernels(abstract=False)], dtype=np.float32)
            _kernel = np.concatenate([_kernel for _ in range(n_in_)], axis=0).transpose((2, 3, 0, 1))

            p_kernel = tf.constant(kernel_, dtype=tf.float32, name='p_kernel')
            z_out = tf.nn.conv2d(img_op * 100.0, p_kernel, (1, 1, 1, 1), padding='SAME')
            t_kernel = tf.constant(_kernel, dtype=tf.float32, name='t_kernel')
            q_i_out = tf.nn.conv2d(Q_i, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_i_out')
            q_j_out = tf.nn.conv2d(Q_j, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_j_out')
            message_i = tf.exp(-tf.square(z_out)) * q_j_out
            message_j = tf.exp(-tf.square(z_out)) * q_i_out

            # inference for both Q_i and Q_j
            appearance_i = tf.reduce_sum(message_i, axis=3, keepdims=True, name='appearance_i')
            appearance_j = tf.reduce_sum(message_j, axis=3, keepdims=True, name='appearance_j')
            smooth_i = const_conv2d(Q_j, ink=gaussian_5x5, n_out=1, name='smooth_i') * Q_i
            smooth_j = const_conv2d(Q_i, ink=gaussian_5x5, n_out=1, name='smooth_j') * Q_j

            # non-parametric is also implement here for the convenience of testing
            if has_variables:
                probs = tf.concat([Q_j, Q_i], axis=3, name='probs')
                concat = tf.concat([appearance_i, smooth_i, appearance_j, smooth_j], axis=3, name='concat')
                k_out = tf.get_variable('kernel_i', shape=(1, 1, 4, 2), dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0))
                output = tf.log(probs + 1e-10) - tf.nn.conv2d(concat, k_out, (1, 1, 1, 1), padding='SAME')
            else:
                psi_i = tf.log(Q_i + 1e-10) - 5.0 * appearance_i - 1.0 * smooth_i
                psi_j = tf.log(Q_j + 1e-10) - 5.0 * appearance_j - 1.0 * smooth_j
                output = tf.concat([psi_j, psi_i], axis=3, name='concat')
            return tf.nn.softmax(output, axis=3, name='softmax')

    def _loss_function(self, weight=0.001):
        logit_vector = tf.reshape(self.fcnn_logits, shape=(self.batch_size * self.input_dim, 2))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_vector,
                                                                       labels=self.label_vector)
        sub_label_vector = tf.reshape(self.sub_label, shape=[self.batch_size * self.input_dim])
        sub_logits_vector = tf.reshape(self.sub_logits, shape=(self.batch_size * self.input_dim, 2))
        sub_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sub_logits_vector,
                                                                           labels=sub_label_vector)
        return tf.reduce_sum(cross_entropy), tf.reduce_sum(sub_cross_entropy)

    def train(self, reader, loop=20000, print_iter=100):
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            self.sess.run(self.fcn_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys[0, :]})
            self.sess.run(self.sub_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys[0, :]})
            rand_imgs = self.sess.run(self.random_image, feed_dict={self.input_image_vector: batch_xs})
            self.sess.run(self.fcn_optim, feed_dict={self.input_image_vector: rand_imgs,
                                                     self.label_vector: batch_ys[0, :]})

            # Log on screen
            if i % print_iter == 5:
                loss = self.sess.run(self.fcn_loss, feed_dict={self.input_image_vector: batch_xs,
                                                               self.label_vector: batch_ys[0, :]})
                logging = ' --Iteration %d --FCN loss %g' % (i, loss)
                _, summary_str = self.sess.run([self.sub_optim, self.img_summary],
                                               feed_dict={self.input_image_vector: batch_xs,
                                                          self.label_vector: batch_ys[0, :]})
                self.writer.add_summary(summary_str, self.counter)
                print(str(datetime.now()) + logging)
            # Log on tensorboard
            _, summary_str = self.sess.run([self.sub_optim, self.summary],
                                           feed_dict={self.input_image_vector: batch_xs,
                                                      self.label_vector: batch_ys[0, :]})
            self.writer.add_summary(summary_str, self.counter)
            self.counter += 1
        print('Training finished, ready to save...')
        self.save()

    def predict(self, imgvec):
        return self.sess.run(self.fcnn, feed_dict={self.input_image_vector: imgvec})

    def predict_n_test(self, imgvec, label_vec):
        return self.sess.run([self.sub_fcnn, self.sub_label], feed_dict={self.input_image_vector: imgvec,
                                                                         self.label_vector: label_vec[0, :]})

    def predict_n_inference(self, imgvec):
        return self.sess.run(self.crf, feed_dict={self.input_image_vector: imgvec})

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
    import pydensecrf as dcrf
    import matplotlib.pyplot as plt
    from scipy.ndimage import filters

    reader = queueReader(x_path='E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\3\\',
                         y_path='E:\\C++\\Projects\\TrainingData\\TrainingData\\')
    crf_rnn = CRF_RNN(input_shape=[512, 512, 1], batch_size=1,
                      input_dim=262144, learning_rate=8e-4, pre_train=True)
    show_all_variables()
    print('----------------------TRAINING----------------------')
    crf_rnn.train(reader, loop=12000)
    print('----------------------TESTING-----------------------')

    iter = 20
    acc_without_crf = 0.0
    acc_with_crf = 0.0
    for _ in range(iter):
        xs, ys = reader.next_batch(1)
        xs_uint_rgb = np.concatenate([xs.reshape((1, 512, 512, 1)) for _ in range(3)], axis=3).astype(np.uint8)

        pred = crf_rnn.predict(xs)
        pred_blur = filters.gaussian_filter(pred, sigma=10.0)
        infe = dense_crf(pred, xs_uint_rgb)

        pred = np.argmax(pred, axis=3).reshape((1, 512**2))
        infe = np.argmax(infe, axis=3).reshape((1, 512**2))

        tmp_without_crf = get_accuracy(pred, ys)
        tmp_with_crf = get_accuracy(infe, ys.reshape((1, 512**2)))
        print('--IoU Precision without CRF: %g --IoU Precision with CRF: %g' % (tmp_without_crf, tmp_with_crf))
        acc_with_crf += tmp_with_crf
        acc_without_crf += tmp_without_crf

        plt.figure()
        plt.subplot(221)
        plt.imshow(pred.reshape((512, 512)), cmap='gray')
        plt.title('without CRF')
        plt.subplot(222)
        plt.imshow(infe.reshape((512, 512)), cmap='gray')
        plt.title('with CRF')
        plt.subplot(223)
        plt.imshow(xs.reshape((512, 512)), cmap='gray')
        plt.title('origin')
        plt.subplot(224)
        plt.imshow(ys.reshape((512, 512)), cmap='gray')
        plt.title('Ground Truth')
        plt.show()
        plt.close()

    print('\nTotal Precision:\n\t--Precision without CRF: %g\n\t--Precision with CRF: %g' %
          (acc_without_crf / float(iter), acc_with_crf / float(iter)))

    print('\nFinished!!!')