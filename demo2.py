import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

path = {'mask': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\masks\\80.png',
        'org': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\orgs\\80.png',
        'label': 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\masks\\80.jpg'}

def test_libs():
    img = np.array(Image.open(path['org']), dtype=np.uint8).reshape((1, 512, 512, 3))
    mask = np.array(Image.open(path['mask']).convert('L'), dtype=np.float32) / 255.0
    # mask = cv2.GaussianBlur(mask, (5, 5), 2.0)
    mask_ = np.zeros_like(mask)
    mask_[mask > 0.5] = 1.0
    mask = mask.reshape((1, 512, 512, 1))
    arg_mask = 1.0 - mask
    probs = np.concatenate([arg_mask, mask], axis=3)

    import utils
    crf = utils.dense_crf(probs, img)

    plt.figure()
    plt.subplot(311)
    plt.title('image')
    plt.imshow(img[0, :, :, :], cmap='gray')
    plt.subplot(312)
    plt.title('mask')
    plt.imshow(mask_, cmap='gray')
    plt.subplot(313)
    plt.title('crf')
    plt.imshow(crf[0, :, :, 1], cmap='gray')
    plt.show()


def wasserstein_adversarial_test(iter, gamma=0.001, shown=False):
    import tensorflow as tf
    from cnn_read import queueReader
    from network import CRF_RNN

    reader = queueReader()
    cnn = CRF_RNN(input_shape=[512,512,1], batch_size=1,
                  input_dim=262144, learning_rate=8e-6, pre_train=True)
    img_hat = tf.get_variable('img_hat', dtype=tf.float32, shape=[1, 512 ** 2],
                              initializer=tf.constant_initializer(0.01))
    loss = -cnn.fcn_loss + gamma * tf.reduce_sum(tf.square(img_hat - cnn.input_image_vector))

    t_vars = tf.trainable_variables()
    out_vars = [var for var in t_vars if 'FCNN' not in var.name]
    optim = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=out_vars)

    init = tf.variables_initializer(out_vars)

    for _ in range(iter):
        xs, ys = reader.next_batch(1)
        cnn.sess.run(init)

        for __ in range(300):
            cnn.sess.run(optim, feed_dict={cnn.input_image_vector: xs,
                                           cnn.label_vector: ys[0, :]})
        wae = cnn.sess.run(img_hat).reshape(512, 512)
        org_loss = cnn.sess.run(cnn.fcn_loss, feed_dict={cnn.input_image_vector: xs,
                                                         cnn.label_vector: ys[0, :]})
        adv_loss = cnn.sess.run(cnn.fcn_loss, feed_dict={cnn.input_image_vector: wae.reshape((1, 512**2)),
                                                         cnn.label_vector: ys[0, :]})
        dist = float(np.square(xs - wae.reshape((1, 512**2))).sum())
        print('--origin image loss: %g --adversarial example loss: %g --L2 distance: %g' %
              (org_loss, adv_loss, dist))

        _, summary_str = cnn.sess.run([cnn.fcn_optim, cnn.summary], feed_dict={cnn.input_image_vector: xs,
                                                                               cnn.label_vector: ys[0, :]})
        cnn.sess.run(cnn.fcn_optim, feed_dict={cnn.input_image_vector: wae.reshape((1, 512**2)),
                                               cnn.label_vector: ys[0, :]})
        cnn.writer.add_summary(summary_str, cnn.counter)
        cnn.counter += 1

        if shown:
            plt.figure()
            plt.subplot(121)
            plt.title('image')
            plt.imshow(xs.reshape((512, 512)), cmap='gray')
            plt.subplot(122)
            plt.title('wae')
            plt.imshow(wae, cmap='gray')
            plt.show()
            plt.close()
    if 'y' in str(input('save??')):
        cnn.save()


if __name__ == '__main__':
    wasserstein_adversarial_test(100)