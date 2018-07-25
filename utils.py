import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import math
import pydensecrf.densecrf as dcrf


def batch_norm(input_op, is_training, epsilon=1e-5, momentum=0.99, name='batch_norm'):
    return tf.contrib.layers.batch_norm(input_op, decay=momentum, updates_collections=None,
                                        epsilon=epsilon, scale=True, is_training=is_training, scope=name)

def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def leaky_relu(input_op, leak=0.2, name='linear'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=3, kw=3, dh=1, dw=1, reuse=False):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name, reuse=reuse):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

def conv2d_relu(input_op, n_out, name, kh=3, kw=3, dh=1, dw=1):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(z_out, name='relu')

def atrous_conv(input_op, n_out, rate, name, kh=3, kw=3, activate='relu'):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.atrous_conv2d(input_op, kernel, rate=rate, padding='SAME')
        biases = tf.get_variable('biases', (n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        if activate == 'relu':
            return tf.nn.relu(z_out, name='relu')
        elif activate == 'lrelu':
            return leaky_relu(z_out)
        else:
            return z_out

def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
    if 'max' in pooling_type:
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
    else:
        return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def deconv2d(input_op, output_shape, kh=3, kw=3, dh=2, dw=2, name='deconv', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]
    # filter : [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.nn.relu(tf.nn.bias_add(deconv, biases), name='deconv_activate')

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0, activate='lrelu', with_kernels=False):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='matrix',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input_op, kernel) + biases

def inception_block(input_op, n_out, name='inception_block', activation='relu'):
    # inception v3 with dilated convolution
    with tf.variable_scope(name):
        conv3x3 = conv2d(input_op, n_out=n_out / 2, name='conv1x1', kh=3, kw=3)
        atrous3x3 = atrous_conv(input_op, n_out=n_out / 2, rate=4, kh=3, kw=3, name='atrous_3x3', activate='None')
        concatenated = tf.concat([conv3x3, atrous3x3], axis=3, name='concatenated')
        if activation == 'relu':
            return tf.nn.relu(concatenated, name='relu')
        else:
            return concatenated

def get_accuracy(xs, ys):
    # xs and ys should have the same shape/size
    # ys must be normalized to 0 - 1
    intersect = np.multiply(xs, ys)
    union = np.array(xs + ys).astype(np.float32)
    union[union > 0.5] = 1.0
    union[union <= 0.5] = 0.0
    return (intersect.sum() / (union.sum() + 1e-10))

'''
    以下CRF内容
    复现论文：Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
    复现论文：Conditional Random Fields as Recurrent Neural Network
'''

gaussian_5x5 = np.array([[1,2,3,2,1],
                         [2,5,6,5,2],
                         [3,6,8,6,3],
                         [2,5,6,5,2],
                         [1,2,3,2,1]]).astype(np.float32)
gaussian_5x5 /= float(gaussian_5x5.sum())

laplacian_5x5 = np.array([[0,0,1,0,0],
                          [0,1,2,1,0],
                          [1,2,-16,2,1],
                          [0,1,2,1,0],
                          [0,0,1,0,0]], dtype=np.float32)

sobel_3x3 = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float32)

def get_message_kernels(theta=5.0, abstract=True):
    output = list()
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            kernel = np.zeros((5, 5), dtype=np.float32)
            kernel[i, j] = 1.0
            if abstract:
                kernel[2, 2] = -1.0
                d = ((i - 2)**2 + (j - 2)**2) / theta
                kernel *= math.exp(-d)
            output.append(kernel)
    return output

def get_big_message_kernels(theta=5.0, abstract=True):
    output = list()
    for i in range(9):
        for j in range(9):
            if i == 4 and j == 4:
                continue
            kernel = np.zeros((9, 9), dtype=np.float32)
            kernel[i, j] = 1.0
            if abstract:
                kernel[4, 4] = -1.0
                d = ((i - 4)**2 + (j - 4)**2) / theta
                kernel *= math.exp(-d)
            output.append(kernel)
    return output

def message_passing(img_op, Q_i, Q_j, name='message_passing', theta=10.0,
                    alpha=10.0, beta=0.1, reuse=False, has_variables=False):
    # Q_i must be one channel for axis==3
    n_in = img_op.get_shape()[-1].value
    n_in_ = Q_j.get_shape()[-1].value

    kernel_ = np.array(get_message_kernels(theta), dtype=np.float32)
    kernel_ = np.array([kernel_ for _ in range(n_in)], dtype=np.float32).transpose((2,3,0,1))
    _kernel = np.array([get_message_kernels(abstract=False)], dtype=np.float32)
    _kernel = np.concatenate([_kernel for _ in range(n_in_)], axis=0).transpose((2,3,0,1))

    with tf.variable_scope(name, reuse=reuse):
        p_kernel = tf.constant(kernel_, dtype=tf.float32, name='p_kernel')
        z_out = tf.nn.conv2d(img_op, p_kernel, (1, 1, 1, 1), padding='SAME')
        t_kernel = tf.constant(_kernel, dtype=tf.float32, name='t_kernel')
        q_i_out = tf.nn.conv2d(Q_i, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_i_out')
        q_j_out = tf.nn.conv2d(Q_j, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_j_out')
        message_i = tf.exp(-tf.square(z_out)) * q_j_out
        message_j = tf.exp(-tf.square(z_out)) * q_i_out

        # inference for both Q_i and Q_j
        appearance_i = tf.reduce_sum(message_i, axis=3, keep_dims=True, name='appearance_i')
        appearance_j = tf.reduce_sum(message_j, axis=3, keep_dims=True, name='appearance_j')
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
            psi_i = tf.log(Q_i + 1e-10) - alpha * appearance_i - beta * smooth_i
            psi_j = tf.log(Q_j + 1e-10) - alpha * appearance_j - beta * smooth_j
            output = tf.concat([psi_j, psi_i], axis=3, name='concat')
        return tf.nn.softmax(output, axis=3, name='softmax')

def const_conv2d(input_op, ink, n_out, name, dh=1, dw=1):
    n_in = input_op.get_shape()[-1].value
    kernel_ = np.array([ink for _ in range(n_in)], dtype=np.float32).transpose((1,2,0))
    kernel_ = np.array([kernel_ for _ in range(n_out)], dtype=np.float32).transpose((1,2,3,0))
    with tf.variable_scope(name):
        kernel = tf.constant(kernel_, dtype=tf.float32, name='const_kernel')
        return tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')

def message_passing_v2(img_op, Q_i, Q_j, name='message_passing', theta=20.0,
                    alpha=2.0, beta=0.2, reuse=False, has_variables=False):
    # Q_i must be one channel for axis==3
    n_in = img_op.get_shape()[-1].value
    n_in_ = Q_j.get_shape()[-1].value

    kernel_ = np.array(get_message_kernels(theta), dtype=np.float32)
    kernel_ = np.array([kernel_ for _ in range(n_in)], dtype=np.float32).transpose((2,3,0,1))
    _kernel = np.array([get_message_kernels(abstract=False)], dtype=np.float32)
    _kernel = np.concatenate([_kernel for _ in range(n_in_)], axis=0).transpose((2,3,0,1))

    with tf.variable_scope(name, reuse=reuse):
        p_kernel = tf.constant(kernel_, dtype=tf.float32, name='p_kernel')
        z_out = tf.nn.conv2d(img_op, p_kernel, (1, 1, 1, 1), padding='SAME')
        t_kernel = tf.constant(_kernel, dtype=tf.float32, name='t_kernel')
        q_i_out = tf.nn.conv2d(Q_i, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_i_out')
        q_j_out = tf.nn.conv2d(Q_j, t_kernel, (1, 1, 1, 1), padding='SAME', name='q_j_out')
        message_i = tf.exp(-tf.square(z_out)) * q_j_out
        message_j = tf.exp(-tf.square(z_out)) * q_i_out

        # inference for both Q_i and Q_j
        appearance_i = tf.reduce_sum(message_i, axis=3, keep_dims=True, name='appearance_i')
        appearance_j = tf.reduce_sum(message_j, axis=3, keep_dims=True, name='appearance_j')
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
            psi_i = tf.log(Q_i + 1e-10) - 15.0 * appearance_i - 2.0 * smooth_i
            psi_j = tf.log(Q_j + 1e-10) - 2.0 * appearance_j - 2.0 * smooth_j
            output = tf.concat([psi_j, psi_i], axis=3, name='concat')
        return tf.nn.softmax(output, axis=3, name='softmax')

# The idea is borrowed from https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/crf
def dense_crf(probs, img=None, n_iters=5, n_classes=2,
              sxy_gaussian=(1, 1), compat_gaussian=1,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=3,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs + 1e-8)  # Unary potential (avoid probs underflow to 0)
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1: 3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)

'''
    以下内容复现论文ICLR2018: Certifiable Distributional Robustness with Principled Adversarial Training
    核心的内容是生成adversarial examples给FCN训练，生成的样本分布与原始样本分布的Wasserstein distance不大于一个常数gamma
    假设神经网络的capacity无限，则神经网络总可以同时包含原始样本与这些adversarial examples
'''