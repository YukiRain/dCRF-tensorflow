import os, dicom, gc
import matplotlib.pyplot as plt
from utils import *


class runningReader(object):
    def __init__(self, path):
        fs = os.listdir(path)
        self.fs = [os.path.join(path, x) for x in fs]
        self.fs.sort(key=lambda x: dicom.read_file(x).data_element('InstanceNumber').value)
        self.cnt = 0
        gc.collect()

    def sample(self):
        file = dicom.read_file(self.fs[self.cnt])
        x_arr = file.pixel_array + file.data_element('RescaleIntercept').value
        x_arr = np.array(x_arr, dtype=np.float32).reshape((1, 512 ** 2))
        x_arr[x_arr < 0.0] = 0.0
        x_arr /= 1024.0
        self.cnt += 1
        return x_arr

    def visualize(self):
        xs = self.sample()
        plt.figure()
        plt.imshow(xs.reshape((512, 512)), cmap='gray')
        plt.show()


class evalModel(object):
    def __init__(self, input_shape, input_dim, batch_size=1, model_dir='./checkpoint/', pre_train=True):
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.model_dir = model_dir
        self.batch_size = batch_size
        self._build_model()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        if pre_train and len(os.listdir(model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.input_image_vector = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        self.input_image = tf.reshape(self.input_image_vector, shape=[self.batch_size] + self.input_shape)
        self.image_uint = tf.cast(self.input_image, dtype=tf.uint8)
        self.label_vector = tf.placeholder(tf.int32, shape=[self.batch_size * self.input_dim])
        self.fcnn, self.fcnn_logits = self._build_fcn(self.input_image, is_training=False, reuse=False)
        self.prediction = tf.argmax(self.fcnn, axis=3, name='prediction')

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

    def evaluate(self, reader):
        res = list()
        while True:
            try:
                sample_t = reader.sample()
                pred = self.sess.run(self.prediction, feed_dict={self.input_image_vector: sample_t})
            except:
                break
            res.append(pred.tolist())
        return res

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


def run(path):
    reader = runningReader(path)
    model = evalModel(input_shape=[512, 512, 1], input_dim=262144, pre_train=True)
    return model.evaluate(reader)


# ans = run('E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\3\\')
# print(ans)
# print(len(ans), len(ans[0]), len(ans[0][0]))