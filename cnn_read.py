from PIL import Image
import numpy as np
import os, gc
import dicom
import matplotlib.pyplot as plt
from datetime import datetime

src_path = 'E:\\C++\\Projects\\TrainingData\\Final!!!\\data\\'
label_path = 'E:\\C++\\Projects\\TrainingData\\Final!!!\\label\\'


class Reader(object):
    def __init__(self, x_path=src_path, y_path=label_path):
        self.x_data = list()
        self.y_data = list()
        self.x_files = self._run_path(x_path)
        self.y_files = self._run_path(y_path)
        self.train, self.labels = self._read()
        self.hashing = np.zeros((len(self.train))).astype(np.uint8)
        self.collect()

    @staticmethod
    def _run_path(path):
        fs = os.listdir(path)
        fs = list(map(lambda x: path + x, fs))
        return fs

    def _read(self, img_size=512, x_norm=1024.0, y_norm=255.0):
        for x_name, y_name in zip(self.x_files, self.y_files):
            x_file = dicom.read_file(x_name)
            x_arr = x_file.pixel_array + x_file.data_element('RescaleIntercept').value
            self.x_data.append(np.array(x_arr, dtype=np.float32).reshape((1, img_size*img_size)))
            y_img = Image.open(y_name).convert('L')
            self.y_data.append(np.array(y_img, dtype=np.float32).reshape((1, img_size*img_size)))
        x_array = np.concatenate(tuple(self.x_data), axis=0)
        y_array = np.concatenate(tuple(self.y_data), axis=0)
        print('Reader Initialization: Finished')
        print('Training Data Size: %d \nTest Data Size: %d' % (len(self.x_data), len(self.y_data)))
        x_array[x_array < 0.0] = 0.0
        x_array /= x_norm
        y_array /= y_norm
        return x_array, y_array

    def next_batch(self, num):
        index = np.random.randint(low=0, high=self.train.shape[0], size=(1, num))
        x_batch = self.train[index]
        y_batch = self.labels[index]
        self.hashing[index] = 1
        return x_batch[0], y_batch[0]

    def ordered_batch(self):
        return np.split(self.train, indices_or_sections=self.train.shape[0], axis=0)

    # 训练时注意控制样本数量与训练迭代采样次数成正比
    def get_test_data(self):
        idx = (self.hashing == 0)
        test_x = self.train[idx]
        test_y = self.labels[idx]
        print('%d samples remained for test!' % test_x.shape[0])
        return test_x, test_y

    def collect(self):
        del self.x_data, self.y_data
        del self.x_files, self.y_files
        gc.collect()

class staticReader(object):
    def __init__(self):
        self.train = np.load('./TrainingData/train.npy')
        self.label = np.load('./TrainingData/label.npy')
        print('Training Data Load Finished!\n\tData Samples: %d\n\tLabel Samples: %d' %
              (self.train.shape[0], self.label.shape[0]))

    def next_batch(self, num):
        index = np.random.randint(low=0, high=self.train.shape[0], size=(1, num))
        x_batch = self.train[index]
        y_batch = self.label[index]
        return x_batch[0], y_batch[0]


def dirlist(path, allfile, word):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile, word)
        elif os.path.isfile(filepath) and word in filepath:
            allfile.append(filepath)
    return allfile


class queueReader(object):
    # only implemented for image files
    def __init__(self, x_path=src_path, y_path=label_path):
        x_files = dirlist(x_path, [], word='.dcm')
        self.x_files = np.array(x_files)
        y_files = dirlist(y_path, [], word='.png')
        self.y_files = np.array(y_files)
        print('Training samples: %d, annotations: %d' % (len(x_files), len(y_files)))
        self.counter = 0

    def next_batch(self, num):
        idx = np.random.randint(0, len(self.x_files), size=(num))
        x_names, y_names = self.x_files[idx].tolist(), self.y_files[idx].tolist()
        x_list = list()
        y_list = list()
        for xn, yn in zip(x_names, y_names):
            x_file = dicom.read_file(xn)
            x_arr = x_file.pixel_array + x_file.data_element('RescaleIntercept').value
            x_arr = np.array(x_arr, dtype=np.float32).reshape((1, 512**2))
            x_arr[x_arr < 0.0] = 0.0
            y_arr = np.array(Image.open(yn).convert('L'), dtype=np.float32).reshape((1, 512**2))
            x_list.append(x_arr / 1024.0)
            y_list.append(y_arr / 255.0)
        self.counter += 1
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)

    def visualize(self):
        xs, ys = self.next_batch(1)
        plt.figure()
        plt.subplot(121)
        plt.imshow(xs.reshape((512, 512)), cmap='gray')
        plt.subplot(122)
        plt.imshow(ys.reshape((512, 512)), cmap='gray')
        plt.show()


class testReader(object):
    def __init__(self, x_path=src_path):
        x_files = os.listdir(x_path)
        self.x_files = np.array([x_path + name for name in x_files])

    def next_batch(self, num):
        idx = np.random.randint(0, len(self.x_files), size=(num))
        x_names = self.x_files[idx].tolist()
        x_list = list()
        for xn in  x_names:
            x_file = dicom.read_file(xn)
            x_arr = x_file.pixel_array + x_file.data_element('RescaleIntercept').value
            x_arr = np.array(x_arr, dtype=np.float32).reshape((1, 512**2))
            x_arr[x_arr < 0.0] = 0.0
            x_list.append(x_arr / 1024.0)
        return np.concatenate(x_list, axis=0)


if __name__ == '__main__':
    from network import CRF_RNN
    import tensorflow as tf
    import cv2

    reader = testReader(x_path='E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\3\\3\\')
    crf_rnn = CRF_RNN(input_shape=[512,512,1], batch_size=1,
                      input_dim=262144, learning_rate=4e-4, pre_train=True)

    print('----------------------TESTING-----------------------')

    q_op = tf.placeholder(tf.float32, shape=[1, 512, 512, 2], name='q_op')
    crf = crf_rnn._build_crf(q_op, crf_rnn.input_image, None, has_variables=False)

    for iter in range(50):
        xs = reader.next_batch(1)
        pred = crf_rnn.predict(xs)
        infe = crf_rnn.predict_n_inference(xs)

        for t in range(30):
            infe_0 = cv2.GaussianBlur(infe[0, :, :, 0], (5, 5), 1.5)
            infe_1 = cv2.GaussianBlur(infe[0, :, :, 1], (5, 5), 1.5)
            infe = np.concatenate([infe_0[None, :, :, None], infe_1[None, :, :, None]], axis=3)
            infe = crf_rnn.sess.run(crf, feed_dict={q_op: infe,
                                                    crf_rnn.input_image_vector: xs})

        pred_0 = cv2.GaussianBlur(pred[0, :, :, 0], (3, 3), 1.0)
        pred_1 = cv2.GaussianBlur(pred[0, :, :, 1], (3, 3), 1.0)
        pred = np.concatenate([pred_0[None, :, :, None], pred_1[None, :, :, None]], axis=3)
        pred = np.argmax(pred, axis=3).reshape((512, 512))
        infe = np.argmax(infe, axis=3).reshape((512, 512))

        plt.figure()
        plt.subplot(221)
        plt.imshow(xs.reshape((512, 512)), cmap='gray')
        plt.title('source image')
        plt.subplot(222)
        plt.imshow(pred, cmap='gray')
        plt.title('without crf')
        plt.subplot(223)
        plt.imshow(infe, cmap='gray')
        plt.title('with crf')
        plt.show()
        plt.close()

    print('Done!!')