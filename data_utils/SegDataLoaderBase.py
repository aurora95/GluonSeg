from __future__ import absolute_import

import threading
import numpy as np
from abc import abstractmethod
from data_utils import DataTransformer

class SegDataLoaderBase(object):
    def __init__(self, num_classes, data_transformer,
                num_sample, batch_size=1, shuffle=True, seed=None):
        self.__dict__.update(locals())
        self.image_shape, self.label_shape = self.data_transformer.get_output_shape()
        if self.image_shape is None and batch_size != 1:
            raise ValueError('Batch size must be 1 when target image size is undetermined')
        self.lock = threading.Lock()
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_generator = self._flow_index(num_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        if self.image_shape is not None:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            batch_y = np.zeros((current_batch_size,) + self.label_shape).astype('uint8')
        for i, index in enumerate(index_array):
            img_x, img_y = self.load_data(index)
            x, y = self.data_transformer.transform(img_x, img_y)
            if self.image_shape == None:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                batch_y = np.zeros((current_batch_size,) + y.shape).astype('uint8')
            else:
                batch_x[i] = x
                batch_y[i] = y

        return batch_x, batch_y

    def _flow_index(self, n, batch_size, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.batch_index = 0
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def get_num_sample(self):
        return self.num_sample

    @abstractmethod
    def load_data(self, index):
        '''
        return image, label (both are PIL images)
        '''
        pass
