import numpy as np
import scipy.ndimage as ndi
import os

from PIL import Image

def img_to_array(img, data_format='channels_last', dtype=np.float32):
    """Converts a PIL Image instance to a Numpy array, borrowed from Keras.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def array_to_img(x, data_format='channels_last', scale=True):
    """Converts a 3D Numpy array to a PIL Image instance, borrowed from Keras.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ValueError: if invalid `x` or `data_format` is passed.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def random_color_jittering(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix, borrowed from Keras.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def center_crop(x, center_crop_size, **kwargs):
    center_h, center_w = x.shape[0] // 2, x.shape[1] // 2
    half_h, half_w = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[center_h - half_w: center_h + half_h, center_w - half_w :center_w + half_w, :]


def pair_center_crop(x, y, center_crop_size, **kwargs):
    center_h, center_w = x.shape[0] // 2, x.shape[1] // 2
    half_h, half_w = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[center_h - half_w: center_h + half_h, center_w - half_w :center_w + half_w, :], \
           y[center_h - half_w: center_h + half_h, center_w - half_w :center_w + half_w]


def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    h, w = x.shape[0], x.shape[1]
    range_h = (h - random_crop_size[0]) // 2
    range_w = (w - random_crop_size[1]) // 2
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    return x[offset_h: offset_h + random_crop_size[0], offset_w: offset_w + random_crop_size[1], :]


def pair_random_crop(x, y, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    h, w = x.shape[0], x.shape[1]
    range_h = (h - random_crop_size[0]) // 2
    range_w = (w - random_crop_size[1]) // 2
    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)
    return x[offset_h: offset_h + random_crop_size[0], offset_w: offset_w + random_crop_size[1], :],\
           y[offset_h: offset_h + random_crop_size[0], offset_w: offset_w + random_crop_size[1]]


class DataTransformer(object):
    def __init__(self, ch_mean, ch_std, resize_size=None, pad_size=None,
                 crop_mode='none', crop_size=None, zoom_range=0.,
                 horizontal_flip=False, color_jittering_range=0.,
                 fill_mode='constant', cval=0., label_cval=255,
                 data_format='channels_last', color_format='RGB',
                 x_dtype=np.float32):
        self.__dict__.update(locals())
        if resize_size:
            self.resize_size = tuple(resize_size)
        if pad_size:
            self.pad_size = tuple(pad_size)
        if crop_size:
            self.crop_size = tuple(crop_size)

        if ch_mean is not None and len(ch_mean) != 3:
            raise Exception('ch_mean should be either None or a 3-element list!'
                            'Received arg: ', str(ch_mean))

        if crop_mode not in {'none', 'random', 'center'}:
            raise Exception('crop_mode should be "none" or "random" or "center" '
                            'Received arg: ', crop_mode)
        if data_format not in {'channels_last', 'channels_first'}:
            raise Exception('data_format should be channels_last (channel after row and '
                            'column) or channels_first (channel before row and column). '
                            'Received arg: ', data_format)
        if color_format not in {'RGB', 'BGR'}:
            raise Exception('crop_mode should be "RGB" or "BGR" '
                            'Received arg: ', color_format)

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def transform(self, img_x, img_y):
        img_w, img_h = img_x.size
        # resizing and transform PIL image to numpy array
        if self.resize_size:
            img_x.resize((self.resize_size[0], self.resize_size[1]), Image.BILINEAR)
            img_y.resize((self.resize_size[0], self.resize_size[1]), Image.NEAREST)
        x = img_to_array(img_x, 'channels_last', dtype=self.x_dtype)
        y = img_to_array(img_y, 'channels_last', dtype=np.uint8)
        # process color format and channel mean and color jittering
        if self.color_format == 'BGR':
            x = x[..., ::-1]
        if self.color_jittering_range != 0:
            x = random_color_jittering(x, self.color_jittering_range, channel_axis=2)
        if self.ch_mean:
            x[..., 0] -= self.ch_mean[0]
            x[..., 1] -= self.ch_mean[1]
            x[..., 2] -= self.ch_mean[2]
        if self.ch_std:
            x[..., 0] /= self.ch_std[0]
            x[..., 1] /= self.ch_std[1]
            x[..., 2] /= self.ch_std[2]


        # process padding
        if self.pad_size:
            pad_h = self.pad_size[0] * 2
            pad_w = self.pad_size[1] * 2
        else:
            pad_h = 0
            pad_w = 0
        if self.crop_size:
            pad_h = max(pad_h, self.crop_size[0] - img_h)
            pad_w = max(pad_w, self.crop_size[1] - img_w)
        x = np.lib.pad(x,
                ((pad_h//2, pad_h - pad_h//2),
                 (pad_w//2, pad_w - pad_w//2),
                 (0, 0)),
                'constant', constant_values=0)
        y = np.lib.pad(y,
                ((pad_h//2, pad_h - pad_h//2),
                 (pad_w//2, pad_w - pad_w//2),
                 (0, 0)),
                'constant', constant_values=self.label_cval)
        # process zooming, flipping
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            z = 1
        else:
            z = np.random.uniform(low=self.zoom_range[0], high=self.zoom_range[1], size=1)
        zoom_matrix  = [[z, 0, 0],
                        [0, z, 0],
                        [0, 0, 1]]
        h, w = x.shape[0], x.shape[1]
        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
        x = apply_transform(x, transform_matrix, 2,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, 2,
                            fill_mode='constant', cval=self.label_cval)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, 1)
                y = flip_axis(y, 1)
        # process cropping
        if self.crop_mode == 'center':
            x, y = pair_center_crop(x, y, self.crop_size)
        elif self.crop_mode == 'random':
            x, y = pair_random_crop(x, y, self.crop_size)
        # transpose to fit data format
        if self.data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
            y = y.transpose(2, 0, 1)
        return x, y

    def get_output_shape(self):
        output_size = None
        if self.resize_size:
            output_size = self.resize_size
        if self.crop_size:
            output_size = self.crop_size
        if output_size is None:
            return None, None
        if self.data_format == 'channels_first':
            return (3,) + output_size, (1,) + output_size
        elif self.data_format == 'channels_last':
            return output_size + (3,), output_size + (1,)

if __name__ == '__main__':
    transformer = DataTransformer(ch_mean=[0.,0.,0.], resize_size=None, pad_size=None,
                 crop_mode='random', crop_size=(480, 480), zoom_range=[0.5, 2.0],
                 horizontal_flip=True, color_jittering_range=20.,
                 fill_mode='constant', cval=0., label_cval=255,
                 data_format='channels_last', color_format='RGB',
                 x_dtype=np.float32)
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    img_x = Image.open(os.path.join(data_dir, '2007_000033.jpg'))
    img_y = Image.open(os.path.join(label_dir, '2007_000033.png'))
    x, y = transformer.transform(img_x, img_y)
    result_x = array_to_img(x, 'channels_last')
    result_y = Image.fromarray(y[:, :, 0], mode='P')
    result_y.palette = img_y.palette
    result_x.show(title='result_x', command=None)
    result_y.show(title='result_y', command=None)
