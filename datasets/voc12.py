from __future__ import absolute_import

import numpy as np
from PIL import Image
from data_utils import *

class VOC12(SegDataLoaderBase):
    def __init__(self, data_list_file, data_source_dir, label_source_dir,
                data_transformer, batch_size=1, shuffle=True, seed=None):
        self.data_source_dir = data_source_dir
        self.label_source_dir = label_source_dir
        self.data_files = []
        self.label_files = []
        with open(data_list_file) as fp:
            lines = fp.readlines()
            self.num_sample = len(lines)
            for line in lines:
                line = line.strip('\n')
                self.data_files.append(line + '.jpg')
                self.label_files.append(line + '.png')
        super(VOC12, self).__init__(num_classes=21, data_transformer=data_transformer,
                                    num_sample=self.num_sample, batch_size=batch_size,
                                    shuffle=shuffle, seed=seed)

    def load_data(self, index):
        data_file = self.data_files[index]
        label_file = self.label_files[index]

        image = Image.open(os.path.join(self.data_source_dir, data_file))
        label = Image.open(os.path.join(self.label_source_dir, label_file))

        return image, label


if __name__ == '__main__':
    from data_utils import GeneratorEnqueuer
    transformer = DataTransformer(ch_mean=[0.,0.,0.], ch_std=[1.,1.,1.], resize_size=None, pad_size=None,
                 crop_mode='random', crop_size=(320, 320), zoom_range=[0.5, 2.0],
                 horizontal_flip=True, color_jittering_range=20.,
                 fill_mode='constant', cval=0., label_cval=255,
                 data_format='channels_first', color_format='RGB',
                 x_dtype=np.float32)
    data_dir        = '/home/aurora/Learning/Data/VOC2012/JPEGImages'
    label_dir       = '/home/aurora/Learning/Data/VOC2012/SegmentationClass'
    val_file_path   = '/home/aurora/Learning/Data/VOC2012/ImageSets/Segmentation/val.txt'
    dataloader = VOC12(data_list_file=val_file_path, data_source_dir=data_dir,
                       label_source_dir=label_dir, data_transformer=transformer,
                       batch_size=1, shuffle=True)

    enqueuer = GeneratorEnqueuer(generator=dataloader)
    enqueuer.start(workers=1, max_queue_size=10)
    output_generator = enqueuer.get()

    x, y = next(output_generator)
    img_y = Image.open(os.path.join(label_dir, '2007_000033.png'))
    result_x = array_to_img(x[0], 'channels_first')
    result_y = Image.fromarray(y[0, 0, :, :], mode='P')
    result_y.putpalette(img_y.getpalette())
    result_x.show(title='result_x', command=None)
    result_y.show(title='result_y', command=None)

    enqueuer.stop()
