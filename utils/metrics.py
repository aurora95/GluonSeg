from __future__ import absolute_import

import mxnet as mx
from mxnet.metric import EvalMetric, check_label_shapes

class AccuracyWithIgnoredLabel(EvalMetric):
    def __init__(self, axis=1, ignore_label=255, name='accuracy',
                 output_names=None, label_names=None):
        super(AccuracyWithIgnoredLabel, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis, keepdims=True)
            label = label.astype('int32')
            pred_label = pred_label.astype('int32').as_in_context(label.context)

            check_label_shapes(label, pred_label)

            correct = mx.nd.sum( (label == pred_label) * (label != self.ignore_label) ).asscalar()
            total = mx.nd.sum( (label != self.ignore_label) ).asscalar()

            self.sum_metric += correct
            self.num_inst += total
