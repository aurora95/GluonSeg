from __future__ import absolute_import

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import Loss, _apply_weighting

class MySoftmaxCrossEntropyLoss(Loss):
    def __init__(self, axis=1, from_logits=False, weight=None,
                 batch_axis=0, ignore_label=255, **kwargs):
        super(MySoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._from_logits = from_logits
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output, axis=self._axis)

        valid_label_map = (label != self._ignore_label)
        loss = -(F.pick(output, label, axis=self._axis, keepdims=True) * valid_label_map )

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        #return loss
        #return F.sum(loss, axis=self._batch_axis, exclude=True) / F.sum(valid_label_map, axis=self._batch_axis, exclude=True)
        return F.sum(loss) / F.sum(valid_label_map)
