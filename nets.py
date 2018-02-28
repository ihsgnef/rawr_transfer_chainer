import numpy

import six


import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.connection import n_step_rnn
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class BiLSTMEncoder(chainer.Chain):

    """A Bi-LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units,
                                   initialW=embed_init)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs, get_embed=False):
        exs = sequence_embed(self.embed, xs, self.dropout)
        _, _, ys = self.encoder(None, None, exs)
        assert len(ys) == len(xs)
        if get_embed:
            return ys, exs
        return ys


class MLP(chainer.ChainList):

    def __init__(self, n_layers, n_units, n_output, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            if i < n_layers - 1:
                nu = n_units
            else:
                nu = n_output
            self.add_link(L.Linear(None, nu))
        self.dropout = dropout
        self.out_units = n_units

    def __call__(self, x):
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=self.dropout)
            x = F.relu(link(x))
        return x

class DoubleMaxClassifier(chainer.Chain):
    """Multi-class classifier with one encoder encoding two input sequences.
    """
    def __init__(self, n_layers, n_vocab, n_units, n_class, dropout=0.1):
        super(DoubleMaxClassifier, self).__init__()
        with self.init_scope():
            self.encoder = BiLSTMEncoder(n_layers=n_layers, n_vocab=n_vocab,
                      n_units=n_units, dropout=dropout)
            self.output = MLP(3, n_units, n_class, dropout=dropout)
        self.dropout = dropout

    def __call__(self, xs, ys, get_embed=False):
        if get_embed:
            concat_outputs, exs0, exs1 = self.predict(xs, get_embed=True)
        else:
            concat_outputs = self.predict(xs, get_embed=False)
        concat_truths = F.concat(ys, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        if get_embed:
            return loss, exs1 # FIXME
        else:
            return loss

    def predict(self, xs, softmax=False, argmax=False, get_embed=False):
        xs0, xs1 = xs # premise, hypothesis
        if get_embed:
            ys0, exs0 = self.encoder(xs0, get_embed=True)
            ys1, exs1 = self.encoder(xs1, get_embed=True)
        else:
            ys0 = self.encoder(xs0, get_embed=False)
            ys1 = self.encoder(xs1, get_embed=False)

        ys0 = [F.max(y, axis=0) for y in ys0]
        ys1 = [F.max(y, axis=0) for y in ys1]
        ys0 = F.dropout(F.stack(ys0, axis=0), ratio=self.dropout)
        ys1 = F.dropout(F.stack(ys1, axis=0), ratio=self.dropout)
        ys = F.concat([ys0, ys1, F.absolute(ys0 - ys1), ys0 * ys1], axis=1)
        ys = self.output(ys)
        if softmax:
            ys = F.softmax(ys).data
        elif argmax:
            ys = self.xp.argmax(ys.data, axis=1)
        if get_embed:
            return ys, exs0, exs1
        return ys 


class SingleMaxClassifier(chainer.Chain):

    """Multi-class clasifier with a given encoder.

     Max-pooling over the encoded sequence.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a list of "sentence_length" variables each with
             shape of "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, n_layers, n_vocab, n_units, n_class, dropout=0.1):
        super(SingleMaxClassifier, self).__init__()
        with self.init_scope():
            self.encoder = BiLSTMEncoder(n_layers=n_layers, n_vocab=n_vocab,
                      n_units=n_units, dropout=dropout)
            self.output = L.Linear(n_units * 2, n_class)
        self.dropout = dropout

    def __call__(self, xs, ys, get_embed=False):
        if get_embed:
            concat_outputs, exs = self.predict(xs, get_embed=True)
        else:
            concat_outputs = self.predict(xs, get_embed=False)
        concat_truths = F.concat(ys, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        if get_embed:
            return loss, exs
        else:
            return loss

    def predict(self, xs, softmax=False, argmax=False, get_embed=False):
        if get_embed:
            ys, exs = self.encoder(xs, get_embed=True)
        else:
            ys = self.encoder(xs, get_embed=False)

        ys = [F.max(y, axis=0) for y in ys]
        concat_encodings = F.stack(ys, axis=0)
        concat_encodings = F.dropout(concat_encodings, ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        ret = concat_outputs
        if softmax:
            ret = F.softmax(concat_outputs).data
        elif argmax:
            ret = self.xp.argmax(concat_outputs.data, axis=1)
        if get_embed:
            return ret, exs
        return ret
