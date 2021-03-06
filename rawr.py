import os
import json
import pickle
import numpy as np
import itertools
import argparse
from tqdm import tqdm

import cupy
import chainer
from chainer import functions as F

import nets
import text_datasets
from nlp_utils import convert_seq


def get_onehot_grad(model, xs, ys=None):
    if ys is None:
        with chainer.using_config('train', False):
            ys = model.predict(xs, argmax=True)
            ys = F.expand_dims(ys, axis=1)
            ys = [y for y in ys]
    loss, exs = model(xs, ys, get_embed=True, no_dropout=True)
    exs_grad = chainer.grad([loss], exs)
    ex_sections = np.cumsum([ex.shape[0] for ex in exs[:-1]])
    exs = F.concat(exs, axis=0)
    exs_grad = F.concat(exs_grad, axis=0)
    onehot_grad = F.sum(exs_grad * exs, axis=1)
    onehot_grad = F.split_axis(onehot_grad, ex_sections, axis=0)
    return onehot_grad


def remove_one(model, xs, n_beams, indices, removed_indices, max_beam_size=5):
    n_examples = len(n_beams)
    onehot_grad = [x.data for x in get_onehot_grad(model, xs)]
    xp = cupy.get_array_module(*onehot_grad)
    # don't remove <eos>
    order = [xp.argsort(x[:-1]).tolist() for x in onehot_grad]

    new_xs = []
    new_n_beams = []
    new_indices = []
    new_removed_indices = []

    start = 0
    for example_idx in range(n_examples):
        if n_beams[example_idx] == 0:
            new_n_beams.append(0)
            continue

        coordinates = []
        for i in range(start, start + n_beams[example_idx]):
            for j in order[i][:max_beam_size]:
                coordinates.append((onehot_grad[i][j], (i, j)))
        if len(coordinates) == 0:
            new_n_beams.append(0)
            start += n_beams[example_idx]
            continue

        coordinates = sorted(coordinates, key=lambda x: -x[0])
        coordinates = [c for _, c in coordinates][:max_beam_size]
        for i, j in coordinates:
            x = xs[i].copy()
            x = xp.concatenate([x[:j], x[j+1:]], axis=0)
            new_xs.append(x)
            new_indices.append(indices[i][:j] + indices[i][j+1:])
            new_removed_indices.append(removed_indices[i] + [indices[i][j]])
        new_n_beams.append(len(coordinates))
        start += n_beams[example_idx]
    return new_xs, new_n_beams, new_indices, new_removed_indices


def get_rawr(model, xs, max_beam_size=5):
    n_examples = len(xs)
    n_beams = [1 for _ in xs]
    indices = [list(range(x.shape[0])) for x in xs]
    removed_indices = [[] for _ in xs]

    final_xs = [[x.tolist()] for x in xs]
    final_removed = [[[]] for _ in xs]
    final_length = [x.shape[0] for x in xs]

    with chainer.using_config('train', False):
        ys_0 = model.predict(xs, argmax=True)

    while True:
        xs, n_beams, indices, removed_indices = remove_one(
                model, xs, n_beams, indices, removed_indices, max_beam_size)
        with chainer.using_config('train', False):
            ys = model.predict(xs, argmax=True)

        new_n_beams = []
        start = 0
        remained_indices = []
        for example_idx in range(n_examples):
            if n_beams[example_idx] == 0:
                new_n_beams.append(0)
                continue

            cnt = 0
            for i in range(start, start + n_beams[example_idx]):
                if not ys[i] == ys_0[example_idx]:
                    continue

                x = xs[i].tolist()
                if len(x) < final_length[example_idx]:
                    final_length[example_idx] = len(x)
                    final_removed[example_idx] = [removed_indices[i]]
                    final_xs[example_idx] = [x]
                elif len(x) == final_length[example_idx]:
                    if x not in final_xs[example_idx]:
                        final_xs[example_idx].append(x)
                        final_removed[example_idx].append(removed_indices[i])
                if len(x) == 1:
                    # only eos left
                    continue

                remained_indices.append(i)
                cnt += 1
            new_n_beams.append(cnt)
            start += n_beams[example_idx]

        if len(remained_indices) == 0:
            break

        xs = [xs[i] for i in remained_indices]
        indices = [indices[i] for i in remained_indices]
        removed_indices = [removed_indices[i] for i in remained_indices]
        n_beams = new_n_beams
    return final_xs, final_removed


class Bunch(object):

    def __init__(self, adict):
        self.__dict__.update(adict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True)
    args_dir = os.path.join(parser.parse_args().load, 'args.json')
    with open(args_dir) as f:
        args = Bunch(json.load(f))

    print(json.dumps(args.__dict__, indent=2))

    # Load a dataset
    with open(args.vocab_path) as f:
        vocab = json.load(f)

    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(
            vocab=vocab, char_based=args.char_based)
    elif args.dataset == 'sst':
        train, test, vocab = text_datasets.get_sst(
            char_based=args.char_based)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            vocab=vocab,
            fine_grained=args.dataset.endswith('.fine'),
            char_based=args.char_based)
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args.dataset, vocab=vocab, char_based=args.char_based)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))
    # i_to_word = {v: k for k, v in vocab.items()}

    # FIXME
    args.batchsize = 64
    max_beam_size = 5

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    if args.dataset == 'snli':
        model = nets.DoubleMaxClassifier(
                n_layers=args.layer, n_vocab=len(vocab),
                n_units=args.unit, n_class=n_class, dropout=args.dropout)
    else:
        model = nets.SingleMaxClassifier(
                n_layers=args.layer, n_vocab=len(vocab),
                n_units=args.unit, n_class=n_class, dropout=args.dropout)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    chainer.serializers.load_npz(args.model_path, model)

    checkpoint = []
    for batch_idx, batch in enumerate(tqdm(test_iter)):
        # if batch_idx > 10:
        #     break

        batch = convert_seq(batch, device=args.gpu)
        xs = batch['xs']
        reduced_xs, removed_indices = get_rawr(
                model, xs, max_beam_size=max_beam_size)

        xp = cupy.get_array_module(*xs)
        n_finals = [len(r) for r in reduced_xs]
        reduced_xs = list(itertools.chain(*reduced_xs))
        removed_indices = list(itertools.chain(*removed_indices))
        reduced_xs = [xp.asarray(x) for x in reduced_xs]
        reduced_xs = convert_seq(reduced_xs, device=args.gpu, with_label=False)
        with chainer.using_config('train', False):
            ss_0 = xp.asnumpy(model.predict(xs, softmax=True))
            ss_1 = xp.asnumpy(model.predict(reduced_xs, softmax=True))
            ys_0 = np.argmax(ss_0, axis=1)
            ys_1 = np.argmax(ss_1, axis=1)

        start = 0
        for example_idx in range(len(xs)):
            oi = xs[example_idx].tolist()  # original input
            op = int(ys_0[example_idx])  # original predictoin
            oos = ss_0[example_idx]  # original output distribution
            label = int(batch['ys'][example_idx])
            checkpoint.append([])
            for i in range(start, start + n_finals[example_idx]):
                ri = reduced_xs[i].tolist()
                rp = int(ys_1[i])
                rs = ss_1[i]
                rr = removed_indices[i]
                entry = {'original_input': oi,
                         'reduced_input': ri,
                         'original_prediction': op,
                         'reduced_prediction': rp,
                         'original_scores': oos,
                         'reduced_scores': rs,
                         'removed_indices': rr,
                         'label': label}
                checkpoint[-1].append(entry)
    with open(os.path.join(args.out, 'rawr_dev.pkl'), 'wb') as f:
        pickle.dump(checkpoint, f)


if __name__ == '__main__':
    main()
