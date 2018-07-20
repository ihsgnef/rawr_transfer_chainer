import os
import json
import pickle
import random
import numpy as np
import itertools
import argparse
from tqdm import tqdm, tqdm_notebook
from plotnine import *
import pandas as pd

import cupy
import chainer
from chainer import functions as F

from nlp_utils import convert_seq, convert_snli_seq
from utils import setup_model

from run_dknn import DkNN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model-setup', required=True,
                    help='Model setup dictionary.')
parser.add_argument('--lsh', action='store_true', default=False,
                    help='If true, uses locally sensitive hashing \
                          (with k=10 NN) for NN search.')
args = parser.parse_args(['--model-setup', 'result/snli_bilstm/args.json'])

model, train, test, vocab, setup = setup_model(args)
if setup['dataset'] == 'snli':
    converter = convert_snli_seq
else:
    converter = convert_seq

# FIXME
args.batchsize = 64
max_beam_size = 5

with open(os.path.join(setup['save_path'], 'calib.json')) as f:
    calibration_idx = json.load(f)

calibration = [train[i] for i in calibration_idx]
train = [x for i, x in enumerate(train) if i not in calibration_idx]
train = random.sample(train, 10000)

dknn = DkNN(model)
dknn.build(train, batch_size=setup['batchsize'], converter=converter, device=args.gpu)
dknn.calibrate(calibration, batch_size=setup['batchsize'], converter=converter, device=args.gpu)

df = {'confidence': [], 'data': [], 'label': [], 'type': []}

og_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

n_batches = len(test) // args.batchsize
for batch in tqdm_notebook(og_iter, total=n_batches):
    batch = converter(batch, device=args.gpu)
    knn_pred, knn_cred, knn_conf, reg_pred, reg_conf = dknn.predict(batch['xs'], calibrated=True, snli=True)
    df['confidence'].extend(reg_conf)
    df['data'].extend('original' for _ in reg_conf)
    df['label'].extend(reg_pred)
    df['type'].extend('reg_conf' for _ in reg_conf)
    
    df['confidence'].extend(knn_conf)
    df['data'].extend('original' for _ in knn_conf)
    df['label'].extend(knn_pred)
    df['type'].extend('knn_conf' for _ in knn_conf)
    
    df['confidence'].extend(knn_cred)
    df['data'].extend('original' for _ in knn_cred)
    df['label'].extend(knn_pred)
    df['type'].extend('knn_cred' for _ in knn_cred)

ckp = pickle.load(open('rawr_dev.pkl', 'rb'))
rd_test =  [x[0]['reduced_input'] for x in ckp]
rd_iter = chainer.iterators.SerialIterator(rd_test, args.batchsize, repeat=False, shuffle=False)

n_batches = len(rd_test) // args.batchsize
for batch in tqdm_notebook(rd_iter, total=n_batches):
    batch = converter(batch, device=args.gpu, with_label=False)
    knn_pred, knn_cred, knn_conf, reg_pred, reg_conf = dknn.predict(batch, calibrated=True, snli=True)
    df['confidence'].extend(reg_conf)
    df['data'].extend('reduced' for _ in reg_conf)
    df['label'].extend(reg_pred)
    df['type'].extend('reg_conf' for _ in reg_conf)
    
    df['confidence'].extend(knn_conf)
    df['data'].extend('reduced' for _ in knn_conf)
    df['label'].extend(knn_pred)
    df['type'].extend('knn_conf' for _ in knn_conf)
    
    df['confidence'].extend(knn_cred)
    df['data'].extend('reduced' for _ in knn_cred)
    df['label'].extend(knn_pred)
    df['type'].extend('knn_cred' for _ in knn_cred)

df = pd.DataFrame(df)
(
    ggplot(df) +
    aes(x='confidence', color='data', fill='data') +
    geom_density(alpha=.45) +
    facet_grid('type ~ label')
)