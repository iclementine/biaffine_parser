import conllu
import dynet as dy
import numpy as np
import pytext
import copy
import logging
import time
import os
import sys
import pickle
from gensim.models import KeyedVectors

import treebank_toolkit as tbtk
from argconfigparse import parser_arg_cfg
from deepbiaffine import DeepBiaffine
from test import test
from lib.utils import word_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')


args, config = parser_arg_cfg()

# build ConllSent from files in conll format 
partitions = ["train_data", "dev_data", "test_data"]
files = [config.get("data", x) for x in partitions]
corpus_files = [open(x) for x in files]
sent_collections = [[tbtk.ConllSent.from_conllu(sent) for sent in conllu.parse(f.read())] for f in corpus_files]

# construc fields & dataset
f_form = pytext.data.Field(lower=True, tokenize=list, include_lengths=True)
f_upos = pytext.data.Field(tokenize=list)
f_head = pytext.data.Field(use_vocab=False, pad_token=0)
f_deprel = pytext.data.Field(tokenize=list)
fields = [('form', f_form), ('upos', f_upos), ('head', f_head), ('deprel', f_deprel)]

example_collections = [[pytext.data.Example.fromlist([sent.form, sent.upos, sent.head, sent.deprel], fields) for sent in sents] for sents in sent_collections]

train_data = pytext.data.Dataset(example_collections[0], fields)
dev_data = pytext.data.Dataset(example_collections[1], fields)
test_data = pytext.data.Dataset(example_collections[2], fields)

# build vocab from dataset for fields that use vocab
f_form.build_vocab(train_data, min_freq=2)
# snapshot how many words in training dataset, cause this vocab would be extended afterwareds
words_in_train = len(f_form.vocab.stoi) 
logging.info("{} words in training dataset".format(words_in_train))
f_upos.build_vocab(train_data)
f_deprel.build_vocab(train_data)

# build a vocab for pretrained word vectors
# use it to extend vocab_form
# load vectors for vocab_form
word_dims = config.getint("network", "word_dims")
f_pre = pytext.data.Field(sequential=False)
if "word_emb" in config["data"]:
    pret_path = config.get("data", "word_emb")
    pret_wvs = KeyedVectors.load_word2vec_format(pret_path, binary=False)
    assert(word_dims == pret_wvs.vectors.shape[1])
    f_pre.build_vocab(pret_wvs.vocab.keys())
    f_form.vocab.extend(f_pre.vocab)
    logging.info("{} extra words added".format(len(f_form.vocab.stoi) - words_in_train))
    stoi = {s:v.index for s, v in pret_wvs.vocab.items()}
    f_form.vocab.set_vectors(stoi, pret_wvs.vectors, word_dims)
    f_form.vocab.vectors /= np.std(f_form.vocab.vectors) # a special and effective trick to keep variance
else:
    f_form.vocab.vectors = None

# build iterators to specify how to iterate data
train_iters = config.getint("run", "train_iters")
train_batch_size = config.getint("run", "train_batch_size")
test_batch_size = config.getint("run", "test_batch_size")
validate_every = config.getint("run", "validate_every")
save_after = config.getint("run", "save_after")

train_it = pytext.data.BucketIterator(train_data, train_batch_size, sort_key=lambda x: len(x.form), shuffle=True, repeat=True,  batch_size_fn=word_count)
dev_it = pytext.data.Iterator(dev_data, test_batch_size, train=False, sort=False, batch_size_fn=word_count)
test_it = pytext.data.Iterator(test_data, test_batch_size, train=False, sort=False, batch_size_fn=word_count)

# get some configs and build model
lstm_layers = config.getint("network", "lstm_layers")
tag_dims = config.getint("network", "tag_dims")
dropout_emb = config.getfloat("network", "dropout_emb")
lstm_hiddens = config.getint("network", "lstm_hiddens")
dropout_lstm_input = config.getfloat("network", "dropout_lstm_input")
dropout_lstm_hidden = config.getfloat("network", "dropout_lstm_hidden")
mlp_arc_size = config.getint("network", "mlp_arc_size")
mlp_rel_size = config.getint("network", "mlp_rel_size")
dropout_mlp = config.getfloat("network", "dropout_mlp")

model = dy.ParameterCollection()
biaffine_parser = DeepBiaffine(model, f_form.vocab, word_dims, words_in_train, dropout_emb, f_upos.vocab, tag_dims, f_deprel.vocab, lstm_layers, lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp)

# trainer
learning_rate = config.getfloat("trainer", "learning_rate")
beta1 = config.getfloat("trainer", "beta_1")
beta2 = config.getfloat("trainer", "beta_2")
epsilon = config.getfloat("trainer", "epsilon")

trainer = dy.AdamTrainer(model, learning_rate , beta1, beta2, epsilon)
decay = config.getfloat("trainer", "decay")
decay_steps = config.getfloat("trainer", "decay_steps")

def update_parameters(trainer, global_step):
    trainer.learning_rate = learning_rate * decay ** (global_step / decay_steps)
    trainer.update()

# save paths
save_dir = config.get("save", "save_dir")
save_model_path = config.get("save", "save_model_path")
save_config_path = config.get("save", "save_config_file")
with open(save_config_path, 'wt') as f:
    config.write(f)
records_path = config.get("save", "records_path")

# load paths for testing
load_dir = config.get("load", "load_dir")
load_model_path = config.get("load", "load_model_path")

# train
epoch = 0
best_UAS = 0.
history = lambda x, y : open(records_path,'at').write('%.2f %.2f\n'%(x,y))

if not args.test:
    for global_step, batch in enumerate(train_it, 1):
        if global_step > train_iters: break
        dy.renew_cg()
        arc_accuracy, rel_accuracy, overall_accuracy, loss = biaffine_parser.run(batch.form[0], batch.form[1], batch.upos, batch.head, batch.deprel)
        loss_value = loss.scalar_value()
        loss.backward()
        sys.stdout.write("Step #%d: Acc: arc %.2f, rel %.2f, overall %.2f, loss %.3f\r\r" %(global_step, arc_accuracy, rel_accuracy, overall_accuracy, loss_value))
        sys.stdout.flush()
        update_parameters(trainer, global_step)

        if global_step % validate_every == 0:
            print('\nTest on development set')
            LAS, UAS = test(biaffine_parser, dev_it, f_deprel.vocab, files[1], os.path.join(save_dir, 'valid_tmp'))
            history(LAS, UAS)
            if global_step > save_after and UAS > best_UAS:
                best_UAS = UAS
                dy.save(save_model_path, [biaffine_parser])

# load best model to test on the test set
biaffine_parser.load(load_model_path)
test_output = config.get("load", "test_output")
LAS, UAS = test(biaffine_parser, test_it, f_deprel.vocab, files[2], test_output)
logging.info("output file saved at {}".format(test_output))
logging.info("Final Test: LAS: {} UAS: {}".format(LAS, UAS))

