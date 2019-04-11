import sys
import os
import logging
from functools import reduce
import time, os, pickle
import dynet as dy
import pytext
import conllu
from deepbiaffine import DeepBiaffine
from argconfigparse import parser_arg_cfg
from lib.utils import word_count
import treebank_toolkit as tbtk
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

def test(parser, test_it, vocab_deprel, test_file, output_file):
    results = []
    idx = 0
    for batch in test_it:
        dy.renew_cg()
        outputs = parser.run(batch.form[0], batch.form[1], batch.upos, isTrain=False)
        for output in outputs:
            results.append(output)
    arcs = reduce(lambda x, y: x + y, [ list(result[0]) for result in results ])
    rels = reduce(lambda x, y: x + y, [ list(result[1]) for result in results ])
    idx = 0
    with open(test_file, 'rt') as f:
        with open(output_file, 'wt') as fo:
            for line in f.readlines():
                info = line.strip().split()
                if info:
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[6] = str(arcs[idx])
                    info[7] = vocab_deprel.itos[rels[idx]]
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')

    os.system('perl eval.pl -q -b -g %s -s %s -o tmp' % (test_file, output_file))
    os.system('tail -n 3 tmp > score_tmp')
    LAS, UAS = [float(line.strip().split()[-2]) for line in open('score_tmp').readlines()[:2]]
    print('LAS %.2f, UAS %.2f'%(LAS, UAS))
    #os.system('rm tmp score_tmp')
    return LAS, UAS

if __name__ == "__main__":
    args, config = parser_arg_cfg()
    
    # load model with high level save/load API
    load_model_path = config.get("load", "load_model_path")
    pc = dy.ParameterCollection()
    biaffine_parser = dy.load(load_model_path, pc)[0]
    

    # get vocabs from the model, which is then used for create fields
    vocab_form, vocab_upos, vocab_deprel = biaffine_parser.vocab_form, biaffine_parser.vocab_pos, biaffine_parser.vocab_deprel
    
    # create data fields for building test dataset, vocabs is extracted from model 
    # instead of built from data itself
    f_form = pytext.data.Field(lower=True, tokenize=list, include_lengths=True)
    f_upos = pytext.data.Field(tokenize=list)
    f_head = pytext.data.Field(use_vocab=False, pad_token=0)
    f_deprel = pytext.data.Field(tokenize=list)
    f_form.vocab = vocab_form
    f_upos.vocab = vocab_upos
    f_deprel.vocab = vocab_deprel
    
    # build test dataset
    test_path = config.get("data", "test_data")
    test_batch_size = config.getint("run", "test_batch_size")
    with open(test_path, 'rt') as f:
        test_sentences = [tbtk.ConllSent.from_conllu(sent) for sent in conllu.parse(f.read())]
    fields = [('form', f_form), ('upos', f_upos), ('head', f_head), ('deprel', f_deprel)]
    examples = [pytext.data.Example.fromlist([sent.form, sent.upos, sent.head, sent.deprel], fields) for sent in test_sentences]
    test_data = pytext.data.Dataset(examples, fields)
    test_it = pytext.data.Iterator(test_data, test_batch_size, train=False, sort=False, batch_size_fn=word_count)
    
    # test
    test_output = config.get("load", "test_output")
    LAS, UAS = test(biaffine_parser, test_it, f_deprel.vocab, test_path, test_output)
    logging.info("output file saved at {}".format(test_output))
    logging.info("Final Test: LAS: {} UAS: {}".format(LAS, UAS))
    
