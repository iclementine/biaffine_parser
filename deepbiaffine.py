import dynet as dy
import numpy as np

from lib.components import biLSTM, leaky_relu, bilinear, orthonormal_initializer, orthonormal_VanillaLSTMBuilder
from decode import arc_argmax, rel_argmax

class DeepBiaffine(object):
    def __init__(self, model, vocab_form, d_form, v_train, dropout_emb, vocab_pos, d_pos, vocab_deprel, layers, d_lstm, dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp):
        spc = model.add_subcollection("deepbiaffine")
        
        # lookup parameters
        self.vocab_form = vocab_form
        self.vocab_pos = vocab_pos
        self.vocab_deprel = vocab_deprel
        self.v_train = v_train
        self.dropout_emb = dropout_emb
        self.e_form = spc.lookup_parameters_from_numpy(np.random.randn(v_train, d_form) if vocab_form.vectors is not None else np.zeros(v_train, d_form))
        self.e_ext = spc.lookup_parameters_from_numpy(vocab_form.vectors) if vocab_form.vectors is not None else None
        self.e_tag = spc.add_lookup_parameters((len(vocab_pos), d_pos))
        
        # lstm builders, typically
        self.lstm_builders = []
        f = orthonormal_VanillaLSTMBuilder(1, d_form+d_pos, d_lstm, spc)
        b = orthonormal_VanillaLSTMBuilder(1, d_form+d_pos, d_lstm, spc)
        self.lstm_builders.append((f, b))
        for i in range(layers - 1):
            f = orthonormal_VanillaLSTMBuilder(1, 2*d_lstm, d_lstm, spc)
            b = orthonormal_VanillaLSTMBuilder(1, 2*d_lstm, d_lstm, spc)
            self.lstm_builders.append((f,b))
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden
        
        # things are cated togather to speed up
        mlp_size = mlp_arc_size + mlp_rel_size
        W = orthonormal_initializer(mlp_size, 2*d_lstm)
        self.mlp_dep_W = spc.parameters_from_numpy(W)
        self.mlp_head_W = spc.parameters_from_numpy(W)
        self.mlp_dep_b = spc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_head_b = spc.add_parameters((mlp_size,), init=dy.ConstInitializer(0.))
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp
        
        self.arc_W = spc.add_parameters((mlp_arc_size, mlp_arc_size + 1), init=dy.ConstInitializer(0.))
        self.rel_W = spc.add_parameters((len(vocab_deprel) * (mlp_rel_size + 1), mlp_rel_size + 1), init=dy.ConstInitializer(0.))
        
        self.spec = (vocab_form, d_form, v_train, dropout_emb, vocab_pos, d_pos, vocab_deprel, layers, d_lstm, dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size, dropout_mlp)
        self.pc = spc
        
    def generate_emb_msk(self, seq_len, batch_size):
        ret = []
        for i in range(seq_len):
            word_mask = np.random.binomial(1, 1. - self.dropout_emb, batch_size).astype(np.float32)
            tag_mask = np.random.binomial(1, 1. - self.dropout_emb, batch_size).astype(np.float32)
            scale = 3. / (2.*word_mask + tag_mask + 1e-12)
            word_mask *= scale
            tag_mask *= scale
            word_mask = dy.inputTensor(word_mask, batched = True)
            tag_mask = dy.inputTensor(tag_mask, batched = True)
            ret.append((word_mask, tag_mask))
        return ret 
    
    @staticmethod
    def dynet_flatten_numpy(ndarray):
        # inputs, targets: seq_len x batch_size
        return np.reshape(ndarray, (-1,), 'F')
    
    def run(self, word_inputs, lengths, tag_inputs, arc_targets = None, rel_targets = None, isTrain = True):
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        mask = (np.broadcast_to(np.reshape(np.arange(seq_len), (seq_len, 1)), (seq_len, batch_size)) < lengths).astype(np.float32)
        mask[0] = 0.
        num_tokens = int(np.sum(mask))
    
        if isTrain or arc_targets is not None:
            mask_1D = self.dynet_flatten_numpy(mask)
            # batched here means that the last dim is treated as batch dimension, both in input and output
            mask_1D_tensor = dy.inputTensor(mask_1D, batched = True)
    
        # TODO: 注意 _words_in_train
        # 两个 embedding 相加, [Expression of dim=((embedding_dim,), batch_size)] * seq_len
        if self.e_ext is not None:
            word_embs = [dy.lookup_batch(self.e_form, np.where(w < self.v_train, w, self.vocab_form.stoi["<unk>"])) + dy.lookup_batch(self.e_ext, w, update=False) for w in word_inputs] # 两个 embedding 相加 [Expression] * seq_len
        else:
            word_embs = [dy.lookup_batch(self.e_form, np.where(w < self.v_train, w, self.vocab_form.stoi["<unk>"])) for w in word_inputs]
        tag_embs = [dy.lookup_batch(self.e_tag, pos) for pos in tag_inputs]
    
        if isTrain:
            emb_masks = self.generate_emb_msk(seq_len, batch_size)
            emb_inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos,posm)]) for w, pos, (wm, posm) in zip(word_embs,tag_embs,emb_masks)]
        else:
            emb_inputs = [dy.concatenate([w, pos]) for w, pos in zip(word_embs,tag_embs)]
    
        top_recur = dy.concatenate_cols(biLSTM(self.lstm_builders, emb_inputs, batch_size, self.dropout_lstm_input if isTrain else 0., self.dropout_lstm_hidden if isTrain else 0.))
        if isTrain:
            # drop some dim for lstm_output for all words, all sentences
            top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)
    
        dep = leaky_relu(dy.affine_transform([self.mlp_dep_b, self.mlp_dep_W, top_recur]))
        head = leaky_relu(dy.affine_transform([self.mlp_head_b, self.mlp_head_W, top_recur]))
        if isTrain:
            dep, head= dy.dropout_dim(dep, 1, self.dropout_mlp), dy.dropout_dim(head, 1, self.dropout_mlp) 
            # drop dim k means, it is possible that the whole dim k is set to zeros
            # for matrix with batch, ((R, C), B)
            # drop dim 0 means drop some cols, drop dim 1 means drop some rows
            # drop 2 means drop some batches, and it only supports for Tensor with rank <=3
    
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        arc_logits = bilinear(dep_arc, self.arc_W, head_arc, self.mlp_arc_size, seq_len, batch_size, num_outputs= 1, bias_x = True, bias_y = False)
        # (#head x #dep) x batch_size
    
        flat_arc_logits = dy.reshape(arc_logits, (seq_len,), seq_len * batch_size) 
        # flatten it to compute loss
        # (#head ) x (#dep x batch_size)
        
        arc_preds = np.reshape(arc_logits.npvalue().argmax(0), (seq_len, batch_size))
        # seq_len x batch_size
        # here if an Expression's batch size is 1
        # npvalue() will drop the batch dimension
        # so add it back if needed
    
        if isTrain or arc_targets is not None:
            # tarin it in a neg log likelihood fashion, but enforce tree constraint when testing
            arc_correct = np.equal(arc_preds, arc_targets).astype(np.float32) * mask 
            # mask is used to filter <pad>'s out in summing loss
            arc_accuracy = np.sum(arc_correct) / num_tokens
            targets_1D = self.dynet_flatten_numpy(arc_targets)
            losses = dy.pickneglogsoftmax_batch(flat_arc_logits, targets_1D)
            arc_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
    
        if not isTrain:
            arc_probs = np.transpose(np.reshape(dy.softmax(flat_arc_logits).npvalue(), (seq_len, seq_len, batch_size), 'F'))
            # #batch_size x #dep x #head, transpose reverse all, and since layout has changed, it's totally fine

        rel_logits = bilinear(dep_rel, self.rel_W, head_rel, self.mlp_rel_size, seq_len, batch_size, num_outputs = len(self.vocab_deprel), bias_x = True, bias_y = True)
        # (#head x rel_size x #dep) x batch_size
    
        flat_rel_logits = dy.reshape(rel_logits, (seq_len, len(self.vocab_deprel)), seq_len * batch_size)
        # (#head x rel_size) x (#dep x batch_size)
    
        partial_rel_logits = dy.pick_batch(flat_rel_logits, targets_1D if isTrain else self.dynet_flatten_numpy(arc_preds))
        # (rel_size) x (#dep x batch_size)
    
        if isTrain or arc_targets is not None:
            rel_preds = partial_rel_logits.npvalue().argmax(0)
            targets_1D = self.dynet_flatten_numpy(rel_targets)
            rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D # 这里的形状如此， 需要用 mask1d
            rel_accuracy = np.sum(rel_correct) / num_tokens
            losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
            rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
    
        if not isTrain:
            rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), (len(self.vocab_deprel), seq_len, seq_len, batch_size), 'F'))
            # batch_size x #dep x #head x #nclasses
    
        if isTrain or arc_targets is not None:
            loss = arc_loss + rel_loss
            correct = rel_correct * self.dynet_flatten_numpy(arc_correct)
            overall_accuracy = np.sum(correct) / num_tokens 
    
        if isTrain:
            return arc_accuracy, rel_accuracy, overall_accuracy, loss
    
        outputs = []
    
        for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
            # parse sentences one by ones
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = arc_argmax(arc_prob, sent_len, msk)
            rel_prob = rel_prob[np.arange(len(arc_pred)),arc_pred]
            rel_pred = rel_argmax(rel_prob, sent_len, self.vocab_deprel, "root" if "root" in self.vocab_deprel.stoi else "ROOT")
            outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len])) # w_0 is <roor>
        assert(len(outputs) == batch_size)
    
        if arc_targets is not None:
            return arc_accuracy, rel_accuracy, overall_accuracy, outputs
        return outputs

    def param_collection(self):
        return self.pc
    
    @staticmethod
    def from_spec(spec, model):
        # use high level save and load
        return DeepBiaffine(model, *spec)

    #def save(self, save_path):
        #self.pc.save(save_path)
        
    def load(self, load_path):
        self.pc.populate(load_path + ".data")

        
