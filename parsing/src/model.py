import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from utils import normalize


class DependencyParser(nn.Module):
    def __init__(self, w_emb_dim, w_i_counter, w2i, n_lstm_l,
                 mlp_hid_dim, lstm_hid_dim, device, ex_w_vec=None, ext_emb_w2i=None, n_arc_relations=1, alpha=0.25):
        super(DependencyParser, self).__init__()

        self.w_i_counter = w_i_counter
        self.w2i = w2i
        self.ex_emb_w2i = ext_emb_w2i
        self.ex_emb_flag = False
        self.alpha = alpha
        self.device = device

        # embedding layers initialization
        self.word_embeddings = nn.Embedding(len(w2i), w_emb_dim)
        if ex_w_vec is not None and ext_emb_w2i is not None:  # Use external word embeddings
            self.ex_emb_flag = True
            self.ex_word_emb = nn.Embedding.from_pretrained(ex_w_vec, freeze=True)

        # LSTM dimensions
        input_dim = w_emb_dim
        if self.ex_emb_flag:
            input_dim += ex_w_vec.size(-1)

        # bidirectional LSTM initialization
        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=lstm_hid_dim,
                               num_layers=n_lstm_l,
                               bidirectional=True,
                               batch_first=True)

        # arc scorer initialization
        self.hid_arc_h = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_m = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_arc_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        DependencyParser.param_init(self.hid_arc_bias)

        self.slp_out_arc = SLP(hidden_size=mlp_hid_dim)

        # arc relations MLP initialization
        self.hid_rel_h = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_m = nn.Linear(2 * lstm_hid_dim, mlp_hid_dim, bias=False)
        self.hid_rel_bias = nn.Parameter(torch.empty((1, mlp_hid_dim)))
        DependencyParser.param_init(self.hid_rel_bias)

        self.slp_out_rel = SLP(hidden_size=mlp_hid_dim,
                               n_relations=n_arc_relations,
                               for_relations=True)

        # initialize model weights
        for name, module in self.named_children():
            if name == 'ex_word_emb':
                continue
            else:
                DependencyParser.modules_init(module)

    def forward(self, sentence, word_dropout=False):

        n_words = len(sentence)

        s_w_i = torch.tensor([self.w2i.get(normalize(w[0]), 0) for w in sentence]).to(self.device).unsqueeze(0)

        # word dropout
        if word_dropout:
            unk_probs = torch.tensor([1 - (self.alpha / (self.w_i_counter[w_i.item()] + self.alpha))
                                      for w_i in s_w_i.squeeze()]).to(self.device)
            do_w_ber_sample = Bernoulli(probs=unk_probs).sample().int()
            do_s_w_i = s_w_i * do_w_ber_sample
            w_emb_tensor = self.word_embeddings(do_s_w_i)

            do_ext_emb_probs = torch.abs(do_w_ber_sample - 1) * 0.5
            do_ext_emb_ber_sample = Bernoulli(probs=do_ext_emb_probs).sample().int()
        else:
            w_emb_tensor = self.word_embeddings(s_w_i)

        # embeddings concatenation
        if self.ex_emb_flag:
            ex_s_w_i = torch.tensor(
                [self.ex_emb_w2i.get(w[0], self.ex_emb_w2i.get(normalize(w[0]), 0)) for w in sentence]).to(
                self.device).unsqueeze(0)
            if word_dropout:
                ex_do_s_w_i = ex_s_w_i * (do_ext_emb_ber_sample + do_w_ber_sample)
                ex_word_em_tensor = self.ex_word_emb(ex_do_s_w_i)
            else:
                ex_word_em_tensor = self.ex_word_emb(ex_s_w_i)
            input_vectors = torch.cat((w_emb_tensor, ex_word_em_tensor), dim=-1)
        else:
            input_vectors = w_emb_tensor

        self.encoder.flatten_parameters()
        hidden_vectors, _ = self.encoder(input_vectors)

        heads, mods = [], []
        for h, m in product(range(0, n_words), range(0, n_words)):
            heads.append(h)
            mods.append(m)

        # score all possible arcs
        arc_h_scores = self.hid_arc_h(hidden_vectors)
        arc_m_scores = self.hid_arc_m(hidden_vectors)
        arc_scores = self.slp_out_arc(arc_h_scores[0, heads, :] + arc_m_scores[0, mods, :] + self.hid_arc_bias)
        arc_scores = arc_scores.view(n_words, n_words)

        # score relations for all possible arcs
        rel_h_scores = self.hid_rel_h(hidden_vectors)
        rel_m_scores = self.hid_rel_m(hidden_vectors)
        rel_scores = self.slp_out_rel(rel_h_scores[0, heads, :] + rel_m_scores[0, mods, :] + self.hid_rel_bias)
        rel_scores = rel_scores.view(n_words, n_words, -1)

        return arc_scores, rel_scores

    @staticmethod
    def modules_init(m):
        if isinstance(m, nn.Embedding):
            emb_bound = math.sqrt(3. / m.embedding_dim)
            nn.init.uniform_(m.weight, -emb_bound, emb_bound)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if 'bias' in name:
                    h_dim = p.shape[-1] // 4
                    nn.init.constant_(p[: h_dim], 0.)
                    nn.init.constant_(p[h_dim: 2 * h_dim], 0.5)  # forget gate bias initialization
                    nn.init.constant_(p[2 * h_dim:], 0.)
                else:
                    nn.init.xavier_uniform_(p)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, SLP):
            nn.init.xavier_uniform_(m.fc.weight)
            if m.fc.bias is not None:
                DependencyParser.param_init(m.fc.bias)

    @staticmethod
    def param_init(p):
        bound = math.sqrt(3. / p.shape[-1])
        nn.init.uniform_(p, -bound, bound)


class SLP(nn.Module):
    def __init__(self, hidden_size, n_relations=None, for_relations=False):
        super(SLP, self).__init__()

        self.activation = torch.tanh

        # initialize MLP layers
        if for_relations and n_relations:
            self.fc = nn.Linear(hidden_size, n_relations)
        else:
            self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, edge):
        x = self.activation(edge)
        output = self.fc(x)
        return output


class Decoder(nn.Module):
    def __init__(self, hid_dim, out_dim, ext_emb_w2i, ex_w_vec, device, n_lstm_l=1):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.ex_emb_w2i = ext_emb_w2i
        self.device = device

        # load pretrained embedding weights and freeze them
        self.embedding = nn.Embedding.from_pretrained(ex_w_vec, freeze=True)
        self.lstm = nn.LSTM(input_size=ex_w_vec.size(-1),
                            hidden_size=self.hid_dim,
                            num_layers=n_lstm_l,
                            batch_first=True)

        self.head_fc = nn.Linear(self.hid_dim, self.out_dim)
        self.mod_fc = nn.Linear(self.hid_dim, self.out_dim)
        self.curr_fc = nn.Linear(self.hid_dim, self.out_dim)
        self.output_fc = nn.Linear(self.out_dim, len(ext_emb_w2i))

        # initialize model weights
        for name, module in self.named_children():
            Decoder.modules_init(module)

    def forward(self, sentence, dep_tree):

        s_length = dep_tree.shape[0] - 1

        s_w_i = torch.tensor([self.ex_emb_w2i.get(w[0], self.ex_emb_w2i.get(normalize(w[0]), 0)) for w in sentence[1:]])\
            .to(self.device)

        head_scores = torch.empty(size=(s_length, self.out_dim)).to(self.device)
        mod_scores = torch.empty(size=(s_length, self.out_dim)).to(self.device)

        h_i, c_i = self.init_hidden()
        w_i = self.embedding(torch.tensor([self.ex_emb_w2i[sentence[0][0]]], dtype=torch.long).to(self.device)).unsqueeze(0)
        rec_loss = 0.
        for i in range(s_length):

            self.lstm.flatten_parameters()
            hidden_vec, (h_i, c_i) = self.lstm(w_i, (h_i, c_i))

            head_scores[i, :] = self.head_fc(hidden_vec)
            mod_scores[i, :] = self.mod_fc(hidden_vec)

            i_mod_dep = dep_tree[:i, i]
            i_head_dep = dep_tree[i, :i]

            heads_gcn_term = torch.sum(i_mod_dep.view(-1, 1) * head_scores[:i, :].clone(), dim=0)
            mods_gcn_term = torch.sum(i_head_dep.view(-1, 1) * mod_scores[:i, :].clone(), dim=0)

            gcn_output = torch.tanh(self.curr_fc(hidden_vec) + heads_gcn_term + mods_gcn_term)

            log_probs = F.log_softmax(self.output_fc(gcn_output).squeeze(), dim=-1)
            rec_loss += -log_probs[s_w_i[i]]
            w_i = self.embedding(s_w_i[i]).view(1, 1, -1)

        return rec_loss / s_length

    @staticmethod
    def modules_init(m):
        if isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if 'bias' in name:
                    h_dim = p.shape[-1] // 4
                    nn.init.constant_(p[: h_dim], 0.)
                    nn.init.constant_(p[h_dim: 2 * h_dim], 0.5)  # forget gate bias initialization
                    nn.init.constant_(p[2 * h_dim:], 0.)
                else:
                    nn.init.xavier_uniform_(p)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                Decoder.param_init(m.bias)

    @staticmethod
    def param_init(p):
        bound = math.sqrt(3. / p.shape[-1])
        nn.init.uniform_(p, -bound, bound)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(1, 1, self.hid_dim),
                weight.new_zeros(1, 1, self.hid_dim))