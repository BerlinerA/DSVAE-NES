import io
import pickle
import os
import re
import math
from collections import Counter

import torch
import numpy as np
from nltk.parse import DependencyGraph

from constants import *

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def save_obj(obj, dir, name):
    with open(os.path.join(dir, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_path):
    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def vocab(data_paths, root_sym=ROOT_SYMBOL, rel_root_sym=RROOT_SYMBOL):

    words_count = Counter()
    unique_rels = {rel_root_sym, }

    for path in data_paths:
        sentences_count = 0
        with open(path, 'r') as fh:

            for line in fh:
                if line.startswith('#'):
                    continue
                if line == '\n':
                    sentences_count += 1
                else:
                    parsed_line = line.strip().split('\t')

                    word_loc, word, rel = parsed_line[0], normalize(parsed_line[1]), parsed_line[7]
                    if len(word.split(' ')) > 1:
                        word = '_'.join(word.split(' '))
                    if word_loc == '1':
                        words_count[root_sym] += 1

                    words_count[word] += 1
                    unique_rels.add(rel)

    w2i = {word: i for i, word in enumerate(words_count, 1)}

    return words_count, sorted(list(unique_rels)), w2i


def load_pretrained_word_embed(embed_path, vocab, unk_sym=UNK_SYMBOL, root_sym=ROOT_SYMBOL, init_sym=INIT_SYMBOL, dtype=np.float64):
    ex_w2i = {unk_sym: 0, root_sym: 1, init_sym: 2}
    wiki_multi_flag = 'cc.' in embed_path
    vector_dim = None
    word_vectors = []
    line_i = 3 if not wiki_multi_flag else 2

    with io.open(embed_path,  'r', encoding='utf-8', newline='\n', errors='ignore') as fh:
        for line in fh:
            if line_i == 2 and wiki_multi_flag:
                line_i += 1
                vector_dim = int(line.split()[1])
                continue
            parsed_line = line.rstrip().split(' ')
            word, word_vector = parsed_line[0], np.array([float(val) for val in parsed_line[1:]], dtype=dtype)
            if vector_dim is not None:
                if len(word_vector) != vector_dim:
                    continue
            if word in ex_w2i:
                continue
            if word not in vocab:
                continue
            ex_w2i[word] = line_i
            word_vectors.append(word_vector)
            line_i += 1

    # add random word vectors for the special symbols and initialize them using Xavier uniform
    vector_dim = word_vectors[0].shape[-1]
    bound = math.sqrt(3. / vector_dim)
    special_word_vec = np.random.uniform(-bound, bound, (3, vector_dim)).astype(dtype)
    return ex_w2i, torch.from_numpy(np.concatenate((special_word_vec, np.stack(word_vectors))))


def to_dependency_graph(sentence, pos_tag, tree, rels):
    dep_graph_string = ''
    n = len(sentence)
    for i in range(n):
        dep_graph_string += sentence[i] + "\t" + pos_tag[i] + "\t" + str(tree[i]) + "\t" + str.upper(rels[i]) + "\n"
    return DependencyGraph(dep_graph_string)


def to_adj_matrix(tree):
    n = tree.shape[-1] + 1
    adj_mat = torch.zeros((n, n))
    adj_mat[tree, torch.arange(1, n)] = 1.
    return adj_mat


def log_results(eval_dict, logger, data_split):
    logger.info('-----------+-----------+-----------+-----------+-----------')
    logger.info(data_split + ' results:')
    logger.info('{:20}|{:6}|'.format('Metric', 'Score'))
    logger.info('-----------+-----------+-----------+-----------+-----------')
    for metric_name, value in eval_dict.items():
        if 'loss' in metric_name:
            logger.info('{:20}|{:.4f}|'.format(metric_name, value))
        else:
            logger.info('{:20}|{:6.2f}|'.format(metric_name, 100 * value))


def write_results_file(eval_dict, output_file):
    with open(output_file, 'w') as f:
        f.write('{:20}|{:6}|\n'.format('Metric', 'Score'))
        f.write('-----------+-----------+-----------+-----------+-----------')
        for metric_name, value in eval_dict.items():
            f.write('\n')
            if 'loss' in metric_name:
                f.write('{:20}|{:.4f}|'.format(metric_name, value))
            else:
                f.write('{:20}|{:6.2f}|'.format(metric_name, 100 * value))


def write_predicted_labels(args, dep_graph_pred, data_split, ds_name):
    if data_split == 'validation':
        data_path = args.valid_path
    else:
        data_path = args.test_path

    lines = []
    sentence_idx = 0
    with open(data_path, 'r') as f:
        for line in f:
            if line == '\n':
                lines.append(line)
                sentence_idx += 1
                continue
            if line.startswith('#'):
                lines.append(line)
                continue
            parsed_line = line.split('\t')
            if '-' in parsed_line[0] or '.' in parsed_line[0]:
                lines.append(line)
            else:
                word_idx = int(parsed_line[0])
                parsed_line[6] = str(dep_graph_pred[sentence_idx].nodes[word_idx]['head'])
                parsed_line[7] = dep_graph_pred[sentence_idx].nodes[word_idx]['rel'].lower()
                parsed_line = '\t'.join(parsed_line)
                lines.append(parsed_line)

    with open(os.path.join(args.experiment_dir, ds_name + '_' + data_split + '_pred.conllu'), 'w') as f:
        for item in lines:
            f.write("%s" % item)