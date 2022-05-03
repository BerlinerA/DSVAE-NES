import numpy as np
from torch.utils.data.dataset import Dataset
from nltk.parse import DependencyGraph

from constants import *


class DependencyDataSet(Dataset):
    def __init__(self, words_count, unique_rels, w2i, data_path,
                 unk_sym=UNK_SYMBOL, root_sym=ROOT_SYMBOL, root_rel_sym=RROOT_SYMBOL,):
        super().__init__()

        self.data_path = data_path

        self.w2i = w2i
        self.w2i[unk_sym] = 0

        # create index-relation mapping
        self.r2i = {rel: rel_i for rel_i, rel in enumerate(unique_rels)}
        self.i2r = list(self.r2i.keys())
        self.n_unique_rel = len(self.r2i)

        # create word index counts dict
        self.w_i_counts = {self.w2i[w]: count for w, count in words_count.items()}

        # special tokens
        self.unk_idx = self.w2i.get(unk_sym)
        self.root_sym = root_sym
        self.root_idx = self.w2i.get(root_sym)
        self.root_rel_sym = root_rel_sym

        # read the data
        self.sentences, self.pos_tags, self.gold_trees, self.relations, self.dep_graph_gold = self._data_reader()

        # convert to dataset format
        self.sentences_dataset = self._convert_to_dataset()

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        sentence, pos_tags, gold_tree, relations = self.sentences_dataset[index]
        return sentence, pos_tags, gold_tree, relations

    def _data_reader(self):
        with open(self.data_path, 'r', encoding="utf8") as fh:

            sentences = []
            pos_tags = []
            gold_trees = []
            relations = []
            dep_graph_gold = []

            sentence, pos_tag, gold_tree, sen_rel, dep_graph_string = [self.root_sym], [self.root_sym], [-1], [self.r2i[self.root_rel_sym]], ''

            for line in fh:
                if line.startswith('#'):
                    continue
                if line == '\n' and len(sentence) > 1:

                    sentences.append(sentence)
                    pos_tags.append(pos_tag)
                    gold_trees.append(np.array(gold_tree))
                    relations.append(np.array(sen_rel))
                    dep_graph_gold.append(DependencyGraph(dep_graph_string))

                    sentence = [self.root_sym]
                    pos_tag = [self.root_sym]
                    gold_tree = [-1]
                    sen_rel = [self.r2i[self.root_rel_sym]]
                    dep_graph_string = ''

                else:
                    parsed_line = line.strip().split('\t')
                    if '-' in parsed_line[0] or '.' in parsed_line[0]:
                        continue
                    word, pos, head, arc_rel = parsed_line[1], parsed_line[3], int(parsed_line[6]), parsed_line[7]
                    if len(word.split(' ')) > 1:
                        word = '_'.join(word.split(' '))
                    sentence.append(word)
                    pos_tag.append(pos)
                    gold_tree.append(head)
                    sen_rel.append(self.r2i.get(arc_rel, 0))

                    dep_graph_string += word + "\t" + pos + "\t" + str(head) + "\t" + str.upper(arc_rel) + "\n"

        return sentences, pos_tags, gold_trees, relations, dep_graph_gold

    def _convert_to_dataset(self):
        return {i: sample_tuple for i, sample_tuple in
                enumerate(zip(self.sentences, self.pos_tags, self.gold_trees, self.relations))}