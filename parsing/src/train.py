import argparse
import logging
import sys
from copy import deepcopy
from collections import defaultdict

import ray
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.distributions.gumbel import Gumbel
from nltk.parse import DependencyEvaluator

from parsing.src.utils import *
from parsing.src.data import *
from parsing.src.model import DependencyParser, Decoder
from parsing.solvers.eisner_surrogate import eisner_surrogate
from parsing.solvers.chu_liu_edmonds import decode_mst
from parsing.solvers.decoder_eisner import parse_proj
from parsing.solvers.sparsemap import tree_layer
from parsing.src.optim import AdamOptim
from parsing.src.nes import train_using_nes


def main():
    parser = argparse.ArgumentParser('description=cross-domain dependency parsing '
                                     'using a structured variational auto-encoder')
    parser.add_argument('--data_path', type=str, default='data/ud',
                        help='path to the datasets directory')
    parser.add_argument('--source', type=str,
                        help='source dataset name')
    parser.add_argument('--target', type=str,
                        help='target dataset name')
    parser.add_argument('--pretrained_path',
                        help='path to pretrained model directory', type=str, default=None)
    parser.add_argument('--ext_emb',
                        help='path to external word embeddings file')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--source_epochs', type=int, default=30,
                        help='number of supervised epochs for training on the source domain (default: 30)')
    parser.add_argument('--target_epochs', type=int, default=10,
                        help='number of unsupervised epochs for training on the target domain (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--source_lr', type=float, default=1e-3,
                        help='learning rate for the supervised phase (default: 1e-3)')
    parser.add_argument('--target_lr', type=float, default=1e-4,
                        help='learning rate for the unsupervised phase (default: 1e-4)')
    parser.add_argument('--w_emb_dim', type=int, default=100,
                        help='word embedding dimension (default: 100)')
    parser.add_argument('--parser_lstm_hid', type=int, default=125,
                        help='LSTM hidden dimension (default: 125)')
    parser.add_argument('--parser_mlp_hid', type=int, default=100,
                        help='MLP hidden dimension (default: 100)')
    parser.add_argument('--parser_lstm_layers', type=int, default=2,
                        help='number of LSTM layers (default: 2)')
    parser.add_argument('--decoder_lstm_hid', type=int, default=100,
                        help='LSTM hidden dimension (default: 100)')
    parser.add_argument('--decoder_mlp_hid', type=int, default=100,
                        help='MLP hidden dimension (default: 100)')
    parser.add_argument('--n_perturb', type=int, default=400,
                        help='number of samples for update direction estimation (default: 400)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='smoothing parameter (default: 0.1)')
    parser.add_argument('--nes', action='store_true', default=False,
                        help='whether to optimize using NES')
    parser.add_argument('--non_projective', action='store_true', default=False,
                        help='whether to learn latent non-projective dependency trees')
    parser.add_argument('--freeze_decoder', action='store_true', default=False,
                        help='whether to freeze the decoder during the cross-domain phase')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=256,
                        help='how many samples to wait before logging training status (default: 256)')
    parser.add_argument('--do_eval', action='store_true', default=False,
                        help='evaluate a given model')

    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    # define experiment path
    if args.do_eval:
        assert 'in evaluation mode, an experiment directory must be supplied.', args.pretrained_path is not None and os.path.isdir(args.pretrained_path)
        args.experiment_dir = args.pretrained_path
    else:
        if not args.target_epochs:
            args.experiment_dir = f'./results/supervised_s={args.source}_proj={not args.non_projective}_lr={args.source_lr}_seed={args.seed}'
        elif args.nes:
            args.experiment_dir = f'./results/nes_s={args.source}_t={args.target}_proj={not args.non_projective}_sigma={args.sigma}_n={args.n_perturb}_lr={args.target_lr}_seed={args.seed}'
        else:
            args.experiment_dir = f'./results/{"sparsemap" if args.non_projective else "dpp"}_s={args.source}_t={args.target}_lr={args.target_lr}_seed={args.seed}'

        if not os.path.exists(args.experiment_dir):
            os.makedirs(args.experiment_dir)

    # initialize logging object
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(args.experiment_dir, 'train.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # save the experiments parameters
    save_obj(vars(args), args.experiment_dir, 'config')

    logger.info(f'Experiment Parameters - \n{vars(args)}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # add data paths to args
    args.source_train_path = os.path.join(args.data_path, args.source, 'train.conllu')
    args.target_train_path = os.path.join(args.data_path, args.target, 'train.conllu')
    args.valid_path = os.path.join(args.data_path, args.source, 'dev.conllu')
    args.test_path = os.path.join(args.data_path, args.target, 'test.conllu')

    data_paths = [args.source_train_path, args.valid_path, args.test_path, args.target_train_path]

    # create vocabulary
    words_count, unique_rels, w2i = vocab(data_paths)
    save_obj([words_count, unique_rels, w2i], args.experiment_dir, 'vocab')
    logger.info('Vocab statistics: unique lower case words - {} | unique relations - {}'
                     .format(len(words_count), len(unique_rels)))

    # load external word embeddings
    ex_w2i, ex_word_vectors = load_pretrained_word_embed(args.ext_emb, w2i)

    # set up training, validation and test data
    source_train_set = DependencyDataSet(words_count, unique_rels, w2i, args.source_train_path)
    target_train_set = DependencyDataSet(words_count, unique_rels, w2i, args.target_train_path)
    valid_set = DependencyDataSet(words_count, unique_rels, w2i, args.valid_path)
    test_set = DependencyDataSet(words_count, unique_rels, w2i, args.test_path)

    # if there is no validation set, use the source domain train set instead
    if len(valid_set) == 0:
        valid_set = deepcopy(source_train_set)

    source_train_gen = DataLoader(source_train_set, shuffle=True)
    target_train_gen = DataLoader(target_train_set, shuffle=True)
    valid_gen = DataLoader(valid_set, shuffle=False)
    test_gen = DataLoader(test_set, shuffle=False)

    # encoder initialization
    parser = DependencyParser(w_emb_dim=args.w_emb_dim,
                              lstm_hid_dim=args.parser_lstm_hid,
                              mlp_hid_dim=args.parser_mlp_hid,
                              n_lstm_l=args.parser_lstm_layers,
                              w_i_counter=source_train_set.w_i_counts,
                              w2i=source_train_set.w2i,
                              device=args.device,
                              ext_emb_w2i=ex_w2i,
                              ex_w_vec=ex_word_vectors,
                              n_arc_relations=source_train_set.n_unique_rel).to(args.device)

    # decoder initialization
    decoder = Decoder(hid_dim=args.decoder_lstm_hid,
                      out_dim=args.decoder_mlp_hid,
                      ext_emb_w2i=ex_w2i,
                      ex_w_vec=ex_word_vectors,
                      device=args.device).to(args.device)

    if args.pretrained_path is not None:
        # load pretrained model weights
        best_parser_path = os.path.join('./results/', args.pretrained_path, 'parser.pt')
        best_decoder_path = os.path.join('./results/', args.pretrained_path, 'decoder.pt')
        parser.load_state_dict(torch.load(best_parser_path, map_location=args.device))
        decoder.load_state_dict(torch.load(best_decoder_path, map_location=args.device))
        parser.to(args.device)
        decoder.to(args.device)
    else:
        best_parser_path = None
        best_decoder_path = None

    vae_params = list(parser.parameters()) + list(decoder.parameters())
    optimizer = Adam(vae_params, lr=args.source_lr)

    if not args.do_eval:

        train_stats = defaultdict(list)
        best_uas = 0.
        best_rec_loss = np.inf
        args.epochs = args.source_epochs + args.target_epochs
        args.pretrain = True
        for epoch in range(1, args.epochs + 1):

            logger.info('-----------+-----------+-----------+-----------+-----------')
            logger.info(f'Train epoch: {epoch}')

            if epoch <= args.source_epochs:  # supervised learning on the source domain
                train(args, parser, decoder, source_train_gen, optimizer, logger)
            else:  # unsupervised learning on the target domain
                if epoch == args.source_epochs + 1:

                    # load best model weights
                    if best_parser_path is not None and best_decoder_path is not None and epoch - 1:
                        parser.load_state_dict(torch.load(best_parser_path, map_location=args.device))
                        decoder.load_state_dict(torch.load(best_decoder_path, map_location=args.device))

                    args.pretrain = False
                    if args.nes:
                        ray.init()
                        optimizer = AdamOptim(parser.state_dict(),
                                  None if args.freeze_decoder else decoder.state_dict(),
                                  lr=args.target_lr)
                    else:
                        if args.freeze_decoder:
                            optimizer = Adam([{'params': parser.parameters()},
                                              {'params': decoder.parameters(), 'lr': 0., 'betas': (0, 0)}],
                                             lr=args.target_lr)
                        else:
                            optimizer = Adam(vae_params,
                                             lr=args.target_lr)

                if args.nes:
                    train_using_nes(args, parser, decoder, target_train_set, optimizer, logger, w2i)
                else:
                    train(args, parser, decoder, target_train_gen, optimizer, logger)

            val_eval_dict = evaluate(args, parser, decoder, valid_gen, logger)

            train_stats['val_reconstruct_loss'].append(val_eval_dict['reconstruct_loss'])
            train_stats['val_discriminative_loss'].append(val_eval_dict['discriminative_loss'])
            train_stats['val_uas'].append(val_eval_dict['uas'])
            train_stats['val_las'].append(val_eval_dict['las'])

            if val_eval_dict['uas'] > best_uas:
                best_uas = val_eval_dict['uas']
                best_parser_path = os.path.join(args.experiment_dir, 'parser.pt')
                torch.save(parser.state_dict(), best_parser_path)
                logger.info("---UAS was improved. The current parser was saved.---")

            if val_eval_dict['reconstruct_loss'] < best_rec_loss:
                best_rec_loss = val_eval_dict['reconstruct_loss']
                best_decoder_path = os.path.join(args.experiment_dir, 'decoder.pt')
                torch.save(decoder.state_dict(), best_decoder_path)
                logger.info("---Negative ELBO was improved. The current decoder was saved.---")

        save_obj(train_stats, args.experiment_dir, 'train_stats')

    # test the best model
    parser.load_state_dict(torch.load(best_parser_path, map_location=args.device))
    decoder.load_state_dict(torch.load(best_decoder_path, map_location=args.device))
    evaluate(args, parser, decoder, valid_gen, logger)
    evaluate(args, parser, decoder, test_gen, logger, data_split=TEST)


def train(args, parser, decoder, train_gen, optimizer, logger):
    parser.train()
    decoder.train()

    batch_loss = 0.
    eff_batch_size = 0
    ds_size = len(train_gen)
    for sample_idx, (sentence, _, gold_tree, gold_rel) in enumerate(train_gen):

        gold_tree = gold_tree.squeeze()[1:].to(args.device)
        gold_rel = gold_rel.squeeze()[1:].to(args.device)
        sentence_len = len(sentence)
        eff_batch_size += 1

        # encode
        arc_scores, rel_scores = parser(sentence, word_dropout=args.pretrain)

        # perturb and parse
        gumbel_noise = Gumbel(loc=GUMBEL_LOC, scale=GUMBEL_SCALE).sample(arc_scores.shape).to(args.device).squeeze(-1)
        if args.non_projective:
            # sparseMAP
            pred_tree = torch.zeros_like(arc_scores)
            pred_tree[:, 1:] = tree_layer(
                torch.flatten((arc_scores + gumbel_noise).T[1:, :]), sentence_len - 1)\
                .view(sentence_len - 1, -1).T
        else:
            # DPP
            pred_tree = eisner_surrogate(arc_scores + gumbel_noise)

        # if the sample is 'labeled', feed the decoder with the corresponding gold tree
        if args.pretrain:
            tree = to_adj_matrix(gold_tree).to(args.device)
        else:
            tree = pred_tree

        # decode
        # following the paper "DIFFERENTIABLE PERTURB-AND-PARSE" (https://arxiv.org/pdf/1807.09875.pdf]),
        # we only consider the reconstruction term of the ELBO.
        rec_loss = decoder(sentence, tree)  # reconstruction loss

        arcs_loss = F.cross_entropy(arc_scores[:, 1:].t(), gold_tree)
        rels_loss = F.cross_entropy(rel_scores[gold_tree, torch.arange(1, sentence_len), :], gold_rel)
        dis_loss = arcs_loss + rels_loss  # discriminative loss

        batch_loss += (dis_loss if args.pretrain else 0.) + rec_loss

        if args.pretrain or sample_idx % args.batch_size == 0 or sample_idx == len(train_gen) - 1:
            batch_loss /= eff_batch_size
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss = 0.
            eff_batch_size = 0

        if sample_idx % args.log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)]\tReconstruction Loss: {:.6f}\tDiscriminative Loss: {:.6f}'.format(
                sample_idx, ds_size,
                100. * sample_idx / ds_size, rec_loss.item(), dis_loss.item()))


def evaluate(args, parser, decoder, test_gen, logger, data_split=VALID):

    parser.eval()
    decoder.eval()

    i2r_map = test_gen.dataset.i2r
    sentences = test_gen.dataset.sentences
    dep_graph_gold = test_gen.dataset.dep_graph_gold

    dep_graph_pred = []
    rec_loss, dis_loss = 0., 0.
    ds_size = len(test_gen)
    with torch.no_grad():
        for sample_idx, (sentence, pos_tags, gold_tree, gold_rel) in enumerate(test_gen):

            gold_tree = gold_tree.squeeze()[1:].to(args.device)
            gold_rel = gold_rel.squeeze()[1:].to(args.device)
            pos_tags = [p[0] for p in pos_tags[1:]]
            sentence_len = len(sentence)

            # encode
            arc_scores, rel_scores = parser(sentence)

            # parse
            if args.non_projective:
                mst, _ = decode_mst(arc_scores.detach().cpu().numpy(), sentence_len, has_labels=False)
                pred_tree = to_adj_matrix(mst[1:]).to(parser.device)
            else:
                pred_tree = to_adj_matrix(
                    np.array(parse_proj(arc_scores.detach().cpu().numpy())[1:])).to(parser.device)

            # decode
            if args.do_eval or not args.pretrain:
                rec_loss += decoder(sentence, pred_tree).item()
            else:
                rec_loss += decoder(sentence, to_adj_matrix(gold_tree).to(args.device)).item()

            arcs_loss = F.cross_entropy(arc_scores[:, 1:].t(), gold_tree)
            rels_loss = F.cross_entropy(rel_scores[gold_tree, torch.arange(1, sentence_len), :], gold_rel)
            dis_loss += (arcs_loss + rels_loss).item()

            comp_pred_tree = torch.argmax(pred_tree, dim=0).cpu().detach().numpy()[1:]
            pred_rel = torch.argmax(rel_scores[gold_tree, torch.arange(1, sentence_len), :], dim=-1)
            pred_rel = [i2r_map[rel_i] for rel_i in pred_rel]

            dep_graph_pred.append(to_dependency_graph(sentences[sample_idx][1:],
                                                      pos_tags, comp_pred_tree, pred_rel))

    rec_loss /= ds_size
    dis_loss /= ds_size

    data_split = data_split.lower()
    ds_name = args.source if data_split == VALID else args.target

    de = DependencyEvaluator(dep_graph_pred, dep_graph_gold)
    las, uas = de.eval()

    eval_dict = {'las': las,
                 'uas': uas,
                 'reconstruct_loss': rec_loss,
                 'discriminative_loss': dis_loss}

    log_results(eval_dict, logger, data_split)
    output_file = os.path.join(args.experiment_dir, ds_name + '_' + data_split + '_results.txt')
    write_results_file(eval_dict, output_file)
    write_predicted_labels(args, dep_graph_pred, data_split, ds_name)

    return eval_dict


if __name__ == '__main__':
    main()