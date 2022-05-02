import ray
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.gumbel import Gumbel
from scipy.stats import zscore

from parsing.solvers.decoder_eisner import parse_proj
from parsing.solvers.chu_liu_edmonds import decode_mst
from parsing.src.utils import normalize, to_adj_matrix
from parsing.src.constants import *


@ray.remote(num_gpus=FORWARD_NUM_GPUS)
def nes_forward(args, parser, decoder, sentences, batch_w_i, mirror_flag, gauss_seed, gumbel_seed):

    parser_sd = parser.state_dict()
    decoder_sd = decoder.state_dict()

    torch.manual_seed(gauss_seed)
    # perturb parser weights
    for param_name in parser_sd:
        if param_name in FROZEN_EMB_LAYERS_NAME:
            continue
        if param_name == ACTIVE_EMB_LAYER_NAME:
            noise = (-1 if mirror_flag else 1) * torch.randn_like(parser_sd[param_name][batch_w_i])
            parser_sd[param_name][batch_w_i] += args.sigma * noise
        else:
            noise = (-1 if mirror_flag else 1) * torch.randn_like(parser_sd[param_name])
            parser_sd[param_name] += args.sigma * noise

    if not args.freeze_decoder:
        # perturb decoder weights
        for param_name in decoder_sd:
            if param_name in FROZEN_EMB_LAYERS_NAME:
                continue
            noise = (-1 if mirror_flag else 1) * torch.randn_like(decoder_sd[param_name])
            decoder_sd[param_name] += args.sigma * noise

    parser.load_state_dict(parser_sd)
    decoder.load_state_dict(decoder_sd)

    parser.train()
    decoder.train()

    with torch.no_grad():

        torch.manual_seed(gumbel_seed)
        rec_loss = 0.
        for batch_idx, sentence in enumerate(sentences):
            # encode
            arc_scores, rel_scores = parser(sentence)

            # perturb and parse
            gumbel_noise = Gumbel(loc=GUMBEL_LOC, scale=GUMBEL_SCALE).sample(arc_scores.shape).to(parser.device).squeeze(-1)

            if args.non_projective:
                mst, _ = decode_mst((arc_scores + gumbel_noise).detach().cpu().numpy(), len(sentence), has_labels=False)
                pred_tree = to_adj_matrix(mst[1:]).to(parser.device)
            else:
                pred_tree = to_adj_matrix(
                    np.array(parse_proj((arc_scores + gumbel_noise).detach().cpu().numpy())[1:])).to(parser.device)

            # decode
            # following the paper "DIFFERENTIABLE PERTURB-AND-PARSE" (https://arxiv.org/pdf/1807.09875.pdf]),
            # we only consider the reconstruction term of the ELBO.
            rec_loss += decoder(sentence, pred_tree)

    return rec_loss.item() / len(sentences)


def train_using_nes(args, parser, decoder, train_set, optimizer, logger, w2i):
    parser_sd = parser.state_dict()
    decoder_sd = decoder.state_dict()
    assert args.n_perturb % 2 == 0, 'n_perturb should be dividable by 2'

    args_id = ray.put(args)

    n_batch = int(np.ceil(len(train_set) / args.batch_size))
    indices = [i for i in range(len(train_set))]
    for batch_idx in range(n_batch):

        # set up batch dataloader
        batch_subset = Subset(train_set, indices[batch_idx * args.batch_size:
                                                 (batch_idx + 1) * args.batch_size if batch_idx != n_batch + 1
                                                 else len(train_set)])
        batch_loader = DataLoader(batch_subset, shuffle=True)

        sentences_ls = []
        batch_w_i = set()
        for sentence, _, _, _ in batch_loader:
            sentences_ls.append(sentence)
            for w in sentence:
                batch_w_i.add(w2i.get(normalize(w[0]), 0))
        batch_w_i = list(batch_w_i)

        random_seeds = np.random.randint(low=0, high=HIGHEST_RANDINT_VAL, size=args.n_perturb // 2)
        gumbel_seed = np.random.randint(low=0, high=HIGHEST_RANDINT_VAL)

        # put in the local object store
        parser_id = ray.put(parser)
        decoder_id = ray.put(decoder)
        batch_w_i_id = ray.put(batch_w_i)
        sentences_ls_id = ray.put(sentences_ls)

        # evaluate
        eval_outputs = np.array(ray.get([nes_forward.remote(args_id, parser_id, decoder_id, sentences_ls_id,
                                                                   batch_w_i_id, mirror_flag, gauss_seed, gumbel_seed)
                                         for gauss_seed in random_seeds for mirror_flag in range(2)]))

        # normalize
        norm_outputs = zscore(eval_outputs)

        # parser gradient dict initialization
        enc_grad_est = {}
        for param_name in parser_sd:
            if param_name in FROZEN_EMB_LAYERS_NAME:
                continue
            if param_name == ACTIVE_EMB_LAYER_NAME:
                enc_grad_est[param_name] = torch.zeros_like(parser_sd[param_name][batch_w_i])
            else:
                enc_grad_est[param_name] = torch.zeros_like(parser_sd[param_name])

        if not args.freeze_decoder:
            # decoder gradient dict initialization
            dec_grad_est = {}
            for param_name in decoder_sd:
                if param_name in FROZEN_EMB_LAYERS_NAME:
                    continue
                dec_grad_est[param_name] = torch.zeros_like(decoder_sd[param_name])

        # parser & decoder gradient estimation
        for idx in range(args.n_perturb // 2):
            gauss_seed = random_seeds[idx]
            torch.manual_seed(gauss_seed)

            for param_name in enc_grad_est:
                if param_name == ACTIVE_EMB_LAYER_NAME:
                    noise = torch.randn_like(parser_sd[param_name][batch_w_i])
                else:
                    noise = torch.randn_like(parser_sd[param_name])
                enc_grad_est[param_name] += norm_outputs[2 * idx] * noise + norm_outputs[2 * idx + 1] * -noise
                if idx == (args.n_perturb // 2) - 1:
                    enc_grad_est[param_name] *= (1 / (args.n_perturb * args.sigma))

            if not args.freeze_decoder:
                for param_name in dec_grad_est:
                    noise = torch.randn_like(decoder_sd[param_name])
                    dec_grad_est[param_name] += norm_outputs[2 * idx] * noise + norm_outputs[2 * idx + 1] * -noise
                    if idx == (args.n_perturb // 2) - 1:
                        dec_grad_est[param_name] *= (1 / (args.n_perturb * args.sigma))

        # update
        if args.freeze_decoder:
            parser_sd = optimizer.update(parser_sd, enc_grad_est, batch_w_i=batch_w_i)
        else:
            parser_sd, decoder_sd = optimizer.update(parser_sd, enc_grad_est, decoder_sd,
                                                     dec_grad_est, batch_w_i=batch_w_i)
        parser.load_state_dict(parser_sd)
        decoder.load_state_dict(decoder_sd)

        if ((batch_idx + 1) * args.batch_size) % args.log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)]\tReconstruction Loss: {:.6f}'.format(
                (batch_idx + 1) * args.batch_size, len(train_set),
                100. * batch_idx / n_batch, eval_outputs.mean()))
