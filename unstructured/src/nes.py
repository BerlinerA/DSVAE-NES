import copy
from collections import defaultdict

import ray
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import zscore

from utils import kl_discrete

FORWARD_NUM_GPUS = 0.15
HIGHEST_RANDINT_VAL = 10000000


@ray.remote(num_gpus=FORWARD_NUM_GPUS)
def nes_forward(args, model, data, mirror_flag, gauss_seed, gumbel_seed):
    model_copy = copy.copy(model)
    model_params = model_copy.state_dict()
    torch.manual_seed(gauss_seed)
    batch_size = data.shape[0]

    # perturbation
    model_params = {param_name: model_params[param_name] + args.sigma
                                * (-1 if mirror_flag else 1) * torch.randn_like(model_params[param_name])
                    for param_name in model_params}

    model_copy.load_state_dict(model_params)
    model_copy = model_copy.train()

    with torch.no_grad():
        torch.manual_seed(gumbel_seed)
        outputs = model_copy(data)

        x_hat, logits = outputs
        bce = F.binary_cross_entropy(x_hat, data.view(batch_size, -1), reduction='none').sum(-1).mean()
        kl = kl_discrete(logits, reduction='mean')
        neg_elbo = bce + args.kl_weight * kl

    return neg_elbo.item()


def train_using_nes(args, model, train_loader, optimizer, logger):

    assert args.n_perturb % 2 == 0, 'n_perturb should be dividable by 2'

    model_sd = model.state_dict()
    args_id = ray.put(args)
    n_batch = len(train_loader)
    ds_len = len(train_loader.dataset)

    stats_dict = defaultdict(float)
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(args.device), target.to(args.device)

        grad_estimation = {param_name: torch.zeros_like(model_sd[param_name]) for param_name in model_sd}

        gauss_seeds = np.random.randint(low=0, high=HIGHEST_RANDINT_VAL, size=args.n_perturb // 2)
        gumbel_seed = np.random.randint(low=0, high=HIGHEST_RANDINT_VAL)

        # store in the local object store
        model_id = ray.put(model)
        data_id = ray.put(data)

        # evaluate
        eval_outputs = np.array(ray.get([nes_forward.remote(args_id, model_id, data_id, mirror_flag, gauss_seed, gumbel_seed)
                                         for gauss_seed in gauss_seeds for mirror_flag in range(2)]))
        # normalize
        norm_outputs = zscore(eval_outputs)

        # gradient estimation
        for idx in range(args.n_perturb // 2):
            gauss_seed = gauss_seeds[idx]
            torch.manual_seed(gauss_seed)

            for param_name in grad_estimation:
                noise = torch.randn_like(model_sd[param_name])
                grad_estimation[param_name] += norm_outputs[2 * idx] * noise + norm_outputs[2 * idx + 1] * -noise
                if idx == (args.n_perturb // 2) - 1:
                    grad_estimation[param_name] *= (1 / (args.n_perturb * args.sigma))

        # update
        model_sd = optimizer.update(model_sd, grad_estimation)
        model.load_state_dict(model_sd)

        neg_elbo = eval_outputs.mean()
        stats_dict['elbo'] -= neg_elbo
        if batch_idx % args.log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)]\tELBO: {:.4f}'.format(
                batch_idx * args.batch_size, ds_len, 100. * batch_idx / n_batch, -neg_elbo))

    stats_dict['elbo'] /= n_batch
    return stats_dict
