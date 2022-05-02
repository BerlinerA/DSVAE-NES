import pickle
import os
import copy

import torch
import torch.nn.functional as F


def save_obj(obj, dir, name):
    with open(os.path.join(dir, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(obj_path):
    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def kl_discrete(logits, reduction='sum'):
    decision_dim = logits.size(-1)
    q_z = F.softmax(logits, dim=-1)
    log_ratio = torch.log(q_z * decision_dim + 1e-20)
    kl_div = (q_z * log_ratio).sum(-1)
    if reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'mean':
        return kl_div.mean()
    elif reduction == 'none':
        return kl_div
    else:
        raise ValueError(f"{reduction} is not a valid reduction method")


def get_elbo(args, model, data):
    model = model.eval()
    with torch.no_grad():
        batch_size = data.shape[0]
        x_hat, logits = model(data)

        bce = F.binary_cross_entropy(x_hat, data.view(batch_size, -1), reduction='none')
        kl_dis = kl_discrete(logits, reduction='none')
        neg_elbo = torch.sum(bce, dim=-1) + args.kl_weight * torch.sum(kl_dis, dim=-1)

    return -1 * neg_elbo


def gauss_approx_dist(args, model, test_loader, n_nes_samples=1000):

    data, _ = next(iter(test_loader))
    data = data.to(args.device)

    elbo = get_elbo(args, model, data)
    elbo_approx = torch.zeros_like(elbo)
    for _ in range(n_nes_samples):
        model_copy = copy.deepcopy(model)
        sd = model_copy.state_dict()
        # perturbation
        for param_name in sd:
            noise = torch.randn_like(sd[param_name])
            sd[param_name] += args.sigma * noise

        model_copy.load_state_dict(sd)
        elbo_approx += (1 / n_nes_samples) * get_elbo(args, model_copy, data)

    return torch.abs(elbo - elbo_approx).mean()


class Binarize(object):
    def __call__(self, sample):
        return torch.round(sample)
