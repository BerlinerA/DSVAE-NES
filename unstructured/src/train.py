import argparse
import math
import logging
import sys
from collections import defaultdict

import ray
import numpy as np
import torch.optim as optim
from torchvision import transforms

from unstructured.src.data import dataset_creation, get_data_loaders
from unstructured.src.model import VAE
from unstructured.src.utils import *
from unstructured.src.nes import train_using_nes
from unstructured.src.optim import Adam


def main():
    parser = argparse.ArgumentParser(description='unstructured discrete VAE')
    parser.add_argument('--dataset', default='MNIST', metavar='DS',
                        help='choose one of the four supported datasets: MNIST, FashionMNIST, KMNIST, Omniglot')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dim_size', type=int, default=28,
                        help='input dimension size (default: 28)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of unsupervised epochs to train (default: 250)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dimension (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='latent dimension (default: 10)')
    parser.add_argument('--log_prob_bound', type=int, default=100,
                        help='log probability bound (default: 100)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='smoothing parameter (default: 0.1)')
    parser.add_argument('--n_perturb', type=int, default=300,
                        help='number of samples for update direction estimation (default: 300)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--tau', type=float, default=1.,
                        help='softmax temperature (default: 1.)')
    parser.add_argument('--ar', type=float, default=0.,
                        help='annealing rate (default: 0.)')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='KL term weight (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--binarize', action='store_true', default=False,
                        help='whether to binarize the dataset')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='whether to use a validation dataset')
    parser.add_argument('--nes', action='store_true', default=False,
                        help='whether to optimize using NES')
    parser.add_argument('--sst', action='store_true', default=False,
                        help='whether to optimize using SST')
    parser.add_argument('--valid_prop', type=float, default=1 / 6,
                        help='validation set proportion (default: 1/6)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status (default: 100)')

    args = parser.parse_args()

    # check arguments
    assert 'Please choose one out of the two possible optimization methods', \
        (args.nes + args.sst) == 1

    # define experiment path
    args.dataset_dir = f'./data/{args.dataset}'
    if args.nes:
        args.experiment_dir = f'./results/NES_{args.dataset}_M={args.log_prob_bound}_SIGMA={args.sigma}' \
                              f'_N={args.n_perturb}_HDIM={args.hidden_dim}_LDIM={args.latent_dim}_LR={args.lr}_SEED={args.seed}'
    else:
        args.experiment_dir = f'./results/SST_{args.dataset}_M={args.log_prob_bound}_TAU={args.tau}' \
                              f'_AR={args.ar}_HDIM={args.hidden_dim}_LDIM={args.latent_dim}_LR={args.lr}_SEED={args.seed}'

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # initialize logging object
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(args.experiment_dir, 'train.log'))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # save the experiment parameters
    save_obj(vars(args), args.experiment_dir, 'config')

    root_logger.info(f'Experiment Parameters - \n{vars(args)}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')

    # define input transformations
    transform = [transforms.Resize(args.dim_size), transforms.ToTensor()]
    if args.binarize:
        transform.append(Binarize())
    transform = transforms.Compose(transform)

    # set up training, validation and test data
    train_set, test_set = dataset_creation(args, transform)
    train_loader, valid_loader, test_loader = get_data_loaders(args, train_set, test_set, use_cuda)

    input_size = args.dim_size ** 2
    model = VAE(input_size, args.hidden_dim, args.latent_dim,
                log_prob_bound=args.log_prob_bound,
                nes=args.nes).to(args.device)

    # stats dictionaries
    stats = {'train': defaultdict(list), 'test': defaultdict(list), 'valid': defaultdict(list)}

    if args.nes:
        ray.init()
        optimizer = Adam(model.state_dict(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # collect the pretraining statistics
    test_epoch_stats = evaluate(args, model, test_loader, root_logger)
    for stat_type, value in test_epoch_stats.items():
        stats['test'][stat_type].append(value)

    if args.nes:
        train = train_using_nes
    else:
        train = train_using_sst

    args.train_step = 0
    best_elbo = -np.infty
    for epoch in range(1, args.epochs + 1):

        root_logger.info('-' * 30)
        root_logger.info(f'Train epoch: {epoch}')

        train_epoch_stats = train(args, model, train_loader, optimizer, root_logger)

        for stat_type, value in train_epoch_stats.items():
            stats['train'][stat_type].append(value)

        if args.validate:
            valid_epoch_stats = evaluate(args, model, valid_loader, root_logger, data_split='validation')
            for stat_type, value in valid_epoch_stats.items():
                stats['valid'][stat_type].append(value)

            # save best model
            if valid_epoch_stats['elbo'] > best_elbo:
                torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'vae.pt'))
                best_elbo = valid_epoch_stats['elbo']
                root_logger.info("---ELBO was improved. The current model was saved.---")

        test_epoch_stats = evaluate(args, model, test_loader, root_logger)
        for stat_type, value in test_epoch_stats.items():
            stats['test'][stat_type].append(value)

    if not args.validate:
        torch.save(model.state_dict(), os.path.join(args.experiment_dir, 'vae.pt'))

    save_obj(stats, args.experiment_dir, 'stats')

    # test best model
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, 'vae.pt'), map_location=args.device))
    root_logger.info('-' * 30)
    evaluate(args, model, test_loader, root_logger)


def train_using_sst(args, model, train_loader, optimizer, logger):
    model = model.train()
    stats_dict = defaultdict(float)
    n_batch = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()

        # temperature annealing
        if args.train_step % 1000 == 0:
            args.tau = max(0.5, math.exp(-args.train_step * args.ar))

        outputs = model(data, args.tau)
        x_hat, logits = outputs
        bce = F.binary_cross_entropy(x_hat, data.view(batch_size, -1), reduction='none').sum(-1).mean()
        kl = kl_discrete(logits, reduction='mean')
        neg_elbo = bce + args.kl_weight * kl
        stats_dict['bce'] += bce.item()
        stats_dict['kl'] += kl.item()
        stats_dict['elbo'] -= neg_elbo.item()

        neg_elbo.backward()
        optimizer.step()
        args.train_step += 1

        if batch_idx % args.log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)]\tELBO: {:.4f}, Tau: {:.2f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / n_batch, -neg_elbo.item(), args.tau))

    return {stat_type: stat / n_batch for stat_type, stat in stats_dict.items()}


def evaluate(args, model, test_loader, logger, data_split='test'):

    model = model.eval()
    stats_dict = defaultdict(float)
    n_batch = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            data, target = data.to(args.device), target.to(args.device)
            outputs = model(data)

            x_hat, logits = outputs
            bce = F.binary_cross_entropy(x_hat, data.view(batch_size, -1), reduction='none').sum(-1).mean()
            kl = kl_discrete(logits, reduction='mean')
            stats_dict['bce'] += bce.item()
            stats_dict['kl'] += kl.item()
            stats_dict['elbo'] -= (bce + args.kl_weight * kl).item()

    stats_dict = {stat_type: stat / n_batch for stat_type, stat in stats_dict.items()}

    if args.nes:
        approx_dist = gauss_approx_dist(args, model, test_loader)
        stats_dict['approx_dist'] = approx_dist.item()

    logger.info(
        data_split + ' set: ELBO: {:.4f}, BCE loss: {:.4f}, KL: {:.4f}'.format(
            stats_dict['elbo'], stats_dict['bce'], stats_dict['kl']))

    return stats_dict


if __name__ == '__main__':
    main()
