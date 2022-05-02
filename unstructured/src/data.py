from torchvision import datasets
from torch.utils.data import DataLoader, random_split


def dataset_creation(args, transform):

    load_train_kwargs = {'root': args.dataset_dir,
                         'train': True,
                         'download': True,
                         'transform': transform}
    load_test_kwargs = {'root': args.dataset_dir,
                         'train': False,
                         'download': True,
                         'transform': transform}

    if args.dataset == 'MNIST':
        train_set = datasets.MNIST(**load_train_kwargs)
        test_set = datasets.MNIST(**load_test_kwargs)
    elif args.dataset == 'FashionMNIST':
        train_set = datasets.FashionMNIST(**load_train_kwargs)
        test_set = datasets.FashionMNIST(**load_test_kwargs)
    elif args.dataset == 'KMNIST':
        train_set = datasets.KMNIST(**load_train_kwargs)
        test_set = datasets.KMNIST(**load_test_kwargs)
    elif args.dataset == 'Omniglot':
        load_train_kwargs['background'] = load_train_kwargs.pop('train')
        load_test_kwargs['background'] = load_test_kwargs.pop('train')
        train_set = datasets.Omniglot(**load_train_kwargs)
        test_set = datasets.Omniglot(**load_test_kwargs)
    else:
        raise ValueError(f"The code doesn't support {args.dataset} dataset")

    return train_set, test_set


def get_data_loaders(args, train_set, test_set, use_cuda):

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_loader = DataLoader(test_set, **test_kwargs)
    if args.validate:
        # split to train and validation sets
        train_set, valid_set = random_split(train_set, [int(len(train_set) * (1 - args.valid_prop)),
                                                        int(len(train_set) * args.valid_prop)])
        train_loader = DataLoader(train_set, **train_kwargs)
        valid_loader = DataLoader(valid_set, **test_kwargs)
        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(train_set, **train_kwargs)
        return train_loader, test_loader, test_loader


