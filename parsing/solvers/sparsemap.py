"""
Parts of this code belongs to the sparseMAP package and are freely distributed under a MIT License.
GitHub: https://github.com/vene/sparsemap
"""

import torch
from torch.autograd import Function

from sparsemap._factors import PFactorTreeFast
from sparsemap._sparsemap import sparsemap
from ad3 import PFactorGraph


def tree_layer(unaries, n_nodes, max_iter=10, verbose=0):
    return SparseMAPTreeLayer.apply(unaries, n_nodes, max_iter, verbose)


class SparseMAPTreeLayer(Function):

    @staticmethod
    def _build_factor(n_nodes):
        g = PFactorGraph()
        arcs = [(h, m)
                     for m in range(1, n_nodes + 1)
                     for h in range(n_nodes + 1)
                     if h != m]
        arc_vars = [g.create_binary_variable() for _ in arcs]
        tree = PFactorTreeFast()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1)
        return tree

    @staticmethod
    def _S_from_Ainv(Ainv):

        # Ainv = torch.FloatTensor(Ainv).view(1 + n_active, 1 + n_active)
        S = Ainv[1:, 1:]
        k = Ainv[0, 0]
        b = Ainv[0, 1:].unsqueeze(0)

        S -= (1 / k) * (b * b.t())
        return S

    @staticmethod
    def _d_vbar(status, M, dy):

        Ainv = torch.from_numpy(status['inverse_A'])
        S = SparseMAPTreeLayer._S_from_Ainv(Ainv)

        if M.is_cuda:
            S = S.cuda()
        # B = S11t / 1S1t
        # dvbar = (I - B) S M dy

        # we first compute S M dy
        first_term = S @ (M @ dy)
        # then, BSMt dy = B * first_term. Optimized:
        # 1S1t = S.sum()
        # S11tx = (S1) (1t * x)
        second_term = (first_term.sum() * S.sum(0)) / S.sum()
        d_vbar = first_term - second_term
        return d_vbar

    @staticmethod
    def forward(ctx, unaries, n_nodes, max_iter, verbose):

        cuda_device = None
        if unaries.is_cuda:
            cuda_device = unaries.get_device()
            unaries = unaries.cpu()

        factor = SparseMAPTreeLayer._build_factor(n_nodes)
        u, _, status = sparsemap(factor, unaries, [],
                                 max_iter=max_iter,
                                 verbose=verbose)
        ctx.status = status

        out = torch.from_numpy(u)
        if cuda_device is not None:
            out = out.cuda(cuda_device)
        return out

    @staticmethod
    def backward(ctx, dy):
        cuda_device = None

        if dy.is_cuda:
            cuda_device = dy.get_device()
            dy = dy.cpu()

        M = torch.from_numpy(ctx.status['M'])

        d_vbar = SparseMAPTreeLayer._d_vbar(ctx.status, M, dy)
        d_unary = M.t() @ d_vbar

        if cuda_device is not None:
            d_unary = d_unary.cuda(cuda_device)

        return d_unary, None, None, None