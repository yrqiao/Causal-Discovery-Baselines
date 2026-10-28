"""DAG-GNN (Yu et al., ICML 2019). Source: https://github.com/ronikobrosly/DAG_from_GNN

`fit` is the original DAG_from_GNN/__main__.py turned into a function: the global CONFIG becomes the
`cfg` argument and the hard-coded `load_data` is replaced by the data passed in.
"""
import math
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from .modules import MLPDecoder, MLPEncoder, SEMDecoder, SEMEncoder
from .utils import (A_connect_loss, A_positive_loss, encode_onehot, get_tril_offdiag_indices,
                    get_triu_offdiag_indices, kl_gaussian_sem, matrix_poly, nll_gaussian)


def run(dataset, cfg, workdir):
    return fit(dataset.X, cfg)


def fit(X, cfg):
    """X: [n, d] array. Returns the binary [d, d] adjacency (thresholded at cfg['graph_threshold'])."""
    CONFIG = SimpleNamespace(**cfg)
    CONFIG.cuda = not CONFIG.no_cuda and torch.cuda.is_available()
    CONFIG.factor = not CONFIG.no_factor

    torch.manual_seed(CONFIG.seed)
    if CONFIG.cuda:
        torch.cuda.manual_seed(CONFIG.seed)

    # ================================================
    # get data (was load_data in utils.py)
    # ================================================
    X = np.expand_dims(np.asarray(X, dtype=np.float64), 2)
    CONFIG.data_variable_size = X.shape[1]
    feat_train = torch.FloatTensor(X)
    train_loader = DataLoader(TensorDataset(feat_train, feat_train), batch_size=CONFIG.batch_size)

    # ===================================
    # load modules
    # ===================================
    # Generate off-diagonal interaction graph
    off_diag = np.ones([CONFIG.data_variable_size, CONFIG.data_variable_size]) - np.eye(
        CONFIG.data_variable_size
    )

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    num_nodes = CONFIG.data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))

    if CONFIG.encoder == "mlp":
        encoder = MLPEncoder(
            CONFIG.data_variable_size * CONFIG.x_dims,
            CONFIG.x_dims,
            CONFIG.encoder_hidden,
            int(CONFIG.z_dims),
            adj_A,
            batch_size=CONFIG.batch_size,
            do_prob=CONFIG.encoder_dropout,
            factor=CONFIG.factor,
        ).double()
    elif CONFIG.encoder == "sem":
        encoder = SEMEncoder(
            CONFIG.data_variable_size * CONFIG.x_dims,
            CONFIG.encoder_hidden,
            int(CONFIG.z_dims),
            adj_A,
            batch_size=CONFIG.batch_size,
            do_prob=CONFIG.encoder_dropout,
            factor=CONFIG.factor,
        ).double()

    if CONFIG.decoder == "mlp":
        decoder = MLPDecoder(
            CONFIG.data_variable_size * CONFIG.x_dims,
            CONFIG.z_dims,
            CONFIG.x_dims,
            encoder,
            data_variable_size=CONFIG.data_variable_size,
            batch_size=CONFIG.batch_size,
            n_hid=CONFIG.decoder_hidden,
            do_prob=CONFIG.decoder_dropout,
        ).double()
    elif CONFIG.decoder == "sem":
        decoder = SEMDecoder(
            CONFIG.data_variable_size * CONFIG.x_dims,
            CONFIG.z_dims,
            2,
            encoder,
            data_variable_size=CONFIG.data_variable_size,
            batch_size=CONFIG.batch_size,
            n_hid=CONFIG.decoder_hidden,
            do_prob=CONFIG.decoder_dropout,
        ).double()

    # ===================================
    # set up training parameters
    # ===================================
    if CONFIG.optimizer == "Adam":
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
        )
    elif CONFIG.optimizer == "LBFGS":
        optimizer = optim.LBFGS(
            list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
        )
    elif CONFIG.optimizer == "SGD":
        optimizer = optim.SGD(
            list(encoder.parameters()) + list(decoder.parameters()), lr=CONFIG.lr
        )

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=CONFIG.lr_decay, gamma=CONFIG.gamma
    )

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(CONFIG.data_variable_size)
    tril_indices = get_tril_offdiag_indices(CONFIG.data_variable_size)

    if CONFIG.prior:
        prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
        print("Using prior")
        print(prior)
        log_prior = torch.DoubleTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)

        if CONFIG.cuda:
            log_prior = log_prior.cuda()

    if CONFIG.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0.0, 0.0)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(optimizer, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group["lr"] = lr

        return optimizer, lr

    # ===================================
    # training:
    # ===================================

    def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []

        encoder.train()
        decoder.train()
        scheduler.step()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):

            if CONFIG.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data).double(), Variable(relations).double()

            # reshape data
            relations = relations.unsqueeze(2)

            optimizer.zero_grad()

            (
                enc_x,
                logits,
                origin_A,
                adj_A_tilt_encoder,
                z_gap,
                z_positive,
                myA,
                Wa,
            ) = encoder(
                data, rel_rec, rel_send
            )  # logits is of size: [num_sims, z_dims]
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(
                data,
                edges,
                CONFIG.data_variable_size * CONFIG.x_dims,
                rel_rec,
                rel_send,
                origin_A,
                adj_A_tilt_encoder,
                Wa,
            )

            if torch.sum(output != output):
                print("nan error\n")

            target = data
            preds = output
            variance = 0.0

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if CONFIG.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if CONFIG.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += 0.1 * (
                    lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap
                )

            # compute h(A)
            h_A = _h_A(origin_A, CONFIG.data_variable_size)
            loss += (
                lambda_A * h_A
                + 0.5 * c_A * h_A * h_A
                + 100.0 * torch.trace(origin_A * origin_A)
                + sparse_loss
            )  # +  0.01 * torch.sum(variance * variance)

            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, CONFIG.tau_A * lr)

            if torch.sum(origin_A != origin_A):
                print("nan error\n")

            # compute metrics
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < CONFIG.graph_threshold] = 0

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        print(h_A.item())

        print(
            "Epoch: {:04d}".format(epoch),
            "nll_train: {:.10f}".format(np.mean(nll_train)),
            "kl_train: {:.10f}".format(np.mean(kl_train)),
            "ELBO_loss: {:.10f}".format(np.mean(kl_train) + np.mean(nll_train)),
            "mse_train: {:.10f}".format(np.mean(mse_train)),
            "time: {:.4f}s".format(time.time() - t),
        )

        return (
            np.mean(np.mean(kl_train) + np.mean(nll_train)),
            np.mean(nll_train),
            np.mean(mse_train),
            graph,
            origin_A,
        )

    # ===================================
    # main
    # ===================================

    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    # optimizer step on hyparameters
    c_A = CONFIG.c_A
    lambda_A = CONFIG.lambda_A
    h_A_new = torch.tensor(1.0)
    h_tol = CONFIG.h_tol
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    for step_k in range(k_max_iter):
        while c_A < 1e20:
            for epoch in range(CONFIG.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(
                    epoch, best_ELBO_loss, lambda_A, c_A, optimizer
                )
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, CONFIG.data_variable_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A *= 10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    # Create binary adjacency matrix using config threshold (the final A, as in the original script)
    final_A = origin_A.data.clone().cpu().numpy()
    print(final_A)
    return (np.abs(final_A) >= CONFIG.graph_threshold).astype(int)
