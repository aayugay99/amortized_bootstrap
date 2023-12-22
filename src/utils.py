from tqdm import tqdm

import torch
import numpy as np

from .bootstrap import get_bootstrap_sample


def eval_bagging(sampler, model_class, X, y, k_list, metric, agg_func, n_rerun=100):
    metric_list = []

    for k in k_list:
        metric_cur = []
        for _ in range(n_rerun):
            theta_sampled = sampler.sample(k)

            models = [model_class(theta_sampled[i]) for i in range(k)]
            y_pred = agg_func([model(X) for model in models])

            metric_cur.append(metric(y, y_pred))
        
        metric_list.append(metric_cur)

    return np.array(metric_list)


def reduce(inputs):
    res = 0
    for inp in inputs:
        res += inp
    return res / len(inputs)


def train_model(
        X,
        y,
        amortized_bootstrap, 
        model_class, 
        optimizer, 
        criterion, 
        T, 
        k, 
        latent_size,
        gradient_steps
    ):

    history = []
    for _ in tqdm(range(T)):
        bootstrap_samples = [get_bootstrap_sample(X, y) for _ in range(k)]
        
        X_k, y_k = zip(*bootstrap_samples)

        z_k = torch.randn(k, latent_size)

        epoch_loss = []
        for _ in range(gradient_steps):
            theta_k = amortized_bootstrap(z_k) 

            models = [model_class(theta_cur) for theta_cur in theta_k]
            y_pred_k = [model(X_cur) for model, X_cur in zip(models, X_k)]

            loss_k = [criterion(y_pred, y) for y_pred, y in zip(y_pred_k, y_k)]
            loss = reduce(loss_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        history.append(np.mean(epoch_loss))

    return history
