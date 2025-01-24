# Deep Generative Clustering with VAEs and Expectation-Maximization

## Introduction

This repo contains the code for the paper [Deep Generative Clustering with VAEs and Expectation-Maximization](https://arxiv.org/abs/2501.07358).

## Acknowledgements

We would like to thank the author of https://github.com/GuHongyang/VaDE-pytorch to make the code public for his implementation of VaDE from which the stucture of VAE in this repo inspired from. 

## Inputs and Arguments

| Argument         | Type     | Default      | Description                                                                                         |
|------------------|----------|--------------|-----------------------------------------------------------------------------------------------------|
| `--file_path`    | `str`    | `"run_1"`    | Directory name for saving weights, and results                                                      |
| `--dataset`      | `str`    | `"MNIST"`    | Dataset to use, options include `"toy", "MNIST"` and `"FashionMNIST"`                               |
| `--epochs`       | `int`    | `300`        | Number of EM iteration                                                                              |
| `--n_class`      | `int`    | `10`         | Number of classes or clusters                                                                       |
| `--load_state`   | `bool`   | `False`      | If `True`, loads pre-trained model state (requires corresponding saved model in `file_path`)        |
| `--batch_size`   | `int`    | `256`        | Batch size                                                                                          |
| `--latent_dim`   | `int`    | `20`         | Latent space dimension                                                                              |
| `--learning_rate`| `float`  | `1e-3`       | Learning rate                                                                                       |
| `--lambd_rec`    | `float`  | `1`          | Reconstruction loss weight                                                                          |
| `--loss`         | `str`    | `MSE`        | 'BCE' or 'MSE' (Bernoulli or Gaussian decoder)                                                      |
| `--VAE_epoch`    | `int`    | `20`         | VAE epoch for each $k$ (only in our model)                                                          |
| `--BatchNorm`    | `bool`   | `True`       | Use BatchNorm ("True" or "False")                                                                   |
| `--Dropout`      | `bool`   | `False`      | Use Dropout ("True" or "False")                                                                     |

## Running an Example

Below is an example command for running the script:

```bash
python3 main.py --file_path mnist_run_1 --lambd_rec 1 --dropout_prob 0.2 --latent_dim 20 --batch_size 256 --VAE_epoch 20 --learning_rate 0.001  --BatchNorm False --Dropout True --dataset MNIST --loss MSE --n_class 10
