# FedEA: Personalized Federated Recommendation with Elastic Adaptation

## Introduction

Federated recommendation (FR) has attracted great attention as a critical paradigm to protect user privacy in recommender systems. However, there exists natural heterogeneity among clients due to the varying user preferences, making it challenging to learn a global model tailored for each client. To mitigate this issue, existing personalized federated recommendation (pFR) methods typically design additional mechanisms to enhance the personalization of the global model, then simply replace the local model with the enhanced global model to facilitate local training. However, we empirically and theoretically find that such replacement does not always work, as the collaborative information from the global model is not always beneficial for each client. To this end, we propose a novel pFR method called personalized Federated recommendation with Elastic Adaptation (FedEA), which selectively absorbs the collaborative information from the global model into the local one. It aims to achieve an elastic balance between the collaborative information of the global model and the off-the-shelf personalized information of the local model. Extensive experiments on several real-world datasets show the effectiveness of our method compared to state-of-the-art methods.

## Requirements

The code is built on `Python=3.8` and `Pytorch=2.0`.

The other necessary Python libraries are as follows:
    
* coloredlogs>=15.0.1
* cvxpy==1.4.2
* numpy>=1.24.3
* pandas>=2.0.3
* scikit_learn>=1.0.2

To install these, please run the following commands:

  `pip install -r requirements.txt`
  
## Code Structure

The structure of our project is presented in a tree form as follows:

```
FedEA  # The root of project.
│   README.md
│   requirements.txt
│   train.py # The entry function file includes the main hyperparameter configurations.
|
└───datasets  # The used datasets in this work.
│   │   filmtrust   
|   │       ratings.dat
|   |   ...
|   |
└───model  # The main components in FR tasks.
│   │  engine.py # It includes the server aggregation and local training processes.
│   │  loss.py # Task-specific loss for local clients.
│   │  model.py # Defined backbone model (e.g., PMF and NCF) network architecture.
│   │  tools.py # Composite aggregation optimization process.
|   |
└───utils  # Other commonly used tools.
|   │   data.py # Codes related to data loading and preprocessing.
|   │   metrics.py # The evaluation metrics used in this work.
|   │   utils.py # Other utility functions.
```

## Parameters Settings

The meanings of the hyparameters are as follows:

`backbone`: the architecture of the backbone model used, the default value is `FCF`.

`dataset`: the name of used datasets, the default value is `filmtrust`.

`data_file `: the path of raw ratings data file, the default value is `ratings.dat`.

`train_frac`: the proportion of the training set used, the default value is `1.0`.

`clients_sample_ratio`: the proportion of user embeddings involved in the updates, the default value is `1.0`.

`top_k`: the specific value of K in evaluation metrics, the default value is `10`.

`global_round`: the number of global aggregation rounds, the default value is `100`.

`local_epoch`: the number of local training rounds, the default value is `10`.

`threshold`: the threshold for convergence of local training, the default value is `1e-4`.

`batch_size`: the number of local batch size, the default value is `256`.

`lr_structure`: the learning rate for training structured parameters, the default value is `1e-2`.

`lr_embedding`: the learning rate for training embedding parameters, the default value is `1e-2`.

`weight_decay`: the parameter regularization coefficient, the default value is `1e-1`.

`latent_dim`: the dimensions of user and item embeddings, the default value is `16`.

`mlp_layers`: the specific number of layers and units used in MLPs, the default value is `[32, 16, 8, 1]`.

`num_negative`: the number of negative samples used for local training, the default value is `4.0`.

`agg_clients_ratio`: the proportion used for participating in item embeddings aggregation, the default value is `0.1`.

`lr_adapter`: the learning rate for EA module, the default value is `1e-2`.

`alpha`: the weight of model similarity, the default value is `0.8`.

`beta`: the weight of aggregation sparsity, the default value is `0.1`.

`hidden_dim`: the number of MLP hidden nodes in EA module, which is consistent with latent_dim, the default value is `16`.

`hidden_num`: the number of MLP layers in EA module, which is consistent with `latent_dim`, the default values are `2` for FCF and `3` for FedNCF.


## Quick Start

Please change the used dataset and hyperparameters in `train.py`.

To run FCF with composite aggregation mode:

  `python train.py --backbone='FCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`

To run FedNCF with composite aggregation mode:

  `python train.py --backbone='FedNCF' --dataset='filmtrust' --data_file='ratings.dat' --lr_structure=1e-2 --lr_embedding=1e-2`
