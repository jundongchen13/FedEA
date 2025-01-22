import copy

import torch
import torch.nn.functional as F

from model.engine import Engine


class Adapter(torch.nn.Module):
    def __init__(self, config):
        super(Adapter, self).__init__()
        # config
        self.input_dim = config['latent_dim'] * 2
        self.hidden_dim = config['hidden_dim']
        self.hidden_num = config['hidden_num']

        # mlp for adapter
        self.layers = [torch.nn.Linear(self.input_dim, self.hidden_dim)]
        for _ in range(self.hidden_num):
            self.layers.append(torch.nn.ReLU(inplace=True))
            self.layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(torch.nn.ReLU(inplace=True))
        self.layers.append(torch.nn.Linear(self.hidden_dim, 1))
        self.layers.append(torch.nn.Sigmoid())
        self.mlp = torch.nn.Sequential(*self.layers)
        # self.param_init()
    
    def param_init(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.constant_(layer.bias, 0)
                if layer.out_features != 1:
                    torch.nn.init.xavier_uniform_(layer.weight)

        last_layer = self.mlp[-2]
        torch.nn.init.zeros_(last_layer.weight)
        torch.nn.init.constant_(last_layer.bias, -5)

    def forward(self, local_params, server_params):
        delta_params = server_params - local_params
        inputs = torch.cat((local_params, delta_params), dim=-1)
        weights = self.mlp(inputs)
        # update embeddings
        self.delta_weight = weights[:,0].unsqueeze(1)
        weighted_params = local_params + self.delta_weight * delta_params
        return weighted_params


class FCF(torch.nn.Module):
    def __init__(self, config):
        super(FCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.adapter = Adapter(self.config)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.user_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()

        # self._init_weight_()

    def setItemEmbeddings(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)

    def _init_weight_(self):
        torch.nn.init.normal_(self.item_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data /= (self.config['latent_dim'] ** 0.5)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        self.user_embedding.weight.data /= (self.config['latent_dim'] ** 0.5)

    def weight_adapt(self, server_model_param):
        local_item_embeddings = copy.deepcopy(self.item_embeddings.weight.data)
        server_item_embeddings = copy.deepcopy(server_model_param['item_embeddings.weight'].data)
        return self.adapter(local_item_embeddings, server_item_embeddings)

    def reset_embeddings(self, server_model_param):
        self.item_embeddings.weight.data = self.weight_adapt(server_model_param)
        self.adapt_weight = 1 - self.adapter.delta_weight

    def adapter_forward(self, item_indices, server_model_param):
        self.adapted_embeddings = self.weight_adapt(server_model_param)
        item_embeddings = self.adapted_embeddings[item_indices,:]
        user_embedding = self.user_embedding.weight.data
        rating = torch.mul(user_embedding, item_embeddings)
        rating = torch.sum(rating, dim=1)
        rating = self.logistic(rating)

        return rating
    
    def forward(self, item_indices):
        item_embeddings = self.item_embeddings(item_indices)
        user_embedding = self.user_embedding.weight
        rating = torch.mul(user_embedding, item_embeddings)
        rating = torch.sum(rating, dim=1)
        rating = self.logistic(rating)

        return rating


class FedNCF(torch.nn.Module):
    def __init__(self, config):
        super(FedNCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.adapter = Adapter(self.config)
        self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.user_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        
        self.mlp_weights = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.config['mlp_layers'][:-1], self.config['mlp_layers'][1:])):
            self.mlp_weights.append(torch.nn.Linear(in_size, out_size))

        self.logistic = torch.nn.Sigmoid()

    def setItemEmbeddings(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings)

    def setMLPweights(self, mlp_weights):
        self.mlp_weights = copy.deepcopy(mlp_weights)

    def weight_adapt(self, server_model_param):
        local_item_embeddings = copy.deepcopy(self.item_embeddings.weight.data)
        server_item_embeddings = copy.deepcopy(server_model_param['item_embeddings.weight'].data)
        return self.adapter(local_item_embeddings, server_item_embeddings)

    def reset_embeddings(self, server_model_param):
        self.item_embeddings.weight.data = self.weight_adapt(server_model_param)
        self.adapt_weight = 1 - self.adapter.delta_weight

    def adapter_forward(self, item_indices, server_model_param):
        self.adapted_embeddings = self.weight_adapt(server_model_param)
        item_embeddings = self.adapted_embeddings[item_indices,:]
        user_embedding = self.user_embedding.weight.data
        repeat_num = item_embeddings.size(0)
        user_embedding = user_embedding.repeat(repeat_num, 1)

        vector = torch.cat([user_embedding, item_embeddings], dim=-1)
        for idx, _ in enumerate(range(len(self.mlp_weights))):
            vector = self.mlp_weights[idx](vector)
            if idx != len(self.mlp_weights) - 1:
                vector = torch.nn.ReLU()(vector)
        rating = self.logistic(vector)

        return rating

    def forward(self, item_indices):
        item_embeddings = self.item_embeddings(item_indices)
        user_embedding = self.user_embedding.weight
        repeat_num = item_embeddings.size(0)
        user_embedding = user_embedding.repeat(repeat_num, 1)
        vector = torch.cat([user_embedding, item_embeddings], dim=-1)
        for idx, _ in enumerate(range(len(self.mlp_weights))):
            vector = self.mlp_weights[idx](vector)
            if idx != len(self.mlp_weights) - 1:
                vector = torch.nn.ReLU()(vector)
        rating = self.logistic(vector)

        return rating


class ModelEngine(Engine):
    """Engine for training & evaluating FCF model"""

    def __init__(self, config):
        model_name = config['backbone']
        self.model = eval(model_name)(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(ModelEngine, self).__init__(config)
        print(self.model)
