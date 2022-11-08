# Generalized Matrix Factorization
import torch
import torch.nn as nn

class GMF__(nn.Module):
    def __init__(
        self, user_num, item_num, factor_num, num_layers, dropout, model,
    ):
        super(GMF, self).__init__()
        self.dropout = dropout
        self.model = model

        # 임베딩 저장공간 확보; num_embeddings, embedding_dim
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        predict_size = factor_num

        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        concat = output_GMF

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
    
class GMF(nn.Module):
    def __init__(self, args):
        super(GMF, self).__init__()
        
        self.user_layer = nn.Sequential(
            nn.BatchNorm1d(args['user_dim']),
            nn.Linear(args['user_dim'], args['layer_dim'])
        )
        self.item_layer = nn.Sequential(
            nn.BatchNorm1d(args['item_dim']),
            nn.Linear(args['item_dim'], args['latent_dim'])
        )
        self.predict_layer = nn.Linear(args['latent_dim'], 1)
        
    def forward(self, user, item):
        u = self.user_layer(user)
        i = self.item_layer(item)
        output = self.predict_layer(u*i)
        
        return output