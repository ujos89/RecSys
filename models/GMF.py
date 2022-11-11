# Generalized Matrix Factorization
import torch
import torch.nn as nn
    
class GMF(nn.Module):
    def __init__(self, args):
        super(GMF, self).__init__()
        
        self.user_layer = nn.Sequential(
            nn.BatchNorm1d(args['user_dim']),
            nn.Linear(args['user_dim'], args['latent_dim'])
        )
        self.item_layer = nn.Sequential(
            nn.BatchNorm1d(args['item_dim']),
            nn.Linear(args['item_dim'], args['latent_dim'])
        )
        self.predict_layer = nn.Sequential(
            nn.Linear(args['latent_dim'], 1),
            nn.Sigmoid()
        )
        
    def forward(self, user, item):
        u = self.user_layer(user)
        i = self.item_layer(item)
        output = self.predict_layer(u*i)
        
        return output