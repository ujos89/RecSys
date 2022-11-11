import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
import models

from pathlib import Path
from dataloader.dataset import *
from tools.evaluation import evaluation
from models.GMF import GMF

def create_model(model_type, args, device):
    if model_type == 'GMF':
        model = GMF(args['model']['GMF']).to(device)
    else:
        raise NotImplementedError
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
    
    return model, loss_fn, optimizer


def main():
    args = yaml.safe_load(Path('args/main.yaml').read_text())
    print(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # build dataloader
    train_pn, valid_pn, test_pn = build_dataset('posneg', args['dataset'])
    train_pnloader = build_dataloader(train_pn, args['dataloader'])
    valid_pnloader = build_dataloader(valid_pn, args['dataloader'])
    test_pnloader = build_dataloader(test_pn, args['dataloader'])
    
    # create model
    model_type = 'GMF'
    model, loss_fn, optimizer = create_model(model_type, args, device)
    
    for epoch in range(args['train']['epochs']):
        start_time = time.time()
        
        train_loss = 0.0
        model.train()
        for (user, item), label in train_pnloader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pred = model(user, item)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            break

        val_loss = 0.0
        model.eval()
        for (user, item), label in valid_pnloader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            
            pred = model(user, item)
            loss = loss_fn(pred, label)
            val_loss += loss.item()
            
            break
            
        print("{} EPOCHS -> loss:({:.3f},{:.3f}), time:{:.2f}s".format(epoch+1, train_loss, val_loss, time.time()-start_time))
        
        if epoch%args['train']['save_epochs']==0:
            torch.save(model.state_dict(), os.path.join(args['train']['save_path'], model_type.lower()+'_'+str(epoch+1)+'.pth'))

        # ## metric TODO
        # mean average precision
        # mae
        # recall
        # precision
            
    
if __name__=='__main__':
    main()