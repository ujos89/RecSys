import os
import torch
import yaml

from pathlib import Path
from dataloader.dataset import *
from tools.evaluation import evaluation
from models.models import create_model


def main():
    args = yaml.safe_load(Path('args/main.yaml').read_text())
    
    # bulid dataloader
    train_pn, test_pn = bulid_dataset('posneg', args['dataset'])
    train_pnloader = bulid_dataloader(train_pn, args['dataloader'])
    test_pnloader = bulid_dataloader(test_pn, args['dataloader'])
    
    # create model
    model = create_model('GMF', args['model']['GMF'])
    
    
if __name__=='__main__':
    main()