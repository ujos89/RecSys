from GMF import GMF

def create_model(model_type, args):
    if model_type == 'GMF':
        model = GMF(args)
    else:
        raise NotImplementedError
    
    return model
