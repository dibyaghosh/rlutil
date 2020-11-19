def set_seed(seed):
    import random
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass
    