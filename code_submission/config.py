import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import os
pgd_path=os.path.join(os.path.dirname(__file__),'pgd')
workers='max'