import pandas as pd
import argparse

from model import *
from trainer import Trainer
from config import Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TreeLinearNet()
trainer = Trainer(model)
trainer.train()
trainer.test()



