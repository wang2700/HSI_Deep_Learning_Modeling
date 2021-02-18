import sys
import yaml
import pprint
from train_regression import train_regre


args = sys.argv
cfg = yaml.load(open('/home/jerry/Documents/Research/HSI_Deep_Learning_Modeling/config/' + args[1]), Loader=yaml.FullLoader)
pprint.pprint(cfg, indent=2)
train_regre(cfg, 'test')