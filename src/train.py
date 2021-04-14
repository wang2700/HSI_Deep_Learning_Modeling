import sys
import yaml
import pprint
import os
import datetime
from train_regression import train_regre
from train_spec_AE import train_spec_AE


args = sys.argv
cfg = yaml.load(open('/home/jerry/Documents/Research/HSI_Deep_Learning_Modeling/config/' +
                     args[1]), Loader=yaml.FullLoader)
# create directory for model and config
time = datetime.datetime.now()
time = time.strftime("%Y-%m-%d-%H-%M-%S")
# model_path = os.path.join(cfg['MODEL']['MODEL_PATH'], time)
# os.mkdir(model_path)
# print("Create folder at: " + model_path)
model_path = '/media/jerrynas/Research/LeafSpec/Models/HSI Deep Learning Model/Dec2020/2021-02-19-15-28-52'
# save yaml specified to this model
with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
    yaml.dump(cfg, f)
pprint.pprint(cfg, indent=2)
# train_spec_AE(cfg, model_path)
train_regre(cfg, model_path)
