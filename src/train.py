import sys
import yaml
import pprint
import os
import datetime
import pickle
import torch
from train_regression import train_regre
from train_spec_AE import train_spec_AE
from Dataset.SpecDataset import SpecDataset


args = sys.argv
cfg = yaml.load(open('./config/' +
                     args[1]), Loader=yaml.FullLoader)
# create directory for model and config
time = datetime.datetime.now()
time = time.strftime("%Y-%m-%d-%H-%M-%S")
model_path = os.path.join(cfg['MODEL']['MODEL_PATH'], time)
os.mkdir(model_path)
print("Create folder at: " + model_path)
# model_path = '/media/jerrynas/Research/LeafSpec/Models/HSI Deep Learning Model/Dec2020/2021-02-19-15-28-52'
# save yaml specified to this model
with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
    yaml.dump(cfg, f)
pprint.pprint(cfg, indent=2)
train_dataset = SpecDataset(cfg=cfg, train=True)
test_dataset = SpecDataset(cfg=cfg, train=False)
with open(os.path.join(model_path, 'train_file_name.data'), 'wb') as filehandle:
    pickle.dump(train_dataset.rawNameList, filehandle)
with open(os.path.join(model_path, 'test_file_name.data'), 'wb') as filehandle:
    pickle.dump(test_dataset.rawNameList, filehandle)
if torch.cuda.device_count() > 1:
    torch.cuda.set_device(1)
best_AE_loss = train_spec_AE(cfg, model_path, train_dataset, test_dataset)
best_REGRE_loss = train_regre(cfg, model_path, train_dataset, test_dataset)
# with open(os.path.join(model_path, 'loss_out.txt'), 'wb') as filehandle:
#     filehandle.write("AE Training - Best Loss: " +
#                      str(best_AE_loss[0]) + 'at Epoch: ' + str(best_AE_loss[1]) + '\n')
#     filehandle.write("Regre Training - Best Loss: " +
#                      str(best_REGRE_loss[0]) + 'at Epoch: ' + str(best_REGRE_loss[1]) + '\n')
