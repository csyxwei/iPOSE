
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from utils.fid_scores import fid_pytorch

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

is_best = fid_computer.update(model, 0, models)

model.eval()
#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    _, label = models.preprocess_input(opt, data_i)
    generated = model(None, label, "generate", None)
    image_saver(label, generated, data_i["name"])
