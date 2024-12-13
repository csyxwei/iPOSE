
import models.models_partsyn as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

acc = utils.multi_acc_withvis(opt, model, models, dataloader_val, image_saver)
print(acc)