import torch
import models.losses as losses
import models.models_partsyn as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import itertools

#--- read options ---#
opt = config.read_arguments(train=True)
losses_computer = losses.losses_computer(opt)
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
results_saver = utils.results_saver(opt)
#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)

best_acc = 0

#--- create optimizers ---#
optimizerG = torch.optim.Adam(itertools.chain(model.module.netG.parameters(),),lr=opt.lr_g, betas=(opt.beta1, opt.beta2))

#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)

        #--- generator update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        if cur_iter % 50 == 0:
            print('Iter', cur_iter, end=' ')

            for loss in losses_G_list:
                print(loss.item(), '  ', end=' ')

            print()

        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch_part_syn(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_ckpt == 0:
            utils.save_networks(opt, cur_iter, model)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)

        if cur_iter % opt.freq_fid == 0:
            acc = utils.multi_acc(opt, model, models, dataloader_val, results_saver)
            if acc.item() > best_acc:
                best_acc = acc.item()
                utils.save_networks(opt, cur_iter, model, best=True)
            print('Iter', acc.item(), best_acc)