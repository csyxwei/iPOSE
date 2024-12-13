import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "ade20kins":
        return "Ade20kInsDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "cityscapesins":
        return "CityscapesInsDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode == "cocoins":
        return "CocoStuffInsDataset"
    if mode == "partsplit":
        return "PartSplitDataset"
    if mode == "image":
        return "ImageDataset"
    if mode == "single":
        return "SingleDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    # if False:
    if 'ade20k' in opt.dataset_mode or 'coco' in opt.dataset_mode or 'part' in opt.dataset_mode:
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=opt.batch_size,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       num_workers=16)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=opt.batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=16)
    else:
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size = opt.batch_size,
                                                       shuffle = True,
                                                       drop_last=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     batch_size = opt.batch_size,
                                                     shuffle = False,
                                                     drop_last=False)



    return dataloader_train, dataloader_val