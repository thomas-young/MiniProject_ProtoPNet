import os
import shutil
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

def main():
    # -------------------------------
    # Device selection for Apple M3 Mac
    # -------------------------------
    parser = argparse.ArgumentParser()
    # Although you may supply a GPU id for CUDA, on an M3 Mac CUDA won’t be available.
    parser.add_argument('-gpuid', nargs=1, type=str, default=['0'])
    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if CUDA is available.
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
        print("CUDA_VISIBLE_DEVICES:", os.environ['CUDA_VISIBLE_DEVICES'])

    # Choose the appropriate device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # -------------------------------
    # Book keeping, file copies, and logging setup
    # -------------------------------
    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                         prototype_activation_function, add_on_layers_type, experiment_run

    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    model_dir = os.path.join('./saved_models', base_architecture, experiment_run)
    makedir(model_dir)
    # Save a copy of key files for reproducibility.
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # -------------------------------
    # Data loading
    # -------------------------------
    from settings import train_dir, test_dir, train_push_dir, \
                         train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean, std=std)

    # Training set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=8, pin_memory=False
    )

    # Push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    )
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    # Test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # -------------------------------
    # Model construction and device transfer
    # -------------------------------
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)
    ppnet = ppnet.to(device)
    # Only use DataParallel if CUDA is available (MPS and CPU do not support DataParallel)
    if device.type == 'cuda':
        ppnet_multi = torch.nn.DataParallel(ppnet)
    else:
        ppnet_multi = ppnet

    class_specific = True

    # -------------------------------
    # Optimizers and Schedulers
    # -------------------------------
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

    from settings import warm_optimizer_lrs
    warm_optimizer_specs = [
        {'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    from settings import coefs, num_train_epochs, num_warm_epochs, push_start, push_epochs

    # -------------------------------
    # Training Loop
    # -------------------------------
    log('start training')
    import copy
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_push_loader,  # DataLoader (expects unnormalized images in [0,1])
                prototype_network_parallel=ppnet_multi,  # the network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,  # normalizes if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,  # directory to save prototype images
                epoch_number=epoch,  # prototype saving versioning
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.70, log=log)

            if base_architecture_type != 'linear':  # Or check your prototype_activation_function as needed
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                                model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.70, log=log)
    logclose()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # This is useful on Windows and sometimes on macOS.
    main()
