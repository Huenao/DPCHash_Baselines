import os
import torch
import time
import numpy as np
import random

from config import get_config
from utils.tools import *

torch.multiprocessing.set_sharing_strategy('file_system')


def train_val(config, bit, t):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, t, config).to(device)

    optimizer = config["optimizer"]["type"]([
        {"params": net.encoder.parameters(), "lr": config["optimizer"]["lr"]},
        {"params": net.vit.parameters(), "lr": config["optimizer"]["backbone_lr"]},
    ])


    # load parameters
    # model_path = 'save/IPHash_KD_RM_CIFAR_v3/cifar10-1-16-0.9422065465113109-model.pt'
    # net.load_state_dict(torch.load(model_path))

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, temperature:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, t, config["dataset"]), end="")

        net.train()

        train_loss = 0
        
        for image, _, _ in train_loader:
            image = image.to(device)

            optimizer.zero_grad()

            losses = net(image)

            loss = losses['loss']

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        

        print("\b\b\b\b\b\b\b loss:%.5f" % train_loss)

        if (epoch + 1) % config["test_map"] == 0:
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # PR curve data
            # d_range = [i for i in range(0, 10000, 500)] + [i for i in range(10000, 50001, 2500)]
            # d_range[0] = 1
            # P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), draw_range=d_range)
            # print(f'Precision Recall Curve data:\n"Ours":[{P},{R}],')

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-%d-" % bit + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-%d-" % bit + str(mAP) + "-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(6)
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        for t in config["temperature"]:
            train_val(config, bit, t)
