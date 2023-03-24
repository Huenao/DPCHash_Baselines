import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, type):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


def save_config(config, save_path):
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, cls=MyEncoder, indent=4, separators=(', ', ': '))


def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))

    P, R = [], []

    for k in tqdm(draw_range):

        p = np.zeros(n_query)
        r = np.zeros(n_query)

        for it in range(n_query):

            gnd = Gnd[it]
            gnd_all = np.sum(gnd)

            if gnd_all == 0:
                continue

            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))

    return P, R


def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.cuda())).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def compute_result_feas(dataloader, net):
    bs, clses, feas = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        b, fea = net(img.cuda())
        bs.append(b.data.cpu())
        feas.append(fea.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses), torch.cat(feas)


def compute_result_CIB(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls in tqdm(dataloader):
        clses.append(cls)
        bs.append((net.module.encode_discrete(img.cuda())).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def compute_result_vit(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        hash_code = net(img.cuda())[1]
        bs.append(hash_code.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def evalModel(test_loader, dataset_loader, net, Best_mAP, bit, config, epoch, f):
    print("calculating test binary code......")
    if config['info'] == "CIBHash":
        tst_binary, tst_label = compute_result_CIB(test_loader, net)
    else:
        tst_binary, tst_label = compute_result(test_loader, net)

    print("calculating dataset binary code.......")
    if  config['info'] == "CIBHash":
        trn_binary, trn_label = compute_result_CIB(dataset_loader, net)
    else:
        trn_binary, trn_label = compute_result(dataset_loader, net)

    print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                     config["topK"])
    if mAP > Best_mAP:

        Best_mAP = mAP

        P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())

        print(f"Precision Recall Curve data:\n{config['info']}:[{P},{R}],")

        f.write('PR | Epoch %d | ' % (epoch))
        for PR in range(len(P)):
            f.write('%.5f %.5f ' % (P[PR], R[PR]))
        f.write('\n')

        if "logs_path" in config:
            if not os.path.exists(config["logs_path"]):
                os.makedirs(config["logs_path"])
            
            print("save in ", config["logs_path"])

            # delete old file
            for file in os.listdir(config['logs_path']):
                if f"{config['dataset']}-{bit}-" in file:
                    os.remove(os.path.join(config['logs_path'], file))

            np.save(os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-trn_binary.npy"),
                    trn_binary.numpy())
            torch.save(net.state_dict(),
                    os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-model.pt"))

    print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.5f, Best MAP: %.5f" % (
        config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
    f.write('Test | Epoch %d | MAP: %.5f | Best MAP: %.5f\n'
        % (epoch, mAP, Best_mAP))
    
    return Best_mAP
