import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm


class MyEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, type):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


def save_config(config, save_path):
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, cls=MyEncoder, indent=4, separators=(', ', ': '))


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


# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
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
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall


def evalModel(test_loader, dataset_loader, net, Best_mAP, bit, config, epoch, f, num_dataset):
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
    if "pr_curve_path" not in  config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), 
                         trn_label.numpy(), tst_label.numpy(),
                         config["topK"])

    else:
        # need more memory
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        
        if mAP > Best_mAP:

            index_range = num_dataset // 100
            index = [i * 100 - 1 for i in range(1, index_range + 1)]
            max_index = max(index)
            overflow = num_dataset - index_range * 100
            index = index + [max_index + i for i in range(1, overflow + 1)]
            c_prec = cum_prec[index]
            c_recall = cum_recall[index]

            pr_data = {
                "index": index,
                "P": c_prec.tolist(),
                "R": c_recall.tolist()
            }
            os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
            with open(config["pr_curve_path"], 'w') as pr_f:
                pr_f.write(json.dumps(pr_data))
            print("pr curve save to ", config["pr_curve_path"])

    if mAP > Best_mAP:

        Best_mAP = mAP

        if "logs_path" in config:
            if not os.path.exists(config["logs_path"]):
                os.makedirs(config["logs_path"])
            
            print("save in ", config["logs_path"])

            # delete old file
            for file in os.listdir(config['logs_path']):
                if f"{config['dataset']}-{bit}-" in file:
                    os.remove(os.path.join(config['logs_path'], file))

            np.save(os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-tst_label.npy"),
                    tst_label.numpy())
            np.save(os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-tst_binary.npy"),
                    tst_binary.numpy())
            np.save(os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-trn_binary.npy"),
                    trn_binary.numpy())
            np.save(os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-trn_label.npy"),
                    trn_label.numpy())
            torch.save(net.state_dict(),
                    os.path.join(config["logs_path"], config["dataset"] + "-%d-" % bit + str(round(mAP, 5)) + "-model.pt"))

    print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.5f, Best MAP: %.5f" % (
        config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
    f.write('Test | Epoch %d | MAP: %.5f | Best MAP: %.5f\n'
        % (epoch, mAP, Best_mAP))
    
    return Best_mAP
