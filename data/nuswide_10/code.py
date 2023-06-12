# you can use code.py to  to generate database.txt, train.txt and test.txt of nuswide_10
import os
import numpy as np

root = '/data2/huwentao/Data/NUS-WIDE'

img = []
with open(os.path.join(root, 'img_tc10.txt')) as f:
    for line in f.readlines():
        img.append(line.strip())

img_label = []
with open(os.path.join(root, 'targets_tc10.txt')) as f:
    for line in f.readlines():
        img_label.append(line.strip())
    
img_label_onehot = []
with open(os.path.join(root, 'targets_onehot_tc10.txt')) as f:
    for line in f.readlines():
        img_label_onehot.append(line.strip())

train_num = 500
test_num = 500
train_count = [train_num] * 10
test_count = [test_num] * 10
train_data = []
test_data = []
database_data = []

perm_index = np.random.permutation(len(img))

for index in perm_index:
    line = img[index] + " " + img_label_onehot[index] + "\n"
    add_position = "database"
    for classnum in img_label[index].split():
        classnum = int(classnum)
        if train_count[classnum]:
            add_position = "train"
            train_count[classnum] -= 1
            break
        if test_count[classnum]:
            add_position = "test"
            test_count[classnum] -= 1
            break

    if add_position == "train":
        train_data.append(line)
        database_data.append(line)
    elif add_position == "test":
        test_data.append(line)
    else:
        database_data.append(line)

with open("database.txt", "w") as f:
    for line in database_data:
        f.write(line)
with open("train.txt", "w") as f:
    for line in train_data:
        f.write(line)
with open("test.txt", "w") as f:
    for line in test_data:
        f.write(line)