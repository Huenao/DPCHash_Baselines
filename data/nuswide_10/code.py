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

print(len(img))
print(len(img_label))
print(len(img_label_onehot))


train_num = 5000
test_num = 5000

perm_index = np.random.permutation(len(img))


all_data = []
for index in perm_index:
    all_data.append(img[index] + " " + img_label_onehot[index] + "\n")

query = all_data[:test_num]
database = all_data[test_num:]
train = database[:train_num]

with open("database.txt", "w") as f:
    for line in database:
        f.write(line)
with open("train.txt", "w") as f:
    for line in train:
        f.write(line)
with open("test.txt", "w") as f:
    for line in query:
        f.write(line)
