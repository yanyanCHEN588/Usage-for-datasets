"""
DATE : 2021-10-12
@author: yan

#自己在裡面生成ImageSets資料夾

for Windows

//2022-06-15
新增一些註解為了保持兩邊同樣分布

"""
#%%
import os
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


#%%setting path
os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser()

# parser.add_argument('--xml_path', default='labels', type=str, help='input xml label path')

# parser.add_argument('--txt_path', default='ImageSets/Main', type=str, help='output txt label path')
# opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = 'labels'
txtsavepath = 'ImageSets/Main'
#%% split
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
#%% for create path

sets = ['train', 'val']
FILE = Path(__file__).absolute()
abs_path=FILE.parents[0].as_posix()
for image_set in sets:
    if not os.path.exists('labels'):
        os.makedirs('labels')
    image_ids = open('ImageSets/Main1/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
       #convert_annotation(image_id)
    list_file.close()
# %%
