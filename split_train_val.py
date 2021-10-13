"""
DATE : 2021-10-12
@author: yan
這份是建立分割train與val清單
train.txt，val.txt，test.txt和trainval.txt四个文件，存放训练集、验证集、测试集图片的名字（无后缀.jpg）

只要load label內的xml檔案
我將原本 default='Annotations' 更改為 default='labels'

###記得分train/val前 要把看標註的classes.txt移除來
###
"""
#%%
import os
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', default='labels', type=str, help='input xml label path')

parser.add_argument('--txt_path', default='ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()
#資料全部是100%、Train90% 剩下10%val
trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
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
#%%

sets = ['train', 'val']
FILE = Path(__file__).absolute()
abs_path=FILE.parents[0].as_posix()
for image_set in sets:
    if not os.path.exists('labels'):
        os.makedirs('labels')
    image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
       #convert_annotation(image_id)
    list_file.close()