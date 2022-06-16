"""
DATE : 2021-10-12
@author: yan

#?芸楛?刻ㄐ?Ｙ??mageSets鞈?憭?

for Windows

//2022-06-15
?啣?銝鈭酉閫?鈭????璅??撣?

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
txtsavepath = 'ImageSets/o12v1_Main1'
#%% for create path

sets = ['train', 'val']
FILE = Path(__file__).absolute()
abs_path=FILE.parents[0].as_posix()
for image_set in sets:
    if not os.path.exists('labels'):
        os.makedirs('labels')
    image_ids = open('ImageSets/o12v1_Main1/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
       #convert_annotation(image_id)
    list_file.close()
# %%
