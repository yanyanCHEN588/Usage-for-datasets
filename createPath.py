# -*- coding: utf-8 -*-
"""
2022-03-29

Windoiw / Linux  both can used!
@author: yan

base on fair_split_tainval.py
產生出的main來去製造路徑
creat absoult path to training

prepare---
ImageSets/o12v1_Main1'

setting-----
setName="o12v1_Main1"

output---- absoult path 
train.txt
val.txt


"""

#%% setting the path of works

from pathlib import Path
import random
import numpy as np
import os

os.chdir(os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__)) #window need comment this line
# os.chdir(os.path.dirname(__file__)) # linux need this line

filterpath=Path(os.path.dirname(__file__))

#%% setting the i/o states

setName="o12v1_Main3"

txtsavepath = 'ImageSets/'+setName

# %% 生成檔案路徑給model 訓練

sets = ['train', 'val']
FILE = Path(__file__).absolute()
abs_path=FILE.parents[0].as_posix()
for image_set in sets:
    if not os.path.exists('labels'):
        os.makedirs('labels')
    image_ids = open(txtsavepath+'/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.jpg\n' % (image_id))
       #convert_annotation(image_id)
    list_file.close()
