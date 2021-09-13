# -*- coding: utf-8 -*-
"""
2021-08-25
@author: yan

從coco官方載的去選取要的類別並挑選CLASS
然後過濾掉非黑色

coco官方是 JSON檔案
以物件偵測為例
且也能保留原始檔案名稱

#參考
    https://blog.csdn.net/weixin_38145317/article/details/103137822

#-----------config----------
範圍內是能改的
    尤其是classes_names
#-----------config----------

目標: 目標可以分出特定類別

測試特色:
	1.模擬在object365中有images有許多patch檔案 -> train2017、train69、train28
		而且不完全照片，也就是緊抓現有的image標註
	2.模擬全部的json
	3.挑出特定類別轉為XML
	4.在挑選出中分出train、val 的txt清單 ImageSets/
	5.製作出要給yolo的label.txt 檔案與影像清單

/cocotest
    /images
        /train2017、(就是coco128 原始train前128張)
        /train28 (原始train 中段28張)
        /train69 (原始train 後段69張)
    /annotations
        instances_train2017.josn (原始JSON檔案)


/coco_5t_class 
    split_train_val.py (完成4.並生成  ImageSets/ 內四個文件
    voc_label.py (完成5. 
    /images
        *.jpg
    /annotations (以前是label但為了後續3.4.5.改為/annotations比較方便
        *.xml
    /ImageSets (在4.中的清單
    /labels
	*.txt
    train.txt (在5.中的清單要給yolo中的data.yaml
    val.txt (在5.中的清單要給yolo中的data.yaml

coco_get_spcificCLASS.py (完成1. 2. 3.

#-------更動-------------



"""
#%% load module
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

#%%json coco
###-----------config----------
#the path you want to save your results for coco to voc
savepath="o365_test/"#"coco_2017_sub1/"  #保存提取类的路径,我放在同一路径下 #++

img_dir=savepath+'images/' 
anno_dir=savepath+'annotations/' #++
# datasets_list=['train2014', 'val2014'] 
# datasets_list=['patch29','patch30','patch31','patch32','patch33','patch34','patch35', 'patch36','patch37','patch40'] #++
datasets_list=['patch0','patch5'] #++
classes_names = ["Chair","Bottle","Cup","Handbag/Satchel","Bowl/Basin","Umbrellaz","Cell Phone","Spoon","Remote","Refrigerator","Microwave","Toothbrush","Tablet"]  #coco有80类，这里写要提取类的名字，以person为例 #++
# classes_names = ["Bus","Car"]
# classes_names = ["Glasses","Person","Street Lights","Umbrella"]
#Store annotations and train2014/val2014/... in this folder
dataDir= 'Objects365/'  #原coco数据集 #++

# osPath=Path.cwd()
# img_dir=osPath / 'images'
# anno_dir=savepath / 'annotations'

###-----------config----------
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''

def mkr(path):
    path=Path(path)
    if Path.exists(path):
        shutil.rmtree(path)
        Path.mkdir(path)
    else:
        Path.mkdir(path)

mkr(img_dir)
mkr(anno_dir)
def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(coco,dataset,filename,objs):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    
    anno_path=anno_dir+filename[:-3]+'xml'
    img_path=dataDir+'images/'+dataset+'/'+filename

    print(img_path)
    dst_imgpath=img_dir+filename
 
    img=cv2.imread(img_path)
    shutil.copy(img_path, dst_imgpath)#複製貼上影像
    #write VOC format by *.xml
    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)
 
 
def annotations_img(coco,dataset,img,classes,cls_id,filename):
    global dataDir

    #由ID得到標註資料
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=0) #obj365
    anns = coco.loadAnns(annIds)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        BBarea = ann['area']
        if class_name in classes_names and BBarea>500 :
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                

    if not objs == []: #非空BBOX才存XML與IMG
        save_annotations_and_imgs(coco, dataset, filename, objs)

 
    return objs
#----load JSON-----LOAD一次就好，因為假設只有一個JSON檔案
#./COCO/annotations/instances_train2014.json
annFile='{}annotations/zhiyuan_objv2_train.json'.format(dataDir) #++ 因為只有一個寫死
#COCO API for initializing annotated data
#製作對應現有影像清單
coco = COCO(annFile)
#show all classes in coco
classes = id2name(coco)
print("classes",classes)
#%% main
for dataset in datasets_list: #在images/中有幾包影像
    # #./COCO/annotations/instances_train2014.json
    # annFile='{}annotations/instances_train2017.json'.format(dataDir) #++ 因為只有一個寫死
    # #COCO API for initializing annotated data
    # #製作對應現有影像清單
    # coco = COCO(annFile)
    img_path=dataDir+'images/'+dataset
    p = Path(img_path)
    # print(p)
    file_name_list=[]
    for i in p.iterdir(): file_name_list.append(i.name)
    # print(file_name_list)

    #[1, 2, 3, 4, 6, 8] classID清單
    classes_ids = coco.getCatIds(catNms=classes_names) 
    print("classesID",classes_ids)
    for cls in classes_names: #一個個 cls去找
        #Get ID number of this class
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        print(cls,len(img_ids))
        # count=0
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            # filename = img['file_name']
            filename = Path(img['file_name']).name #obej 365
            if filename in file_name_list:
                # count=count+1
                # print(filename)
                objs=annotations_img(coco, dataset, img, classes,classes_ids,filename)
                print(objs)
                # save_annotations_and_imgs(coco, dataset, filename, objs)
        # print(count)
            
            
            