# -*- coding: utf-8 -*-
"""
2021-10-07
@author: yan
刪除的過濾也寫好了
使用類別名稱讀出想要的類別並且紀錄ImgID
使用ImgID去讀取標註並存起來
    需要能夠擷取影像位置

"""
#%% load module
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

#%%
#標註名稱轉換表
import csv
import numpy as np
with open('class_id_Customized.csv') as csvFile:
    csvReader = csv.reader(csvFile)
    data = list(csvReader)

cid,keys=np.array([]),np.array([])
for index in data[1:]: #最上面['', 'name', 'AnnsName']不要
    for name in index[2].split(","):
        keys=np.append(keys,name)
        cid=np.append(cid,index[0])

#%%json coco
###-----------config----------
#the path you want to save your results for coco to voc
savepath="pick_test/"#"coco_2017_sub1/"  #保存提取类的路径,我放在同一路径下 #++
labelformat = 'txt'
img_dir=savepath+'images/' 
anno_dir=savepath+'labels/' #++
# datasets_list=['train2014', 'val2014'] 
# datasets_list=['patch29','patch30','patch31','patch32','patch33','patch34','patch35', 'patch36','patch37','patch40'] #++
# datasets_list=['patch0','patch5'] #++
classes_names = ["Toothbrush"]  #coco有80类，这里写要提取类的名字，以person为例 #++
# classes_names = ["Chair","Bottle","Cup","Handbag/Satchel","Bowl/Basin","Umbrellaz","Cell Phone","Spoon","Remote","Refrigerator","Microwave","Toothbrush","Dinning Table","Coffee Table","Side Table","Desk"]  #coco有80类，这里写要提取类的名字，以person为例 #++
# classes_names = ["Bus","Car"]
# classes_names = ["Person"]
# classes_names = ["Dinning Table","Coffee Table","Side Table","Tablet"]
#Store annotations and train2014/val2014/... in this folder
dataDir= 'Objects365/'  #原coco数据集 #++
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


def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    with open(anno_path, "w") as f:
        f.write(head)
        for obj in objs:
            f.write(objstr%(obj[0],obj[1],obj[2],(obj[3]+obj[1]),(obj[4]+obj[2])))
        f.write(tail)

def write_txt(anno_path , objs ,width ,height):
    with open(anno_path, "w") as f:
        for obj in objs:
            y_id=cid[np.where(keys == obj[0])[0][0]]
            #x, y, w, h = a['bbox']  >>>obj[1],obj[2],obj[3],obj[4]
            xc, yc = obj[1] + obj[3] / 2, obj[2] + obj[4] / 2  # xy to center
            #參考 file.write(f"{cid} {x / width:.5f} {y / height:.5f} {w / width:.5f} {h / height:.5f}\n")
            f.write(f"{y_id} {xc / width:.5f} {yc / height:.5f} {obj[3] / width:.5f} {obj[4]/ height:.5f}\n")
 
def save_annotations_and_imgs_xml(coco,dataset,filename,objs,width ,height):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+filename[:-3]+'xml' #extension是副檔名
    img_path=dataDir+'images/'+dataset+'/'+filename

    print(img_path)
    dst_imgpath=img_dir+filename
 
    img=cv2.imread(img_path)
    shutil.copy(img_path, dst_imgpath)#複製貼上影像
    #write VOC format by *.xml
    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)

def save_annotations_and_imgs_txt(coco,dataset,filename,objs,width ,height):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+filename[:-3]+'txt' #extension是副檔名
    img_path=dataDir+'images/'+dataset+'/'+filename
    print(img_path)
    dst_imgpath=img_dir+filename
    img=cv2.imread(img_path)
    shutil.copy(img_path, dst_imgpath)#複製貼上影像
    write_txt(anno_path, objs ,width ,height)
#%%load annotations
if labelformat == 'txt':
    save_annotations_and_imgs = save_annotations_and_imgs_txt
if labelformat == 'xml':
    save_annotations_and_imgs = save_annotations_and_imgs_xml

def annotations_img(coco,dataset,img,classes,cls_id,filename):
    global dataDir

    #由ID得到標註資料
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=1) #obj365
    annsiscrowd = coco.loadAnns(annIds)
    if len(annsiscrowd) == 0 :  #此圖沒有任何有corwd的標註
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=0) #obj365
        anns = coco.loadAnns(annIds)
        AreaOK = 0 #準備紀錄面積OK的標註
        ann_cls = 0 #針對要擷取的標註計算
        width, height = img["width"], img["height"]
        # coco.showAnns(anns)
        objs = []
        for ann in anns:
            class_name=classes[ann['category_id']]
            AreaRatio=ann['area'] / (width*height)
            if class_name in classes_names:
                ann_cls += 1 #符合要得類別計數
                if AreaRatio > 0.001 and AreaRatio < 0.9  : # AR>0.5% and AR<90%
                    AreaOK += 1 #面積也符合
                    x, y, w, h = ann['bbox']  # bounding box in xywh (xy top-left corner)
                    obj = [class_name, x, y, w, h]
                    objs.append(obj)
                    

        if not objs == [] and (AreaOK/ann_cls>0.3): #非空BBOX才存XML與IMG 且不能太多小的照片(2成)
            save_annotations_and_imgs(coco, dataset, filename, objs ,width ,height)
        return objs

#%%
#---file process----
mkr(img_dir)
mkr(anno_dir)

# %%
#刪除的不要的清單資料夾
del_labelDir=Path("del_label")/'Toothbrush_EX/'

nosave=[]
if del_labelDir.exists(): #存在才執行以下，防呆用
    for file in Path(del_labelDir).iterdir():
        with open(file) as f:
            nosave.extend(f.read().split('\n')[:-1])
#%% load annotations
#----load JSON-----LOAD一次就好，因為假設只有一個JSON檔案
#./COCO/annotations/instances_train2014.json
annFile='{}annotations/zhiyuan_objv2_train.json'.format(dataDir) #++ 因為只有一個寫死
#COCO API for initializing annotated data
#製作對應現有影像清單
coco = COCO(annFile)
#show all classes in coco
classes = id2name(coco)
print("classes",classes)
#%% create class id  list
#classID清單[1, 2, 3, 4, 6, 8] 
classes_ids = coco.getCatIds(catNms=classes_names)
print("classesID",classes_ids)
#%% main
#-----main---製作任何有牙刷類別的清單ImgID
cls="Toothbrush"
cls_id=coco.getCatIds(catNms=[cls])
img_ids=coco.getImgIds(catIds=cls_id) #後面有[:50] eg:取前50張

for imgId in tqdm(img_ids): #依照全部符合cls的ImgID一張張跑
    """
    img['file_name'] >>> 'images/v2/patch45/objects365_v2_02060288.jpg'
    img['file_name'].split('/') >>> ['images', 'v2', 'patch45', 'objects365_v2_02060288.jpg']

    filename = Path(img['file_name']).name #obej >>> 'objects365_v2_02060288.jpg'
    """
    # #製作對應現有影像清單
    # dataset 是在images/後的資料包與patch
    # coco = COCO(annFile)
    
    img = coco.loadImgs(imgId)[0] 
    filename_list=img['file_name'].split('/')
    filename = filename_list[-1] # eg: 'objects365_v2_02060288.jpg'
    if filename[:-4] not in nosave: #檔名(無副檔名)沒有在不可儲存的清單
        dataset = filename_list[2] #eg:  'patch45'
        img_path=dataDir+'images/'+dataset
        objs=annotations_img(coco, dataset, img, classes,classes_ids,filename)
    # print(objs)
            

# %%建立給labelimg看的檔案
#create classes.txt
with open(anno_dir+"classes.txt","w") as file:
    for i in data[1:]:
        file.write(f"{i[1]}\n")
# %%create ALL file list
#創建目前所產生的清單
#先用照片當用程式自動篩選出來的清單全部
create_list=[]
for i in Path(img_dir).iterdir():
    create_list.append(i.name)
savetxt_name="ALL_{}_len_{}.txt".format(cls,len(create_list))
with open(savepath+savetxt_name,"w") as file:
    for i in create_list:
        file.write(f"{i}\n")


# %%
