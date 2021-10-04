# -*- coding: utf-8 -*-
"""
2021-10-04
@author: yan

假設已經人工過簡單濾完後的開發過濾標註檔名程式

ALL (*.jpg) :原始清單早已在生標註時生成
DEL (*.xml) :需要刪除照片標註的
CHG (*.json) :已經手動更改過標註框框  先不管#後期要使用此去讀取修改的類別並寫入新標註
SAV (*.txt) :這裡的目的是 希望記錄 SAV = ALL - DEL -CHG  代表此些圖片是程式就能過濾很好的


以後資料不夠時再去管 CHG的內容
NoSAV = DEL + CHG
"""
#%% load module
from pathlib import Path


#%%json coco
###-----------config----------
filterpath=Path("pick_test")
labelformat = 'txt'
img_dir=filterpath/'images' 
anno_dir=filterpath/'labels'

cls="Toothbrush"

allpath=filterpath/"ALL_Toothbrush_len_12.txt"
#讀取ALL.txt清單 一開始創立的
all_list=[]
with open(allpath) as f:
    all_list=f.read().split('.jpg\n')[:-1]

# %%讀取xml檔名並存檔與計算數量
del_list=[]
for xml in Path(anno_dir).glob('*.xml'):
    del_list.append(xml.name[:-4])
savetxt_name=filterpath / "DEL_{}_len_{}.txt".format(cls,len(del_list))
with open(savetxt_name,"w") as file:
    for i in del_list:
        file.write(f"{i}\n")

# %%讀取json檔名並存檔與計算數量
chg_list=[]
for xml in Path(anno_dir).glob('*.json'):
    chg_list.append(xml.name[:-5])
savetxt_name=filterpath / "CHG_{}_len_{}.txt".format(cls,len(chg_list))
with open(savetxt_name,"w") as file:
    for i in chg_list:
        file.write(f"{i}\n")


#%% SAV = ALL - DEL -CHG

for i in del_list:
    all_list.remove(i)
for i in chg_list:
    all_list.remove(i)
savetxt_name=filterpath / "SAV_{}_len_{}.txt".format(cls,len(all_list))
with open(savetxt_name,"w") as file:
    for i in all_list:
        file.write(f"{i}\n")

# %%NoSAV = DEL + CHG
savetxt_name=filterpath / "NoSAV_{}_len_{}.txt".format(cls,len(del_list)+len(chg_list))
with open(savetxt_name,"w") as file:
    for i in del_list:
        file.write(f"{i}\n")
    for i in chg_list:
        file.write(f"{i}\n")
