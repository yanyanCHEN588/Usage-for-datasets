# -*- coding: utf-8 -*-
"""
2021-06-14
@author: yan

因為後綴檔名是大寫JPG，但是檔案類型仍然是jpg，
但是因為我在做fair_split_tanval.py，是做只有純檔名的，我都是預設後面得副檔名都是小寫jpg。

將此份檔案放置在要轉換的資料夾

"""

#%% load module
from pathlib import Path
import os

# %%

for i in os.listdir('.'):
    os.rename(i,i[:-4]+".jpg")
# %%
