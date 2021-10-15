"""
for object365 dataset
    可以做成CSV清單

實測發現用pd的比較好
出來的CSV較無奇怪空格

"""
import json


with open("annotations\zhiyuan_objv2_train.json", 'r') as f:
    data = json.load(f)
    # 900w+
    print("anno count:", len(data["annotations"]))
    print("image count:", len(data["images"]))
    categories = {}
    for category in data['categories']:
        categories[category['id']] = category['name']

# print("write csv ...")
# import csv
# with open('dct.csv', 'w') as f:  
#     writer = csv.writer(f)
#     for k, v in categories.items():
#        writer.writerow([k, v])
print("write PDcsv ...")
# dictionary of lists  
import pandas as pd 
dict = {'name': categories}
df = pd.DataFrame(dict) 
    
# saving the dataframe 
df.to_csv('365_cls_name.csv')