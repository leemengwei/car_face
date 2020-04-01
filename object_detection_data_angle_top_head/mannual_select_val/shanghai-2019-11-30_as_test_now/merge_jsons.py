import json
from IPython import embed

data1 = json.load(open('./output_json1.json', 'r'))
data2 = json.load(open('./output_json2.json', 'r'))

assert data1['categories'] == data2['categories']


len1 = len(data1['images'])
idx_to_add_peer_img = len1
len2 = len(data1['annotations'])
idx_to_add_peer_box = len2

for i in data2['images']: 
    i['id'] = i['id']+idx_to_add_peer_img
for i in data2['annotations']: 
    i['image_id'] = i['image_id']+idx_to_add_peer_img
    i['id'] = i['id']+idx_to_add_peer_box



merge_dict = {}
merge_dict['images'] = data1['images'] + data2['images']
merge_dict['annotations'] = data1['annotations'] + data2['annotations']
merge_dict['categories'] = data1['categories']


json_fp = open("output_merge_json.json", 'w')                         
json_str = json.dumps(merge_dict,indent=4, separators=(',', ': '))
json_fp.write(json_str)
json_fp.close()

print("Done")
