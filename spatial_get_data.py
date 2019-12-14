from collections import Counter
import time
import os,sys
import pandas as pd
import torch
import numpy as np
from IPython import embed
import json
from glob import glob
import matplotlib.pyplot as plt

with_name = False
#with_name = True

def xy_order_correction(inputs, small_cols, big_cols):
    for col1, col2 in zip(small_cols, big_cols):
        wrong_rows = inputs[col1]>inputs[col2]
        tmp = inputs[col1][wrong_rows]
        inputs[col1][wrong_rows] = inputs[col2][wrong_rows]
        inputs[col2][wrong_rows] = tmp
    return inputs

def generate_for_six_image_folder(json_list, head_names, ref_names):
    def fix_data(data):
        #temporary, fix rectangle generated as polygon, may delete:
        for idx,i in enumerate(data['shapes']): 
            if i['shape_type'] == 'polygon': 
                data['shapes'][idx]['points'] = [data['shapes'][idx]['points'][0], data['shapes'][idx]['points'][2]]
        return data
    paths = []
    inputs = np.empty(shape=(0, int(4*2+4+1)))   # ref_pos + head_pos + head_label
    for files in json_list:
        #within one image:
        with open(files) as f:
            data = json.load(f)
            data = fix_data(data)
        if len(data['shapes'])<=0:
            print("Nothing founded, continue")
            continue
        #First do counting, on all kinds of heads in detection list:
        num_of_refs = 0
        num_of_heads = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                num_of_heads += 1
            elif i['label'] in ref_names:
                num_of_refs += 1
            else:
                print("Unkown type %s"%i['label'])
                if i['label']=='head6':
                    sys.exit()
        if num_of_refs != 2:
            print("In file %s, Findding %s refs! Supposing two and only two!"%(files, num_of_refs))
            continue
        elif num_of_heads < 1:
            print("In file %s, Findding %s heads! Supposing at least one head(dirver)!"%(files, num_of_heads))
            continue
        else:
            print("In file %s, Findding %s heads and %s refs, a good sample"%(files, num_of_heads, num_of_refs))
        #开始形成数据：
        ref_positions = np.array([])
        for ref_name in ref_names:
            for i in data['shapes']:
                if i['label'] == ref_name:
                    ref_positions = np.hstack((ref_positions, np.array(i['points']).reshape(-1)))
        #After check, now we get data peer image:
        ref_and_head_position = np.hstack((np.tile(ref_positions, (num_of_heads,1)), np.tile([0,0,0,0, 0], (num_of_heads,1))))
        counter = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                ref_and_head_position[counter][4*2:-1] = np.array(i['points']).reshape(-1)
                ref_and_head_position[counter][-1] = int(i['label'][-1])
                paths.append(files)
                counter += 1
            else:
                pass
        inputs = np.vstack((inputs, ref_and_head_position))
    return inputs, paths

def generate_for_no_folder_images(json_list, head_names, ref_names):
    def fix_data(data):
        #temporary, fix rectangle generated as polygon, may delete:
        for idx,i in enumerate(data['shapes']): 
            if i['shape_type'] == 'polygon': 
                data['shapes'][idx]['points'] = [data['shapes'][idx]['points'][0], data['shapes'][idx]['points'][2]]
        return data
    paths = []
    inputs = np.empty(shape=(0, int(4*2+4+1)))   # ref_pos + head_pos + head_label
    for files in json_list:
        #within one image:
        with open(files) as f:
            data = json.load(f)
            data = fix_data(data)
        if len(data['shapes'])<=0:
            print("Nothing founded, continue")
            continue
        #First do counting, on all kinds of heads in detection list:
        num_of_refs = 0
        num_of_heads = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                num_of_heads += 1
            elif i['label'] in ref_names:
                num_of_refs += 1
            else:
                print("Unkown type %s"%i['label'])
                if i['label']=='head6':
                    sys.exit()
        if num_of_refs != 2:
            print("In file %s, Findding %s refs! Supposing two and only two!"%(files, num_of_refs))
            continue
        elif num_of_heads < 1:
            print("In file %s, Findding %s heads! Supposing at least one head(dirver)!"%(files, num_of_heads))
            continue
        else:
            print("In file %s, Findding %s heads and %s refs, a good sample"%(files, num_of_heads, num_of_refs))
        #开始形成数据：
        ref_positions = np.array([])
        for ref_name in ref_names:
            for i in data['shapes']:
                if i['label'] == ref_name:
                    ref_positions = np.hstack((ref_positions, np.array(i['points']).reshape(-1)))
        #After check, now we get data peer image:
        ref_and_head_position = np.hstack((np.tile(ref_positions, (num_of_heads,1)), np.tile([0,0,0,0, 0], (num_of_heads,1))))
        counter = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                ref_and_head_position[counter][4*2:-1] = np.array(i['points']).reshape(-1)
                ref_and_head_position[counter][-1] = int(i['label'][-1])
                paths.append(files)
                counter += 1
            else:
                pass
        inputs = np.vstack((inputs, ref_and_head_position))
    return inputs, paths

def generate_for_eight_image_folder(json_list, head_names, ref_names):
    paths = []
    inputs = np.empty(shape=(0, int(4*2+4+1)))   # ref_pos + head_pos + head_label
    for files in json_list:
        front, back = files[0], files[1]
        #within one image:
        data = {'shapes':[]} 
        with open(front) as f:
            data_front = json.load(f)
            for idx,i in enumerate(data_front['shapes']):
                if i['label'] in ref_names: 
                    data['shapes'].append(i) 
        with open(back) as f:
            data_back = json.load(f)
            for idx,i in enumerate(data_back['shapes']):
                if i['label'] in head_names: 
                    data['shapes'].append(i)
        if len(data['shapes'])<=0:
            print("Nothing founded, continue")
            continue
        #First do counting, on all kinds of heads in detection list:
        num_of_refs = 0
        num_of_heads = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                num_of_heads += 1
            elif i['label'] in ref_names:
                num_of_refs += 1
            else:
                print("Unkown type %s"%i['label'])
                if i['label']=='head6':
                    sys.exit()
        if num_of_refs != 2:
            print("In file %s, Findding %s refs! Supposing two and only two!"%(files, num_of_refs))
            continue
        elif num_of_heads < 1:
            print("In file %s, Findding %s heads! Supposing at least one head(dirver)!"%(files, num_of_heads))
            continue
        else:
            print("In file %s, Findding %s heads and %s refs, a good sample"%(files, num_of_heads, num_of_refs))
        #开始形成数据：
        ref_positions = np.array([])
        for ref_name in ref_names:
            for i in data['shapes']:
                if i['label'] == ref_name:
                    ref_positions = np.hstack((ref_positions, np.array(i['points']).reshape(-1)))
        #After check, now we get data peer image:
        #ref_and_head_position = np.hstack((np.tile(ref_positions, (num_of_heads,1)), np.tile([0,0,0,0, 0], (num_of_heads,1))))
        ref_and_head_position = np.hstack((np.tile(-ref_positions, (num_of_heads,1)), np.tile([0,0,0,0, 0], (num_of_heads,1))))   #使用取负来标注跨图像的信息
        counter = 0
        for i in data['shapes']:
            if i['label'] in head_names:
                ref_and_head_position[counter][4*2:-1] = np.array(i['points']).reshape(-1)
                ref_and_head_position[counter][-1] = int(i['label'][-1])
                paths.append(files)
                counter += 1
            else:
                pass
        inputs = np.vstack((inputs, ref_and_head_position))
    return inputs, paths

def get_ref_and_heads(data_path, args):
    image_folders = glob("%s/*"%data_path)
    eight_image_folders = []
    eight_image_json_list = []
    six_image_folders = []
    six_image_json_list = []
    ten_image_folders = []
    ten_image_json_list = []
    no_folder_json_list = glob("%s/*.json"%data_path)
    for i in image_folders: 
        num_of_pngs = glob(i+"/*.png") 
        if len(num_of_pngs)==8: 
            eight_image_folders.append(i)
        if len(num_of_pngs)==10: 
            ten_image_folders.append(i)
    six_image_folders = set(image_folders)   # - set(eight_image_folders)  #不用减8个的了，都可以用
    #6图的都可以用：
    for i in six_image_folders: 
        tmp = glob(i+"/*.json")
        if len(tmp)>0:
            six_image_json_list += tmp
    #10 tu is ok as 6 tu:
    for i in ten_image_folders: 
        tmp = glob(i+"/*.json")
        if len(tmp)>0:
            ten_image_json_list += tmp
    #8图的要做特殊跨图处理：
    for i in eight_image_folders: 
        tmp = glob(i+"/*.png")
        tmp.sort()
        back_idx = []
        front_idx = []
        for idx,j in enumerate(tmp):
            if j.find("BACK-")>0: 
                back_idx.append(idx)
            else:
                front_idx.append(idx)
        #孙东方存图命名顺序： （sort后） 左左左、右右右，右左（以前进方向为例）
        back_right = tmp[back_idx[0]].replace(".png",".json")
        back_left = tmp[back_idx[1]].replace(".png",".json")
        first_left = tmp[front_idx[0]].replace(".png",".json")
        first_right = tmp[front_idx[3]].replace(".png",".json")

        if os.path.exists(first_left) and os.path.exists(back_left):
            eight_image_json_list.append([first_left, back_left])
        if os.path.exists(first_right) and os.path.exists(back_right):
            eight_image_json_list.append([first_right, back_right])

    ref_names = ['angle', 'top', 'angle_r', 'top_r']   #TODO：参考物的名称
    #head_names = ["head1", "head2", "head3", "head4", "head5" ]      #TODO: 类别必须形如 head1 head2 等，其中 1 2 表示固定的位置，不允许改变
    head_names = ["head1", "head2", "head3", "head4" ]      #Trying to ignore 5 here.

    #六张图、八张图将根据不同的策略尝试生成数据：
    print("Generating data for no folder image...")
    time.sleep(0.5)
    inputs, paths = generate_for_no_folder_images(no_folder_json_list, head_names, ref_names)

    #print("Generating data for eight image foler...")
    #time.sleep(0.5)
    #inputs, paths = generate_for_eight_image_folder(eight_image_json_list, head_names, ref_names)

    #print("Generating data for six image foler...")
    #time.sleep(0.5)
    #inputs, paths = generate_for_six_image_folder(six_image_json_list, head_names, ref_names)

    #print("Generating data for ten image foler...")
    #time.sleep(0.5)
    #generate_for_ten_image_folder = generate_for_six_image_folder
    #inputs, paths = generate_for_ten_image_folder(ten_image_json_list, head_names, ref_names)

    #Put into dataframe with name
    inputs = pd.DataFrame(inputs, index=None)
    inputs.columns = ["ref1_x1", "ref1_y1", "ref1_x2", "ref1_y2", \
                      "ref2_x1", "ref2_y1", "ref2_x2", "ref2_y2", \
                      "heads_x1", "heads_y1", "heads_x2", "heads_y2", "label"]
    #Random expansion:
    part_new = np.tile(inputs,(args.expander,1))
    paths = paths * (args.expander+1)
    random_factor = np.random.uniform(0.98, 1.02, size=part_new.shape)
    inputs_new = part_new*random_factor
    inputs = inputs.append(pd.DataFrame(inputs_new, columns=inputs.columns))
    inputs = pd.DataFrame(inputs.values, index=None, columns=inputs.columns)
    inputs['label'] = np.round(inputs['label'])
    small_cols = ["ref1_x1", "ref1_y1", "ref2_x1", "ref2_y1", "heads_y1", "heads_x1"]   #这些应该是小的列
    big_cols = ["ref1_x2", "ref1_y2", "ref2_x2", "ref2_y2", "heads_y2", "heads_x2"]  #这些应该是大的列
    inputs = xy_order_correction(inputs, small_cols, big_cols)  #如果不是，则correct
    return inputs, paths

def mannual_feature(inputs, paths, args):
    #Compute new features:
    inputs['heads_y'] = 0.5 * (inputs['heads_y2']+inputs['heads_y1'])
    inputs['heads_x'] = 0.5 * (inputs['heads_x2']+inputs['heads_x1'])
    inputs['ref1_y'] = 0.5 * (inputs['ref1_y2']+inputs['ref1_y1'])
    inputs['ref1_x'] = 0.5 * (inputs['ref1_x2']+inputs['ref1_x1'])
    inputs['ref2_y'] = 0.5 * (inputs['ref2_y2']+inputs['ref2_y1'])
    inputs['ref2_x'] = 0.5 * (inputs['ref2_x2']+inputs['ref2_x1'])
    inputs['y_ratio'] = (inputs['heads_x']-inputs['ref1_x'])/(inputs['ref1_x']-inputs['ref2_x'])
    inputs['x_ratio'] = (inputs['heads_y']-inputs['ref1_y'])/(inputs['ref1_x']-inputs['ref2_x'])
    #inputs = (inputs-inputs.mean())/inputs.std()
    colors = plt.cm.Paired(np.linspace(1,0,2+len(set(inputs['label']))))
    if args.visualization:
        for line_idx in range(len(inputs)):
            this_line = inputs.iloc[line_idx]
            if int(this_line['label']) in [1,2,3,4]:
                plt.scatter(this_line['x_ratio'], this_line['y_ratio'], color=colors[int(this_line['label'])], s=7)
                if with_name:
                    plt.text(this_line['x_ratio'], this_line['y_ratio'], paths[line_idx].split('/')[-1].strip('.json'), size=6)
        plt.legend()
    #plt.draw()
    #plt.pause(1)
    plt.show()
    one_hot = np.zeros((inputs.shape[0],5))
    one_hot_index = (np.arange(inputs.shape[0]), inputs['label'].values.astype(int))
    one_hot[(np.arange(inputs.shape[0]), inputs['label'].values.astype(int)-1)] = 1
    for _i in range(one_hot.shape[1]):
        inputs['seat%s'%int(_i+1)] = one_hot[:,_i].astype(int)
    #inputs = inputs.drop(['ref1_x','ref1_y', 'ref2_x', 'ref2_y', 'heads_x', 'heads_y', 'y_ratio', 'x_ratio'], axis=1)
    return inputs, paths


'''
def get_data():  #Train and validate
    def make_noise(data, nn_input_dims):
        num_of_noises = 3    #max=12
        num_of_features = nn_input_dims/4
        num_of_invisible_parts = 6
        for i in range(parts_and_seats_taken.__len__()):
            x = np.random.randint(0, num_of_features, num_of_invisible_parts)
            for j in x:
                parts_and_seats_taken[i][j*4:(j+1)*4] = 0
        return data
    #Get data:
    titles = open(train_filename).readline().split()
    parts_and_seats_taken = np.loadtxt(train_filename)
    parts_and_seats_taken = make_noise(parts_and_seats_taken, nn_input_dims)
    print("Dumping last %s data"%(parts_and_seats_taken.shape[0]%(nn_input_dims*batch_size)))
    parts_and_seats_taken = parts_and_seats_taken[:-(parts_and_seats_taken.shape[0]%(nn_input_dims*batch_size))]
    parts = parts_and_seats_taken[:,:-5].reshape(-1, batch_size, nn_input_dims)
    seats = parts_and_seats_taken[:,-5:].reshape(-1, batch_size, nn_output_dims)
    indexer = np.linspace(0, parts.shape[0]-1, parts.shape[0], dtype = int)
    np.random.shuffle(indexer)
#    print("Shuffled batched data:",indexer)
    train_pool = zip(parts[indexer[:round(indexer.shape[0]*train_ratio)]], seats[indexer[:round(indexer.shape[0]*train_ratio)]])
    val_pool  = zip(parts[indexer[-round(indexer.shape[0]*test_ratio):]], seats[indexer[-round(indexer.shape[0]*test_ratio):]])
    test_pool = val_pool
    return train_pool, val_pool, test_pool

def get_predict_data():
    def make_noise(data, nn_input_dims):
        num_of_noises = 3    #max=12
        num_of_features = nn_input_dims/4
        num_of_invisible_parts = 6
        for i in range(parts_and_seats_taken.__len__()):
            x = np.random.randint(0, num_of_features, num_of_invisible_parts)
            for j in x:
                parts_and_seats_taken[i][j*4:(j+1)*4] = 0
        return data
    #Get data:
    titles = open(predict_filename).readline().split()
    parts_and_seats_taken = np.loadtxt(predict_filename)
    parts_and_seats_taken = make_noise(parts_and_seats_taken, nn_input_dims)
    print("Dumping last %s data"%(parts_and_seats_taken.shape[0]%(nn_input_dims*predict_batch_size)))
    parts_and_seats_taken = parts_and_seats_taken[:-(parts_and_seats_taken.shape[0]%(nn_input_dims*predict_batch_size))]
    parts = parts_and_seats_taken[:,:-5].reshape(-1, predict_batch_size, nn_input_dims)
    seats = parts_and_seats_taken[:,-5:].reshape(-1, predict_batch_size, nn_output_dims)
    indexer = np.linspace(0, parts.shape[0]-1, parts.shape[0], dtype = int)
    np.random.shuffle(indexer)
#    print("Shuffled predict_batched data:",indexer)
    train_pool = zip(parts[indexer[:round(indexer.shape[0]*train_ratio)]], seats[indexer[:round(indexer.shape[0]*train_ratio)]])
    val_pool  = zip(parts[indexer[-round(indexer.shape[0]*test_ratio):]], seats[indexer[-round(indexer.shape[0]*test_ratio):]])
    test_pool = val_pool
    return train_pool, val_pool, test_pool
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/left_side_45/")
    parser.add_argument('--visualization', action="store_true", default=False)
    parser.add_argument('--expander', type=int, default=10)
    args = parser.parse_args()

    inputs, paths = get_ref_and_heads(args.data_path, args)
    inputs = mannual_feature(inputs, paths, args)
