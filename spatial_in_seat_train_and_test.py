from IPython import embed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from time import sleep
import time
import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn as nn
from tqdm import tqdm
#parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import plot_utils
import spatial_model
import spatial_get_data as get_data
from sklearn import metrics
from config import *

def frame_in_seat(model, test_data):
    test_data = test_data.cuda()
    #start_time = time.time()
    output_probabilities = model(test_data)
    #print(time.time()-start_time)
    output_seat = output_probabilities.argmax(dim=1)+1
    return output_probabilities.detach().cpu().numpy(), output_seat.detach().cpu().numpy()[0]

def train(model, train_loader, optimizer, epoch):
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        #data, target = data.cuda(), target.long().cuda().view(-1, NUM_OF_SEATS_PEER_CAR)
        data, target = data.cuda(), target.float().cuda().view(-1, NUM_OF_SEATS_PEER_CAR)
        optimizer.zero_grad()
        output = model(data)
        #loss = torch.nn.functional.cross_entropy(output, target)  #之前看到说如果pytorch使用交叉熵则自动会做hot的code。
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        LOSS += loss.item() # pile up all loss
        pbar.set_description('Train Epoch: {}, datapiece:[{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
    train_loss_mean = LOSS/len(train_loader.dataset)
    print("Train Epoch: {} LOSS:{:.1f}, Average loss: {:.8f}".format(epoch, LOSS, train_loss_mean))
    return train_loss_mean

def validate(model, validate_loader):
    model.eval()
    LOSS = 0
    pbar = tqdm(validate_loader)
    confusions = np.zeros(shape=(5,5))
    for idx, (data, target) in enumerate(pbar):
        #data, target = data.cuda(), target.long().cuda().view(-1, NUM_OF_SEATS_PEER_CAR)
        data, target = data.cuda(), target.float().cuda().view(-1, NUM_OF_SEATS_PEER_CAR)
        output_probabilities = model(data)
        #LOSS += torch.nn.functional.cross_entropy(output_probabilities, target).item() # pile up all loss
        LOSS += torch.nn.functional.mse_loss(output_probabilities, target).item() # pile up all loss#之前看到说如果pytorch使用交叉熵则自动会做hot的code。
        pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
        output_seat = output_probabilities.argmax(dim=1)+1
        target_seat = target.argmax(dim=1)+1
        confusion_tmp = metrics.confusion_matrix(output_seat, target_seat, labels=[1,2,3,4,5])   #应该先true 再pred
        confusions = confusions + confusion_tmp
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    precision = confusions.diagonal().sum()/confusions.sum()
    print('Validate set LOSS: {:.1f}, Average loss: {:.8f}'.format(LOSS, validate_loss_mean), "\nResult show:\n", confusions.astype(int), "\nPrecision:%s%%"%np.round(precision*100,3))
    last_output_seat = output_probabilities.argmax(dim=1)+1
    last_target_seat = target.argmax(dim=1)+1
    return validate_loss_mean, last_output_seat, last_target_seat

if __name__=="__main__":
    torch.manual_seed(44)
    parser = argparse.ArgumentParser(description = "Car Space Net...")
    parser.add_argument('-V', '--visualization', action="store_true", default=False)
    parser.add_argument('-R', '--restart', action="store_true", default=False)
    parser.add_argument('-SM', '--save_model', action="store_true", default=False)
    parser.add_argument('-RM', '--restart_model', type=str, default = None)
    parser.add_argument('-C', '--cuda_number', type=int, default = 0)
    parser.add_argument('-HD', '--hidden_depth', type=int, default = 5)
    parser.add_argument('-HW', '--hidden_width', type=int, default = 20)
    parser.add_argument('-EX', '--expander', type=int, default=1)
    parser.add_argument('-LR', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-E', '--epochs', type=int, default=100)
    parser.add_argument('-TR', '--test_ratio', type=float, default = 0.)
    parser.add_argument('-BS', '--batch_size', type=int, default=100)
    parser.add_argument('-DP', '--data_path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    #Basic settings:
    args.expander -= 1
    if args.epochs<0:
       args.restart = True
       assert args.restart_model is not None, "You're running test mode, must give restart model"
    device = torch.device("cuda", args.cuda_number)
    model_best_path = '%s/model_best.pt'%args.data_path.replace('data', 'model')
    print("Parameters given:", args)
    input_columns = ["ref1_x1", "ref1_y1", "ref1_x2", "ref1_y2", "ref2_x1", "ref2_y1", "ref2_x2", "ref2_y2", "heads_x1", "heads_y1", "heads_x2", "heads_y2"]
    target_columns = ["seat1", "seat2", "seat3", "seat4", "seat5"]
    #Get data and make data loader:
    if args.epochs>=0:
        data, paths = get_data.get_ref_and_heads(args.data_path, args)
        data, pahts = get_data.mannual_feature(data, paths, args)
        #embed()
        inputs = torch.FloatTensor(data[input_columns].values)
        targets = torch.FloatTensor(data[target_columns].values.reshape(-1,len(target_columns)))-1
        whole_dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_dataset, validate_dataset = torch.utils.data.random_split(whole_dataset, (len(whole_dataset)-int(len(whole_dataset)*args.test_ratio),int(len(whole_dataset)*args.test_ratio)))
        train_loader = torch.utils.data.DataLoader( 
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
    	        num_workers=4,
                pin_memory=True
                )
        validate_loader = torch.utils.data.DataLoader( 
                dataset=train_dataset, #validate_dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
    	        num_workers=4,
                pin_memory=True
                )
        assert (len(train_loader)!=0 and len(validate_loader)!=0), "Either of length of train/val loader is zero, reset batch_size to a proper value"

    #Get model:
    input_size = 12    #TODO
    hidden_size = args.hidden_width
    hidden_depth = args.hidden_depth
    output_size = 5   #TODO
    model = spatial_model.NeuralNet(input_size, hidden_size, hidden_depth, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    #Restart or not:
    if args.restart:
        model_restart_path = model_best_path if args.restart_model is None  else args.restart_model
        print("Restart trainning from %s"%model_restart_path)
        checkpoint = torch.load(model_restart_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss_history = checkpoint['train_loss_history']
        validate_loss_history = checkpoint['validate_loss_history']
        if args.visualization:
            print("Restarting Model paramters loaded: %s"%model_restart_path)
            plt.plot(np.log10(validate_loss_history), linewidth=2, label="validate loss")
            plt.plot(np.log10(train_loss_history), linewidth=2, label="train loss")
            plt.title("Training/Validating loss over epoch")
            plt.legend()
            plt.xlabel("Training epoches")
            plt.ylabel("Logged Training/Valdation loss")
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            #plt.show()
        else:
            pass
    else:
        epoch = 1
        train_loss_history = []
        validate_loss_history = []

    #TRAIN and Validate----------------------------------------------------------------------------------
    for epoch in range(epoch, args.epochs + 1):  # loop over the dataset multiple times
        train_loss = train(model, train_loader, optimizer, epoch)
        validate_loss, last_output, last_label = validate(model, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        #Saving stuff:
        ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'validate_loss_history': validate_loss_history,
                }
        if args.save_model:
            modeldir = "./spatial_model_both_side_%s/model_current.pt"%(args.data_path.strip('./').split('_')[-1])
            print("Saving model at", modeldir)
            torch.save(ckpt, modeldir)
        else:
            print("Not saving middle-process model... pass")
        if epoch>1:
            if validate_loss_history[-1] < np.array(validate_loss_history[:-1]).min():
                print("-------------------------------Best model, epoch: %s-----------------------------"%epoch)
                torch.save(ckpt, model_best_path)
            else:
                print("Not best: %s>%s, Keep running..."%(validate_loss_history[-1], np.array(validate_loss_history[:-1]).min()))
        else:
            pass   #Pass epoch 1
        #Visualization
        if args.visualization:
            plot_utils.animation_of_train_and_validation_loss_and_seat_prediction_vs_seat_target(train_loss_history, validate_loss_history, last_output, last_label)


    if args.epochs>=0:
        print("Train mode done, not runnning test_phase, exiting...")
        sys.exit()
    else:
        print("Epoch given<0, skip training and run test phase...")
        pass

    #TEST---------------------------------------------------------------------------------------------------
    data, paths = get_data.get_ref_and_heads(args.data_path, args)
    data, paths = get_data.mannual_feature(data, paths, args)
    inputs = torch.FloatTensor(data[input_columns].values)   #TODO dimension
    #embed()
    test_dataset = torch.utils.data.TensorDataset(inputs)
    test_loader = torch.utils.data.DataLoader( 
            dataset=test_dataset, 
            batch_size=1,
            shuffle=False,
            drop_last=True,
	        num_workers=4,
            pin_memory=True
            )
    assert len(test_loader)!=0, "Length of test loader is zero, reset batch_size to a proper value"

    positions = []
    model.eval()
    pbar = tqdm(test_loader)
    for idx, test_data in enumerate(pbar):
        test_data = test_data[0]
        position_probabilities, position = frame_in_seat(model, test_data)
        #position = int(np.argmax(position_probabilities.cpu().detach().numpy(), axis=1)+1)
        pbar.set_description('Test: [{}/{} ({:.0f}%)]'.format(idx*len(test_data), len(test_loader.dataset), 100.*idx/len(test_loader)))
        if args.visualization:
            path = paths[idx].strip('json')+'jpg'
            ref1_y1, ref1_x1, ref1_y2, ref1_x2 = data.iloc[idx]['ref1_y1'], data.iloc[idx]['ref1_x1'], data.iloc[idx]['ref1_y2'], data.iloc[idx]['ref1_x2']
            ref2_y1, ref2_x1, ref2_y2, ref2_x2 = data.iloc[idx]['ref2_y1'], data.iloc[idx]['ref2_x1'], data.iloc[idx]['ref2_y2'], data.iloc[idx]['ref2_x2']
            heads_y1, heads_x1, heads_y2, heads_x2 = data.iloc[idx]['heads_y1'], data.iloc[idx]['heads_x1'], data.iloc[idx]['heads_y2'], data.iloc[idx]['heads_x2']
            plt.clf()
            plt.imshow(plt.imread(path))
            currentAxis = plt.gca()
            colors = ['red', 'purple', 'cyan', 'orange', 'green', 'blue', 'yellow']
            plt.text(heads_x1, heads_y1, 'Position_'+str(position), color=colors[position])
            ref1_rect = matplotlib.patches.Rectangle((ref1_x1, ref1_y1), ref1_x2-ref1_x1, ref1_y2-ref1_y1, linewidth=1, edgecolor='r', facecolor='none')
            ref2_rect = matplotlib.patches.Rectangle((ref2_x1, ref2_y1), ref2_x2-ref2_x1, ref2_y2-ref2_y1, linewidth=1, edgecolor='r', facecolor='none')
            heads_rect = matplotlib.patches.Rectangle((heads_x1, heads_y1), heads_x2-heads_x1, heads_y2-heads_y1, linewidth=1, edgecolor=colors[position], facecolor=colors[position], alpha=0.65)
            currentAxis.add_patch(ref1_rect)
            currentAxis.add_patch(ref2_rect)
            currentAxis.add_patch(heads_rect)
            plt.savefig("./spatial_output_%s/%s.jpg"%('_'.join(args.data_path.split('/')[-2].split('_')[2:]), idx))
            plt.draw()
            plt.pause(0.01)
            positions.append(position)
        else:
            pass
