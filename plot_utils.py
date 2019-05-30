'''#-*- coding:utf-8 -*-'''
import warnings
#warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pandas as pd
from matplotlib import rcParams
import os
import re
title_size = 8
#----------------------------fuchen AQI project
mark_points = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
mark_centers = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def svg_to_emf(who):
    os.system("inkscape -z %s.svg -M %s.emf"%who)
    return

def plot_scatter(data,labels,title,_plot=True):
    break_index = np.hstack((np.where(labels[:-1]-labels[1:]!=0)[0],data.shape[0]-1))
    if _plot:
        start = 0
        for i in break_index:
            plt.scatter(data[start:i,0],data[start:i,1])
            start = i+1
        plt.savefig("./pngs/%s.png"%title)
        plt.close()
    return

def plot_show(data,label_pred,centroids,inertia,title,_plot=True):
    if _plot:
        n_clusters = len(list(set(label_pred)))
        ax1 = plt.subplot(121)
        for i in range(n_clusters):
            ax1.plot(data[np.where(label_pred==i)][:,0],data[np.where(label_pred==i)][:,1],mark_points[i])
        if centroids is not None:
            for i in range(n_clusters):
                ax1.plot(centroids[i][0],centroids[i][1],mark_centers[i])
        ax2 = plt.subplot(122)
        ax2.imshow(label_pred.reshape(4*5,-1))
        plt.title(title)
        plt.savefig("./pngs/%s.png"%title)
        plt.close()
    return

def plot_radar(data,col_names,title,_plot):
    if _plot:
        import radar_chart   # radar plot is implemented in another .py
        N_features = data.shape[1]
        data = [list(col_names),data]
        radar_chart.start(data,N_features,title)
    else:
        pass
    return

def plot_mat(data,title,labels,shuffle_within_cluster=False,_plot=True):
    if _plot:
        index_ = np.array(list(range(labels.shape[0])))
        split_index = np.hstack((np.where(labels!=np.delete(np.insert(labels,0,999),-1))[0]-1,labels.shape[0]-1))
        if shuffle_within_cluster:
            for i in range(split_index.shape[0]-1):
                np.random.shuffle(index_[split_index[i]+1:split_index[i+1]+1])
            data = data[index_]
    
        plt.matshow(data, cmap=plt.cm.Blues)
        plt.title("%s"%title)
        plt.plot([data.shape[1],-1],[list(split_index+0.5),list(split_index+0.5)],color='red',alpha=0.65,lw=3)
        plt.xlim(0,data.shape[1])
        plt.ylim(0,data.shape[0])
        plt.savefig("./pngs/%s.png"%title)
        plt.close()
    else:
        pass
    return

def plot_pd_corr(finals, title, _plot=True):
    if _plot:
        plt.figure(figsize=(14, 12))
        plt.title("%s"%title, y=1.03, size=15)
        sns.heatmap(
            np.abs(finals.corr()),
            linewidths=0.1,
            vmax=1.0,
            vmin = 0,
            cmap='inferno',
            square=True,
            linecolor='white',
            annot=True)
        plt.xticks(rotation=90) 
        plt.yticks(rotation=360)
        plt.savefig("./pngs/%s.png"%title)
        plt.close()
    return

#--------------------wangke car_face project
def seat_prediction_vs_seat_target(outputs_left, labels_left, outputs_right, labels_right, outputs_merged, labels_merged):
#    outputs_left =  outputs_left.detach().cpu().numpy()
#    labels_left = labels_left.detach().cpu().numpy()
#    outputs_right =  outputs_right.detach().cpu().numpy()
#    labels_right = labels_right.detach().cpu().numpy()
    plt.clf()
    ax1 = plt.subplot(321)
    ax2 = plt.subplot(322)
    ax3 = plt.subplot(323)
    ax4 = plt.subplot(324)
    ax5 = plt.subplot(325)
    ax6 = plt.subplot(326)

    xticks = [str(i) for i in range(1,1+outputs_left.shape[1])]
    sns.heatmap(outputs_left, linewidths=0.1, vmin=0, vmax=1, cmap='Reds', square=True, linecolor='white', annot=True, ax=ax1, xticklabels=xticks)
    sns.heatmap(labels_left, linewidths=0.1, vmin=0, vmax=1, cmap='Reds', square=True, linecolor='white', annot=True, ax=ax2, xticklabels=xticks)
    sns.heatmap(outputs_right, linewidths=0.1, vmin=0, vmax=1, cmap='Greens', square=True, linecolor='white', annot=True, ax=ax3, xticklabels=xticks)
    sns.heatmap(labels_right, linewidths=0.1, vmin=0, vmax=1, cmap='Greens', square=True, linecolor='white', annot=True, ax=ax4, xticklabels=xticks)
    sns.heatmap(outputs_merged, linewidths=0.1, vmin=0, vmax=1, cmap='Blues', square=True, linecolor='white', annot=True, ax=ax5, xticklabels=xticks)
    sns.heatmap(labels_merged, linewidths=0.1, vmin=0, vmax=1, cmap='Blues', square=True, linecolor='white', annot=True, ax=ax6, xticklabels=xticks)

    ax1.set_title("Prediction", fontsize=title_size)
    ax2.set_title("Target", fontsize=title_size)
    ax3.set_title("Prediction", fontsize=title_size)
    ax4.set_title("Target", fontsize=title_size)
    ax5.set_title("Merged prediction", fontsize=title_size)
    ax6.set_title("Reality", fontsize=title_size)
    ax1.set_xlabel("SeatsNO");ax1.set_ylabel("LeftCamView")
    ax2.set_xlabel("SeatsNO");
    ax3.set_xlabel("SeatsNO");ax3.set_ylabel("RightCamView")
    ax4.set_xlabel("SeatsNO");
    ax5.set_xlabel("SeatsNO");ax5.set_ylabel("BothViewMerged")
    ax6.set_xlabel("SeatsNO");
    plt.draw()
    plt.pause(0.001)
    return    

def animation_of_train_and_validation_loss_and_seat_prediction_vs_seat_target(history_loss, v_history_loss, outputs, labels):
    reshaper = np.int(np.floor((np.sqrt(outputs.shape[0]))))
    outputs = outputs[:reshaper**2].reshape(reshaper, reshaper).cpu()
    labels = labels[:reshaper**2].reshape(reshaper, reshaper).cpu()
    plt.clf()
    #AX1
    ax1 = plt.subplot(211)
    xaxis_train = range(len(history_loss))
    xaxis_validate = range(len(v_history_loss))
    scope = 50
    ax1.scatter(xaxis_train[-scope:], 100*np.array(history_loss[-scope:]), color='k', s=1.5)
    ax1.scatter(xaxis_validate[-scope:], 100*np.array(v_history_loss[-scope:]), color='blue', s=1.5)
    ax1.plot(xaxis_train[-scope:], 100*np.array(history_loss[-scope:]), color='k', label='trainloss')
    ax1.plot(xaxis_validate[-scope:], 100*np.array(v_history_loss[-scope:]), color='blue', label="validateloss")
    ax1.set_xlabel("Epoches")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Train and Validation Loss", fontsize=title_size)
    #AX2
    ax2 = plt.subplot(245)
    xticks = [str(i) for i in range(1,1+outputs.shape[1])]

    sns.heatmap(outputs, linewidths=0.1, vmin=0, vmax=int(labels.max()), cmap='Blues', square=True, linecolor='white', annot=True, ax=ax2, xticklabels=xticks)
    ax2.set_title("Prediction on last validation batch", fontsize=title_size)
    ax2.set_xlabel("Sample");ax2.set_ylabel("Sample")
    #AX3
    ax3 = plt.subplot(246)
    sns.heatmap(labels, linewidths=0.1, vmin=0, vmax=int(labels.max()), cmap='Blues', square=True, linecolor='white', annot=True, ax=ax3, xticklabels=xticks)
    ax3.set_title("Ground truth\n(different number refers to different position)", fontsize=title_size)
    ax3.set_xlabel("Sample");ax3.set_ylabel("Sample")
    #AX4
    ax3 = plt.subplot(247)
    sns.heatmap(np.abs(labels-outputs), linewidths=0.1, vmin=0, vmax=int(labels.max()), cmap='Blues', square=True, linecolor='white', annot=True, ax=ax3, xticklabels=xticks)
    ax3.set_title("Errors values abs(labels-outputs)", fontsize=title_size)
    ax3.set_xlabel("Sample");ax3.set_ylabel("Sample")
    #AX5
    ax3 = plt.subplot(248)
    wrongs = abs(np.round(outputs)-labels)
    sns.heatmap(wrongs, linewidths=0.1, vmin=0, vmax=int(labels.max()), cmap='Reds', square=True, linecolor='white', annot=True, ax=ax3, xticklabels=xticks)
    ax3.set_title("Wrongs: %s"%wrongs.sum(), fontsize=title_size)
    ax3.set_xlabel("Sample");ax3.set_ylabel("Sample")

    plt.draw()
    plt.pause(0.001)
    return   

def animation_of_train_and_test_loss_and_curve_prediction_vs_curve_target( \
        history_loss, v_history_loss, \
        v_outputs, v_labels, v_t_axis, \
        t_outputs, t_labels, t_t_axis, \
        c_outputs, c_labels, c_t_axis, \
        infos, title):
    (epoch, RR, new_psi) = infos
    window_length = 40
    plt.clf()
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    #subplot 1:
    ax1.plot(np.log10(history_loss[-window_length:]), label="Trainning Loss")
    ax1.plot(np.log10(v_history_loss[-window_length:]), label="Validation Loss")
    ax1.legend()
    ax1.set_title("Logged Train and Validation Loss of %s PSI"%title, fontsize=title_size)
    ax1.set_xlabel("Number of epoch")
    ax1.set_ylabel("Loss over train and testset")
    if history_loss.shape[0]>window_length:
        ax1.set_xticklabels(np.linspace(epoch-window_length+1,\
			epoch+1,10).astype(int))

    #subplot 2:
    ax2.scatter(t_t_axis, t_outputs, \
            label = "Traindata Prediction", \
			s = 1, color = 'orange')
    ax2.scatter(t_t_axis, t_labels, \
			label="Traindata Labels",\
			s = 1, color = 'grey')
    ax2.scatter(v_t_axis, v_outputs, \
            label = "Testdata Prediction",\
			s = 1, color = 'blue',alpha=0.5)
    ax2.scatter(v_t_axis ,v_labels,\
            label="Testdata Labels",\
			s = 1, color = 'k', alpha=0.5)
    ax2.legend()
    ax2.set_title("Prediction of probability increments",\
			fontsize=title_size)
    ax2.set_xlabel("LogT time")
    ax2.set_ylabel("Probability of Damage")

    #subplot3, culmulative sum of alpha
    ax3.scatter(c_t_axis ,c_labels,\
            label="Labels for baseline data",\
			s = 1, color = 'grey', alpha=0.5)
    ax3.scatter(c_t_axis, c_outputs, \
            label = "Culmulative probability",\
			s = 0.8, color = 'red')
    ax3.legend()
    ax3.set_title("Performance on culmulative probability RR:{0:.4f}%".\
			format(RR*100), fontsize=title_size)
    ax3.set_xlabel("LogT time")
    ax3.set_ylabel("Probability of Damage(%)")
    
#    embed()
#    import os
#    name=str(int(os.popen("ls *.png|wc -l").read()))
#    plt.savefig(name.zfill(3)+".png")
    plt.draw()
    plt.pause(0.001)
    return    

def damage_prediction_test_time( \
        damage_curve,\
        c_outputs, c_labels, c_t_axis, \
        n_outputs, n_labels, n_t_axis, \
        infos, status, outputname='out', filetype='pdf',\
        ):
    (markers, RR, psi, mean, cov,alters) = infos
    plt.rcParams["font.family"] = "Times New Roman"
#    plt.rcParams["font.family"] = "Fantasy"
#    rcParams.update({
#		    'font.family':'Times New Roman',
#			'font.sans-serif':['Liberation Sans'],
#				})
    plt.figure(figsize=(24,12))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    size = 21
    textsize = size
    headnamesize = size+10
    x_pos = 7.0
    n_outputs *= 1.045
    #LEFT FIG-----------------------------------------------------------------
    df = pd.DataFrame([])
    df_names = []
    for index,_ in enumerate(alters[:-1]):
        ax1.scatter(n_t_axis[alters[index]:alters[index+1]], \
				damage_curve[alters[index]:alters[index+1]],\
				label="Damage rate of:"+str(int(markers[index][-1]))+" psi, wood:("+str(markers[index][2])+","+str(markers[index][1])+")",\
		    s=4 , cmap='Blues')
        #save_data left_for_tao:
        case_string = str(int(markers[index][-1]))+" psi, wood:("+str(markers[index][2])+","+str(markers[index][1])+")"
        #df = pd.DataFrame({\
        #           case_string+'x':n_t_axis[alters[index]:alters[index+1]].reshape(-1),\
        #           case_string+'y':damage_curve[alters[index]:alters[index+1]].reshape(-1)\
        df = pd.concat([df, pd.DataFrame(n_t_axis[alters[index]:alters[index+1]].reshape(-1))],ignore_index=True,axis=1)
        df = pd.concat([df, pd.DataFrame(damage_curve[alters[index]:alters[index+1]].reshape(-1))],ignore_index=True,axis=1)
        df_names.append(case_string+'x')
        df_names.append(case_string+'y')
    df.columns = df_names
    df.to_csv(outputname+"_left.csv", line_terminator="\r\n")

    ax1.legend(prop={'size':size})
#    ax1.set_title("Prediction of probability increments",size=size+5)
    ax1.set_xlabel("Log-time (log hours)",size=size+5,labelpad=10)
    ax1.set_ylabel("Increments of Damage",size=size+5,labelpad=10)
    ax1.tick_params(labelsize=size)
    ax1.text(ax1.get_xbound()[0]+(ax1.get_xbound()[1]-ax1.get_xbound()[0])/25,\
			ax1.get_ybound()[1]-(ax1.get_ybound()[1]-ax1.get_ybound()[0])/15,\
			"a)",fontsize=headnamesize)

    #RIGHT FIG----------------------------------------------------------------
    ax2.scatter(c_t_axis ,c_labels,\
            label = "Basic",\
			s = 3, color = 'grey')
    ax2.scatter(n_t_axis, n_outputs, \
			label = status,\
			s = 3, color = 'red')
    #save_data right_for_tao:
    names = []
    right_datas = pd.DataFrame([])
    seqs = np.hstack((np.insert(np.where(np.diff(c_t_axis)<0),0,-1),np.array(-1)))
    for index,seq in enumerate(seqs): 
        part1 = pd.DataFrame([])
        part2 = pd.DataFrame([])
        part1_red = pd.DataFrame([])
        part2_red = pd.DataFrame([])
        if index==len(seqs)-1:
           break
        part1["x_gray_of_line_%s"%index] = c_t_axis[seqs[index]+1:seqs[index+1]]
        right_datas = pd.concat([right_datas, part1], ignore_index=True, axis=1)
        part2["y_gray_of_line_%s"%index] = c_labels[seqs[index]+1:seqs[index+1]]
        right_datas = pd.concat([right_datas, part2], ignore_index=True, axis=1)
        names.append("x_gray_of_line_%s"%index)
        names.append("y_gray_of_line_%s"%index)
        if status != "Generalized":
            part1_red['x_red_of_line_%s'%index] = n_t_axis[seqs[index]+1:seqs[index+1]]
            right_datas = pd.concat([right_datas, part1_red], ignore_index=True, axis=1)
            part2_red['y_red_of_line_%s'%index] = n_outputs[seqs[index]+1:seqs[index+1]]
            right_datas = pd.concat([right_datas, part2_red], ignore_index=True, axis=1)
            names.append("x_red_of_line_%s"%index)
            names.append("y_red_of_line_%s"%index)
    if status == "Generalized":
        part1_red = pd.DataFrame([])
        part2_red = pd.DataFrame([])
        part1_red['x_red_of_generalized'] = n_t_axis
        right_datas = pd.concat([right_datas, part1_red], ignore_index=True, axis=1)
        part2_red['y_red_of_generalized'] = n_outputs
        right_datas = pd.concat([right_datas, part2_red], ignore_index=True, axis=1)
        names.append("x_red_of_generalized")
        names.append("y_red_of_generalized")
    right_datas.columns = names
    right_datas.to_csv(outputname+"_right.csv", line_terminator="\r\n")
    for marker in markers:
#        ax1.scatter(7.5, marker[0], s=15, color='green')
        if int(marker[-1])==3000:
            ax2.text(x_pos, marker[0], \
				str(marker[-1])+", ("+str(marker[2])+","+str(marker[1])+")",\
				fontsize=size*2/3)
        else:
            ax2.text(x_pos, marker[0], \
				str(marker[-1])+", ("+str(marker[2])+","+str(marker[1])+")",\
				fontsize=textsize)
    if status == "Generalized":
        ax2.text(x_pos, n_outputs[-1]-0.02, "%s, (%s,%s)"%(psi, mean, cov),\
				fontsize=textsize, color='red')
    ax2.legend(loc=(0.03,0.8),prop={'size':size})
#    ax2.set_title("Performance of M-D model (test RR:{0:.1f}%)".format(RR*100),\
#			size=size+5)
    ax2.set_xlabel("Log-time (log hours)",size=size+5,labelpad=10)
    ax2.set_ylabel("Probability of Damage",size=size+5,labelpad=10)
    ax2.tick_params(labelsize=size)
    ax2.text(ax2.get_xbound()[0]+(ax2.get_xbound()[1]-ax2.get_xbound()[0])/25,\
			ax2.get_ybound()[1]-(ax2.get_ybound()[1]-ax2.get_ybound()[0])/15,\
			"b)",fontsize=headnamesize)
    plt.savefig("%s.%s"%(outputname,filetype),format='%s'%filetype)
    plt.show()
    return    


def prediction_vs_real( \
                t_outputs_close_loop, t_outputs_open_loop,\
                t_labels, t_t_axis, \
                infos, title):
    RR = infos
    window_length = 40
    plt.clf()
    ax1 = plt.subplot(111)
    ax1.scatter(t_t_axis, t_outputs_close_loop, \
            label = "PredictionCloseLoop", \
			s = 1, color = 'red')
    ax1.scatter(t_t_axis, t_outputs_open_loop, \
            label = "PredictionOpenLoop", \
			s = 1, color = 'green',alpha=0.5)
    ax1.scatter(t_t_axis, t_labels, \
			label="Labels",\
			s = 1, color = 'k')
    ax1.legend()
    ax1.set_title("Probability predictionof {} PSI, RR:{}%".\
			format(title, np.round(RR*100,4)),\
			fontsize=title_size)
    ax1.set_xlabel("Number of sampled data")
    ax1.set_ylabel("Probability")
    
    plt.show()
    return    

class dict_upgrader:     #Upgrade dict to class
    def __init__(self, **entries):
        self.__dict__.update(entries)

def sort_labels_and_handles_by_labels(ax):                                  
    handles, labels = ax.get_legend_handles_labels()
    indexer = np.array(np.argsort([float(label.split(':')[-1]) for label in labels]))
    labels, handles = list(np.array(labels)[indexer]),list(np.array(handles)[indexer])
    return labels, handles

# evaluation of cross validation plot:
def historical_best(seq):
    seq = list(seq)
    for index,value in enumerate(seq[:-1]):
        if seq[index+1]>seq[index]:
            seq[index+1] = min(seq[:index+1])
    return np.array(seq)
def sort_on_name(model_files, args):
    sorted_files =\
        list(np.array(model_files)[np.argsort([float(re.findall("%s_(.*?)_"%args.who.capitalize(),\
        tmp)[0]) for tmp in model_files])])
    return sorted_files

#===
#    cbar = plt.colorbar(tmp)
#    cbar.set_ticks(classid.values())
#    cbar.set_ticklabels(classid.keys())

def scatter3d(single_class, class_i,ax,lb,ub):
    ###plot points:
    size = 1 if class_i == "DontCare" else 3
    if size == 1:
        ###continue
        pass
    tmp=ax.scatter3D(single_class[:,0], single_class[:,1], single_class[:,2],\
			s=single_class[:,3]*4, label=class_i)
    ax.set_xlim(lb,ub)
    ax.set_ylim(lb,ub)
    ax.set_zlim(lb,ub)
   #ax.set_zbound(min(ax.xy_dataLim.xmin,ax.xy_dataLim.ymin),max(ax.xy_dataLim.xmax,ax.xy_dataLim.ymax))

    ax.set_xlabel("x");ax.set_ylabel("y")
    ax.legend()
    return

def scatter_flat(points):
#    ax0 = plt.subplot(411)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

#    ax0.tricontour(points[:,0],points[:,1],points[:,2],100,cmap=plt.cm.jet)
    ax1.scatter(points[:,0], points[:,1], s=1, c=points[:,2])
    ax1.set_title("Color of Height(m)")
    ax2.scatter(points[:,0], points[:,1], s=1, c=points[:,3])
    ax2.set_title("Color of Intensity")
    ax3.scatter(points[:,0], points[:,1], s=1, c=points[:,4])
    ax3.set_title("Color of class")
    return

def hist_distribution(data):
    ###plot bar distribution:
    data = pd.DataFrame(data)
    data.plot.hist(bins=250)
    return
