import argparse
import os.path as osp
import shutil
import tempfile
import time
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result,init_detector

import sys,copy
from IPython import embed
import matplotlib.pyplot as plt
import sklearn.metrics
sys.path.append("/home/user/PersonDetection99/car_face/")

def coco_eval_substitude(outputs, dataset, show=False):
    #f1s = np.empty(shape=(0,len(dataset.cat_ids)))
    f1s = np.array(np.tile(0,len(dataset.cat_ids)))
    precisions = np.array(np.tile(0,len(dataset.cat_ids)))
    recalls = np.array(np.tile(1,len(dataset.cat_ids)))
    axis_conf = np.linspace(0.01, 0.99, 99)
    for SCORE_THR in axis_conf:
        for iou_threshold in [0.5]: #np.linspace(0.3,0.5,21):
            #print("Score threshold:%s, iou_threshold:%s"%(SCORE_THR, iou_threshold))
            f1, precision, recall = my_csv_eval(outputs, dataset, SCORE_THR, iou_threshold)
            f1s = np.vstack((f1s, f1))
            precisions = np.vstack((precisions, precision))
            recalls = np.vstack((recalls, recall))
    head_col = 2
    #most_good = np.where(f1s.sum(axis=1)==f1s.sum(axis=1).max())[0][-1]
    most_good = np.where(f1s[:,head_col].max()==f1s[:,head_col])[0][f1s[np.where(f1s[:,head_col].max()==f1s[:,head_col])].sum(axis=1).argmax()]
    print("***Please set score thr to %s, to get class(es) performance of %s"%(axis_conf[most_good], f1s[most_good]))
    #model selection:
    precisions = np.vstack((precisions, np.tile(1,len(dataset.cat_ids))))
    recalls = np.vstack((recalls, np.tile(0,len(dataset.cat_ids))))
    for i in range(f1s.shape[1]):
        plt.scatter(precisions[:,i], recalls[:,i], label=i)
        plt.title("PR-curve")
    plt.legend()
    if show:
        plt.show()
    AUC_area = 0
    for i in range(len(dataset.cat_ids)):
        #must sort to motonic..... alright...
        _tmp1_, _tmp2_ = precisions[:,i], recalls[:,i]
        seq = _tmp1_.argsort()
        tmp_AUC = sklearn.metrics.auc(_tmp1_[seq], _tmp2_[seq])
        AUC_area = AUC_area + tmp_AUC
        print("AUC performance on %s: %s"%(i, tmp_AUC))
    print("***Total AUC:", AUC_area)
    return AUC_area

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap_area(recall, precision):     #Name modified by LMW
    """ Compute the average precision, given the recall and precision CURVES.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def my_csv_eval(outputs, dataset, SCORE_THR, iou_threshold):
    #使用旧版（retinanet的csveval），先准备其需要的detections：
    all_detections = copy.deepcopy(outputs)
    for idx1,detections in enumerate(all_detections):
        for idx2,i in enumerate(detections): 
            detections[idx2] = i[np.where(i[:,-1]>SCORE_THR)]
        all_detections[idx1] = detections
    
    #使用旧版（retinanet的csveval），再准备其需要的annotations：
    all_annotations = []
    for i in range(len(dataset)):
        tmp_annotation_now = dataset.get_ann_info(i)['bboxes'][np.argsort(dataset.get_ann_info(i)['labels'])]
        class_idx_now = dataset.get_ann_info(i)['labels'][np.argsort(dataset.get_ann_info(i)['labels'])]
        annotation_now = list(np.tile(0,len(dataset.cat_ids)))
        for idx,i in enumerate(class_idx_now): 
            try: 
                annotation_now[i-1] = np.vstack((annotation_now[i-1], tmp_annotation_now[idx]))
            except: 
                annotation_now[i-1] = tmp_annotation_now[idx].reshape(1,-1)
        _fill_idx = set(dataset.cat_ids) - set(class_idx_now)
        for i in _fill_idx:
            annotation_now[i-1] = np.empty(shape=(0,len(dataset.cat_ids)))
        all_annotations.append(annotation_now)

    #以下由object_detection_csv_eval改写，复制重写了，不调用了太乱了。
    counters = {}
    average_precisions = {}
    f1_scores = {}
    recalls = {}
    precisions = {}
    for label in dataset.cat_ids:
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        for i in range(len(dataset)):
            detections           = all_detections[i][label-1]
            annotations          = all_annotations[i][label-1]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            counters[dataset.cat2label[label]] = 0
            average_precisions[dataset.cat2label[label]] = 0
            f1_scores[dataset.cat2label[label]] = 0
            recalls[dataset.cat2label[label]] = 0
            precisions[dataset.cat2label[label]] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute precision_single and recall_single by LMW
        if len(true_positives)==0:
            precision_single = 0
            recall_single = 0
        else:
            precision_single = true_positives[-1]/(true_positives[-1]+false_positives[-1])
            recall_single = true_positives[-1]/num_annotations
        # compute f1 score by LMW:
        if recall_single+precision_single!=0:
            f1_score = (2*recall_single*precision_single)/(recall_single+precision_single)
        else:
            f1_score = 0
        f1_scores[dataset.cat2label[label]] = np.round(f1_score,3)
        precisions[dataset.cat2label[label]] = np.round(precision_single,3)
        recalls[dataset.cat2label[label]] = np.round(recall_single,3)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision 这里原作者是根据precision和recall曲线的area来计算area的，和我使用conf点做出来的一样。
        average_precision  = _compute_ap_area(recall, precision)
        average_precisions[dataset.cat2label[label]] = np.round(average_precision,3)
        counters[dataset.cat2label[label]] = num_annotations  
    #print('\nmAP:')
    #for label in range(generator.num_classes()):
    #    label_name = dataset.cat2label[label]
    #    print('{}: {}'.format(label_name, average_precisions[label][0]))
    #我自己的recalls和他用area算得一样，这里借用原作者的，只为顺便显示一下“多少个”。
    #print("Counters:", counters)
    #print("Precising:", precisions, "\nRecalling:", average_precisions, "\nF1_at_this_conf:", f1_scores, np.array(list(f1_scores.values())).sum())
    return np.array(list(f1_scores.values())), np.array(list(precisions.values())), np.array(list(recalls.values()))

def single_gpu_frame_detection(model, _data, CONFIDENCE_THRESHOLD, show=False):
    #print("Using confidence score:", CONFIDENCE_THRESHOLD)
    model.eval()
    start = time.time()
    result = inference_detector(model, _data)
    elapsed_time = time.time()-start
    labels = np.concatenate([
                           np.full(bbox.shape[0], i, dtype=np.int32)
                          for i, bbox in enumerate(result)])
    #labels = labels + 1
    bboxes = np.vstack(result)
    x1s = np.array([], dtype=int)
    y1s = np.array([], dtype=int)
    x2s = np.array([], dtype=int)
    y2s = np.array([], dtype=int)
    scores = np.array([])
    label_names = []
    for label,bbox in zip(labels,bboxes):
        threshold = bbox[-1]
        if threshold < CONFIDENCE_THRESHOLD:
            continue
        x1,y1 = bbox[0],bbox[1]
        x2,y2 = bbox[2],bbox[1]
        x3,y3 = bbox[2],bbox[3]
        x4,y4 = bbox[0],bbox[3]
        x1s = np.hstack((x1s, x1))
        y1s = np.hstack((y1s, y1))
        x2s = np.hstack((x2s, x3))
        y2s = np.hstack((y2s, y3))
        scores = np.hstack((scores, threshold))
        label_names.append(label)
    return x1s,y1s,x2s,y2s, scores,label_names, elapsed_time

def single_gpu_test(model, data_loader, SCORE_THR, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        #sys.stdout.flush()
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        if show:
            print("\nTesting on %s"%data_loader.dataset.img_prefix+data_loader.dataset.img_infos[i]['filename'])
            if SCORE_THR is not None:
                model.module.show_result(data, result, dataset.img_norm_cfg, score_thr=SCORE_THR)   #score_thr is here, hidden
            else:
                model.module.show_result(data, result, dataset.img_norm_cfg)   #score_thr is here, hidden

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    SCORE_THR = 0.49

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model.CLASSES = ("1angle?","2top?","3head?")
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, SCORE_THR, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()   #Distribution info
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                #coco_eval(result_file, eval_types, dataset.coco)
                coco_eval_substitude(outputs, dataset, args.show)
            else:
                if not isinstance(outputs[0], dict):
                    #car_face走着里::::::::::::::::::::::::
                    result_file = args.out + '.json'
                    #results2json(dataset, outputs, result_file)
                    #coco_eval(result_file, eval_types, dataset.coco)
                    coco_eval_substitude(outputs, dataset, args.show)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        #outputs_ = [out[name] for out in outputs]
                        outputs_ = []
                        for out in outputs:
                            outputs_.append(out[name])
                        result_file = args.out + '.{}.json'.format(name)
                        #results2json(dataset, outputs_, result_file)
                        #coco_eval(result_file, eval_types, dataset.coco)
                        coco_eval_substitude(outputs, dataset, args.show)


if __name__ == '__main__':
    main()

