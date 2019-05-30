import torch
#import nms    #we don't use this .so nms anymore. By lmw
import numpy as np
import copy
from object_detection_nms_py import nms_py
from IPython import embed

def pth_nms(dets, thresh):
  """
  dets has to be a tensor
  """
  if not dets.is_cuda:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    #nms.cpu_nms(keep, num_out, dets, order, areas, thresh)

    return keep[:num_out[0]]
  else:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.sort(0, descending=True)[1]
    # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

    dets = dets[order].contiguous()

    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    # keep = torch.cuda.LongTensor(dets.size(0))
    # num_out = torch.cuda.LongTensor(1)
    #Get my own by py: lmw
    py_dets = copy.deepcopy(dets)
    order_py_idx = nms_py(py_dets[:,:-1], py_dets[:, -1], thresh)
    order_py = order[order_py_idx]
    #Get by .so:  I don't use this anymore, by lmw.
    #nms.gpu_nms(keep, num_out, dets, thresh)
    #order_c = order[keep[:num_out[0]].cuda()].contiguous()
    return order_py
    # return order[keep[:num_out[0]]].contiguous()

