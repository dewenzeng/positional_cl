import threading
import torch
import numpy as np
import torch.nn.functional as F

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scroes"""
    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union
            return

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def get(self, mode='mean'):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        Dice = 2.0 * self.total_inter / (np.spacing(1) + self.total_union + self.total_inter)
        if mode=='mean':
            mIoU = IoU.mean()
            Dice = Dice.mean()
            return pixAcc, mIoU, Dice
        else:
            return pixAcc, IoU, Dice
 
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        return

def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    # predict = torch.max(output, 1)[1]
    predict = torch.argmax(output, dim=1)
    # predict = output

    # label: 0, 1, ..., nclass - 1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass): #只区分背景和器官: nclass = 2
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor                      #model的输出
        target: label 3D Tensor                       #label
        nclass: number of categories (int)            #只区分背景和器官: nclass = 2
    """
    predict = torch.max(output, dim=1)[1]                 #获得了预测结果
    # predict = output
    mini = 1                                                                          
    maxi = nclass-1                                   #nclass = 2, maxi=1
    nbins = nclass-1                                  #nclass = 2, nbins=1

    # label is: 0, 1, 2, ..., nclass-1
    # Note: 0 is background
    predict = predict.cpu().numpy().astype('int64')
    target = target.cpu().numpy().astype('int64')

    predict = predict * (target >= 0).astype(predict.dtype)
    intersection = predict * (predict == target)            # 得到TP和TN

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))  #统计(TP、TN)值为1的像素个数，获得TN
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))        #统计predict中值为1的像素个数，获得TN+FN
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))          #统计target中值为1的像素个数，获得TN+FP
    area_union = area_pred + area_lab - area_inter                              #area_union:TN+FN+FP
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union

# ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
def pixel_accuracy(im_pred, im_lab):

    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))

    return pixel_correct, pixel_labeled

def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image. 
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class-1,
                                        range=(1, num_class - 1))
    # Compute area union: 
    area_pred, _ = np.histogram(im_pred, bins=num_class-1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class-1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)

def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true*y_pred
    return ( 2. * intersection.sum(axis=axis) +smooth )/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth )

def dice_coefficient(input, target, smooth=1.0):
    assert smooth > 0, 'Smooth must be greater than 0.'

    probs = F.softmax(input, dim=1)

    encoded_target = probs.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = encoded_target.float()

    num = probs * encoded_target   # b, c, h, w -- p*g
    num = torch.sum(num, dim=3)    # b, c, h
    num = torch.sum(num, dim=2)    # b, c

    den1 = probs * probs           # b, c, h, w -- p^2
    den1 = torch.sum(den1, dim=3)  # b, c, h
    den1 = torch.sum(den1, dim=2)  # b, c

    den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
    den2 = torch.sum(den2, dim=3)  # b, c, h
    den2 = torch.sum(den2, dim=2)  # b, c

    dice = (2 * num + smooth) / (den1 + den2 + smooth) # b, c

    return dice.mean().mean()

def dice_iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, N_class):
    SMOOTH = 1e-5
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(dim=1).float()
    labels = labels.squeeze(dim=1).float()
    dice = torch.ones(N_class-1).float()
    iou = torch.ones(N_class-1).float()
    ## for test
    #outputs = torch.tensor([[1,1],[3,3]]).float()
    #labels = torch.tensor([[0, 1], [2, 3]]).float()

    for iter in range(1,N_class): ## ignore the background
        predict_temp = torch.eq(outputs, iter)
        label_temp = torch.eq(labels, iter)
        intersection = predict_temp & label_temp
        intersection = intersection.float().sum((1,2))
        union_dice = (predict_temp.float().sum((1,2)) + label_temp.float().sum((1,2)))
        union_iou = (predict_temp | label_temp).float().sum((1,2))
        # if intersection>0 and union>0:
        #     dice_temp = (2*intersection)/(union)
        # else:
        #     dice_temp = 0
        dice[iter-1] = ((2 * intersection + SMOOTH) / (union_dice + SMOOTH)).mean()
        iou[iter-1] = ((intersection + SMOOTH) / (union_iou + SMOOTH)).mean()
    return dice, iou  # Or thresholded.mean()

if __name__ == '__main__':
    outputs = torch.zeros(5, 256, 256)
    labels = torch.LongTensor(5, 1, 256, 256).random_(0, 5)
    dice, iou = dice_iou_pytorch(outputs=outputs, labels=labels, N_class=5)
    print(f'dice:{dice}, iou:{iou}')

