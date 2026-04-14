#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import math
import cv2
from random import shuffle                            
import matplotlib.pyplot as plt
import skimage.morphology
import keras.backend as K


# In[2]:


#pred1=torch.randint(0,2,(4,512,512))
#true1=torch.randint(0,2,(4,512,512))


# In[18]:


class AJI():
    
    def Remap_Label(self,pred, by_size=False):
        """
          Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
          not [0, 2, 4, 6]. The ordering of instances (which one comes first)
          is preserved unless by_size=True, then the instances will be reordered
          so that bigger nucler has smaller ID
          Args:
              pred    : the 2d array contain instances where each instances is marked
                        by non-zero integer
              by_size : renaming with larger nuclei has smaller id (on-top)
          """
        pred_id = list(np.unique(pred))
        pred_id.remove(0)
        if len(pred_id) == 0:
            return pred # no label
        if by_size:
            pred_size = []
            for inst_id in pred_id:
                size = (pred == inst_id).sum()
                pred_size.append(size)
                  # sort the id by size in descending order
            pair_list = zip(pred_id, pred_size)
            pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
            pred_id, pred_size = zip(*pair_list)
        new_pred = np.zeros(pred.shape, np.int32)
        for idx, inst_id in enumerate(pred_id):
            new_pred[pred == inst_id] = idx + 1
        return new_pred
    
    def give_value(self,true, pred):
        
        batch_num=true.shape[0]
        
        
        if torch.is_tensor(pred):
                  # convert the prediction tensor to numpy array
            pred1 = pred.detach().cpu().numpy()
        if torch.is_tensor(true):
                  # convert the target tensor to numpy array
            true1 = true.detach().cpu().numpy()

       
        

        AJI=[]
        for i in range(batch_num):
            
            
            
            label_true = skimage.morphology.label(true1[i])
            
            true=self.Remap_Label(label_true)

            label_pred = skimage.morphology.label(pred1[i])
            pred=self.Remap_Label(label_pred)

            true_id_list = list(np.unique(true))
            pred_id_list = list(np.unique(pred))

            true_masks = [None,]
            for t in true_id_list[1:]:
                t_mask = np.array(true == t, np.uint8)
                true_masks.append(t_mask)

            pred_masks = [None,]
            for p in pred_id_list[1:]:
                p_mask = np.array(pred == p, np.uint8)
                pred_masks.append(p_mask)

              # prefill with value
            pairwise_inter = np.zeros([len(true_id_list) -1,
                                        len(pred_id_list) -1], dtype=np.float64)
            pairwise_union = np.zeros([len(true_id_list) -1,
                                        len(pred_id_list) -1], dtype=np.float64)

              # caching pairwise
            for true_id in true_id_list[1:]: # 0-th is background
                t_mask = true_masks[true_id]
                pred_true_overlap = pred[t_mask > 0]
                pred_true_overlap_id = np.unique(pred_true_overlap)
                pred_true_overlap_id = list(pred_true_overlap_id)
                for pred_id in pred_true_overlap_id:
                    if pred_id == 0: # ignore
                        continue # overlaping background
                    p_mask = pred_masks[pred_id]
                    total = (t_mask + p_mask).sum()
                    inter = (t_mask * p_mask).sum()
                    pairwise_inter[true_id-1, pred_id-1] = inter
                    pairwise_union[true_id-1, pred_id-1] = total - inter
                if pairwise_inter.any():
                    pairwise_iou = (pairwise_inter / (pairwise_union + 1.0e-6)) +1.0e-6
                    
                      # pair of pred that give highest iou for each true, dont care
                      # about reusing pred instance multiple times
                    paired_pred = np.argmax(pairwise_iou, axis=1)
                    pairwise_iou = np.max(pairwise_iou, axis=1)
                      # exlude those dont have intersection
                    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
                    paired_pred = paired_pred[paired_true]
                      # print(paired_true.shape, paired_pred.shape)
                    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
                    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
                      #
                    paired_true = (list(paired_true + 1)) # index to instance ID
                    paired_pred = (list(paired_pred + 1))
                      # add all unpaired GT and Prediction into the union
                    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
                    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
                    for true_id in unpaired_true:
                        overall_union += true_masks[true_id].sum()
                    for pred_id in unpaired_pred:
                        overall_union += pred_masks[pred_id].sum()
                      #
                    aji_score = overall_inter / overall_union
                else:
                    
                    aji_score=0
                    
                
                    
            
            AJI.append(aji_score)
            
        return {'AJI':np.mean(AJI)}

class F1():
    def f1(self,y_true, y_pred):
        if torch.is_tensor(y_pred):
                  # convert the prediction tensor to numpy array
            y_pred = y_pred.detach().cpu().numpy()
        if torch.is_tensor(y_true):
                  # convert the target tensor to numpy array
            y_true = y_true.detach().cpu().numpy()
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
            predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return {"F1":2*((precision*recall)/(precision+recall+K.epsilon()))}


def WS(predicts):
    
    if torch.is_tensor(predicts):
        predicts=predicts.detach().cpu().numpy().astype('uint8')
    final=np.zeros(predicts.shape,dtype='float')
    
    #watershed
    for i in range(predicts.shape[0]):
        
        predict3=cv2.merge((predicts[i],predicts[i],predicts[i]))
        predict1=predicts[i]
        
        kernel=np.ones((3,3), np.uint8)
        opening= cv2.morphologyEx(predict1,cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg=cv2.dilate(opening, kernel, iterations=2)
        dist=cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret2, sure_fg= cv2.threshold(dist, 0.2* dist.max(), 255,0)
        sure_fg= np.uint8(sure_fg)
        unknown= cv2.subtract(sure_bg ,sure_fg)
        ret3, markers= cv2.connectedComponents(sure_fg)
        markers= markers+10
        np.unique(markers)
        markers[unknown==255]=0
        markers= cv2.watershed(predict3,markers)
        predict3[markers==-1]=[0,0,0]
        
        predict_one_layer=predict3[:,:,0]
        final[i]=predict_one_layer


    final=final.astype('uint8')
    final=torch.tensor(final)
    
    return final
