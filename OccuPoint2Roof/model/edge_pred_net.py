import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_stack_utils import *
from .model_utils import *
from scipy.optimize import linear_sum_assignment
from utils import loss_utils
import itertools

class EdgeAttentionNet(nn.Module):
    def __init__(self, model_cfg, input_channel):
        super().__init__()
        self.model_cfg = model_cfg
        self.freeze = False

        self.att_layer = PairedPointAttention(input_channel)
        num_feature = self.att_layer.num_output_feature

        self.shared_fc = LinearBN(num_feature * 2, num_feature)
        self.drop = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(num_feature, 1)

        if self.training:
            self.train_dict = {}
            self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(gamma=2.0, alpha=0.5))
            self.loss_weight = self.model_cfg.LossWeight

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        batch_idx = batch_dict['keypoint'][:, 0].long()
        point_fea = batch_dict['keypoint_features']

        keypoint_xyz = batch_dict['keypoint_xyz']
        backbone_xyz = batch_dict['backbone_xyz']
        backbone_fea = batch_dict['backbone_fea']

        idx = 0
        pair_idx_list = []
        pair_idx_list1, pair_idx_list2 = [], []
        phy_fea_list = []
        bin_label_list = []

        for i in range(batch_dict['batch_size']):
            mask = batch_idx == i
            tmp_idx = batch_idx[mask]
            num_pts = tmp_idx.shape[0]
            if num_pts <= 1:
                continue

            k_xyz = keypoint_xyz[mask]
            b_xyz = backbone_xyz[i]
            b_fea = backbone_fea[i]

            row, col = torch.triu_indices(num_pts, num_pts, offset=1, device=point_fea.device)
            pair_idx = torch.stack([row, col], dim=1).long()
            pair_idx_list.append(pair_idx)

            if self.training:
                raw_gt_xyz = batch_dict['vectors'][i]
                valid_gt_mask = (raw_gt_xyz[:, 0] > -5.0)
                gt_xyz = raw_gt_xyz[valid_gt_mask]

                raw_gt_edges = batch_dict['edges'][i]
                valid_edge_mask = (raw_gt_edges[:, 0] > -5.0)
                gt_edges = raw_gt_edges[valid_edge_mask].cpu().numpy().astype(int)

                gt_edges_set = set([tuple(sorted(e)) for e in gt_edges])

                if len(gt_xyz) > 0:
                    dist_mat = torch.cdist(k_xyz, gt_xyz)

                    min_dist, gt_assignment = torch.min(dist_mat, dim=-1)

                    dist_thresh = 0.20
                    valid_mask = min_dist < dist_thresh

                    cur_labels = []
                    for r, c in pair_idx:
                        r, c = r.item(), c.item()
                        if valid_mask[r] and valid_mask[c]:
                            gt_r = gt_assignment[r].item()
                            gt_c = gt_assignment[c].item()

                            if tuple(sorted((gt_r, gt_c))) in gt_edges_set:
                                cur_labels.append(1.0)
                            else:
                                cur_labels.append(0.0)
                        else:

                            cur_labels.append(-1.0)
                else:
                    cur_labels = [0.0] * len(pair_idx)

                bin_label_list.append(torch.tensor(cur_labels, dtype=torch.float32, device=k_xyz.device))

            pmid = (k_xyz[pair_idx[:, 0]] + k_xyz[pair_idx[:, 1]]) / 2.0
            dist = torch.cdist(pmid, b_xyz)
            knn_dist, knn_idx = torch.topk(dist, k=3, dim=-1, largest=False)

            weight = 1.0 / (knn_dist + 1e-8)
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)

            knn_fea = b_fea[knn_idx]
            phy_fea = torch.sum(knn_fea * weight.unsqueeze(-1), dim=1)
            phy_fea_list.append(phy_fea)

            pair_idx_list1.append(pair_idx[:, 0] + idx)
            pair_idx_list2.append(pair_idx[:, 1] + idx)
            idx += tmp_idx.shape[0]

        if len(pair_idx_list1) == 0:
            return batch_dict

        if self.training and len(bin_label_list) > 0:
            self.train_dict['label'] = torch.cat(bin_label_list)

        pair_idx1 = torch.cat(pair_idx_list1).long()
        pair_idx2 = torch.cat(pair_idx_list2).long()

        pair_fea1 = point_fea[pair_idx1]
        pair_fea2 = point_fea[pair_idx2]

        edge_fea_sem = self.att_layer(pair_fea1, pair_fea2)
        phy_fea_all = torch.cat(phy_fea_list, dim=0)
        edge_fea_fused = torch.cat([edge_fea_sem, phy_fea_all], dim=-1)

        edge_pred = self.cls_fc(self.drop(self.shared_fc(edge_fea_fused)))

        batch_dict['pair_points'] = torch.cat(pair_idx_list, 0)
        batch_dict['edge_score'] = torch.sigmoid(edge_pred).view(-1)

        if self.training:
            self.train_dict['edge_pred'] = edge_pred
        return batch_dict

    def loss(self, loss_dict, disp_dict):
        pred_cls = self.train_dict['edge_pred']
        label_cls = self.train_dict['label']

        valid_mask = label_cls >= 0
        pred_valid = pred_cls[valid_mask]
        label_valid = label_cls[valid_mask]

        if pred_valid.shape[0] > 0:
            cls_loss = self.get_cls_loss(pred_valid, label_valid, self.loss_weight['cls_weight'])
        else:
            cls_loss = torch.tensor(0.0, device=pred_cls.device)
        loss = cls_loss
        loss_dict.update({
            'edge_loss': loss.item()
        })

        pred_cls_sq = pred_valid.squeeze(-1)
        label_cls_sq = label_valid.squeeze(-1)
        pred_logit = torch.sigmoid(pred_cls_sq)

        pred_mask = (pred_logit >= 0.5)
        label_mask = (label_cls_sq == 1)

        TP = torch.sum(pred_mask & label_mask).item()
        FP = torch.sum(pred_mask & ~label_mask).item()
        FN = torch.sum(~pred_mask & label_mask).item()

        m3_P = TP / (TP + FP + 1e-6)
        m3_R = TP / (TP + FN + 1e-6)

        disp_dict.update({
            'm3_P': round(m3_P, 3),
            'm3_R': round(m3_R, 3)
        })

        return loss, loss_dict, disp_dict

    def get_cls_loss(self, pred, label, weight):
        num_samples = torch.clamp(torch.tensor(pred.shape[0]).float(), min=1.0)

        cls_weights = torch.ones_like(label) / num_samples

        cls_loss_src = self.cls_loss_func(pred.squeeze(-1), label, weights=cls_weights)

        cls_loss = cls_loss_src.sum()
        cls_loss = cls_loss * weight
        return cls_loss

class PairedPointAttention(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.edge_att1 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.edge_att2 = nn.Sequential(
            nn.Linear(input_channel, input_channel),
            nn.BatchNorm1d(input_channel),
            nn.ReLU(),
            nn.Linear(input_channel, input_channel),
            nn.Sigmoid(),
        )
        self.fea_fusion_layer = nn.MaxPool1d(2)

        self.num_output_feature = input_channel

    def forward(self, point_fea1, point_fea2):
        fusion_fea = point_fea1 + point_fea2
        att1 = self.edge_att1(fusion_fea)
        att2 = self.edge_att2(fusion_fea)
        att_fea1 = point_fea1 * att1
        att_fea2 = point_fea2 * att2
        fea = torch.cat([att_fea1.unsqueeze(1), att_fea2.unsqueeze(1)], 1)
        fea = self.fea_fusion_layer(fea.permute(0, 2, 1)).squeeze(-1)
        return fea