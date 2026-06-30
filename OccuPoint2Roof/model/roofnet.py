import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .pointnet2 import PointNet2
from .cluster_refine import ClusterRefineNet
from .edge_pred_net import EdgeAttentionNet
from .completion_net import FCABlock, OccludedCompletionNet
from .model_utils import *

class RoofNet(nn.Module):
    def __init__(self, model_cfg, input_channel=3):
        super().__init__()
        self.use_edge = False
        self.model_cfg = model_cfg

        self.keypoint_det_net = PointNet2(model_cfg.PointNet2, input_channel)
        m1_dim = self.keypoint_det_net.num_output_feature

        self.cluster_refine_net = ClusterRefineNet(model_cfg.ClusterRefineNet, input_channel=m1_dim)
        m2_dim = self.cluster_refine_net.num_output_feature

        self.completion_net = OccludedCompletionNet(feature_dim=m2_dim)

        self.edge_att_net = EdgeAttentionNet(model_cfg.EdgeAttentionNet, input_channel=m2_dim)

        self.align_dim = nn.Sequential(
            nn.Linear(m1_dim, m2_dim),
            nn.BatchNorm1d(m2_dim) if m2_dim == 256 else nn.Identity(),
            nn.ReLU()
        )

    def forward(self, batch_dict):
        device = batch_dict['points'].device
        b = batch_dict['batch_size']

        if self.training and 'visible_corners_gt' in batch_dict:
            batch_dict['vectors_full'] = batch_dict['vectors']
            batch_dict['vectors'] = batch_dict['visible_corners_gt']

        batch_dict = self.keypoint_det_net(batch_dict)

        raw_backbone_fea = batch_dict.get('point_features')
        if raw_backbone_fea is not None:

            b_size, n_pts, c_dim = raw_backbone_fea.shape
            aligned_fea = self.align_dim(raw_backbone_fea.reshape(-1, c_dim))
            batch_dict['backbone_fea'] = aligned_fea.reshape(b_size, n_pts, -1)

        batch_dict['backbone_xyz'] = batch_dict['points']

        batch_dict = self.cluster_refine_net(batch_dict)

        batch_dict['keypoint_xyz'] = batch_dict['keypoint'][:, 1:4]
        batch_dict['m1_keypoint_idx'] = batch_dict['keypoint'][:, 0].clone()
        batch_dict['m1_keypoint_xyz'] = batch_dict['keypoint_xyz'].clone()

        if 'global_feature' not in batch_dict:
            batch_dict['global_feature'] = torch.max(batch_dict['point_features'], dim=1)[0]
        global_fea = self.align_dim(batch_dict['global_feature'])

        max_vis = max([(batch_dict['keypoint'][:, 0] == i).sum().item() for i in range(b)])
        if max_vis == 0: max_vis = 1
        C = batch_dict['keypoint_features'].shape[-1]

        padded_vis_fea = torch.zeros((b, max_vis, C), device=device)
        padded_vis_xyz = torch.ones((b, max_vis, 3), device=device) * -10.0

        for i in range(b):
            mask = (batch_dict['keypoint'][:, 0] == i)
            v_f = batch_dict['keypoint_features'][mask]
            v_x = batch_dict['keypoint_xyz'][mask]
            padded_vis_fea[i, :len(v_f), :] = v_f
            padded_vis_xyz[i, :len(v_x), :] = v_x

        pred_occ_coords, pred_valid_logits = self.completion_net(
            global_fea.detach(),
            padded_vis_fea.detach(),
            padded_vis_xyz.detach()
        )

        batch_dict['pred_occ_coords'] = pred_occ_coords
        batch_dict['pred_valid_logits'] = pred_valid_logits

        m2_clustered_list = []

        for i in range(b):
            probs = torch.sigmoid(pred_valid_logits[i])

            valid_mask = probs > 0.5
            valid_pts = pred_occ_coords[i][valid_mask]
            valid_probs = probs[valid_mask]

            if valid_pts.shape[0] > 1:
                order = torch.argsort(valid_probs, descending=True)
                valid_pts = valid_pts[order]
                valid_probs = valid_probs[order]
                keep = torch.ones(valid_pts.shape[0], dtype=torch.bool, device=device)
                for j in range(valid_pts.shape[0]):
                    if not keep[j]:
                        continue
                    dists_nms = torch.norm(valid_pts[j + 1:] - valid_pts[j], dim=-1)
                    keep[j + 1:] = keep[j + 1:] & (dists_nms >= 0.12)
                valid_pts = valid_pts[keep]

            m2_clustered_list.append(valid_pts)

        batch_dict['m2_keypoint_xyz'] = m2_clustered_list

        mixed_xyz_list = []
        mixed_fea_list = []
        keypoint_idx_list = []
        matches_list = []

        backbone_xyz = batch_dict['backbone_xyz']
        backbone_fea = batch_dict['backbone_fea']

        for i in range(b):
            mask = (batch_dict['keypoint'][:, 0] == i)
            k_x = batch_dict['keypoint_xyz'][mask]
            k_f = batch_dict['keypoint_features'][mask]

            p_occ_for_m3 = m2_clustered_list[i].detach()

            if p_occ_for_m3.shape[0] > 0:
                dist_to_backbone = torch.cdist(p_occ_for_m3, backbone_xyz[i])
                knn_dist, knn_idx = torch.topk(dist_to_backbone, k=3, dim=-1, largest=False)
                weight = 1.0 / (knn_dist + 1e-8)
                weight = weight / torch.sum(weight, dim=-1, keepdim=True)
                p_fea_clustered = torch.sum(backbone_fea[i][knn_idx] * weight.unsqueeze(-1), dim=1)
            else:
                p_fea_clustered = torch.empty((0, backbone_fea.shape[-1]), device=device)

            m_x = torch.cat([k_x.detach(), p_occ_for_m3], dim=0)
            m_f = torch.cat([k_f, p_fea_clustered], dim=0)

            mixed_xyz_list.append(m_x)
            mixed_fea_list.append(m_f)

            k_idx = torch.full((m_x.shape[0], 2), i, device=device)
            keypoint_idx_list.append(k_idx)

            if self.training and 'vectors_full' in batch_dict:
                gt_vec = batch_dict['vectors_full'][i]
                valid_mask = gt_vec[:, 0] > -5.0
                valid_gt = gt_vec[valid_mask]
                if len(valid_gt) == 0:
                    matches_list.append(torch.zeros((m_x.shape[0], 1), device=device).long())
                else:
                    dist = torch.cdist(m_x, valid_gt)
                    min_idx = torch.argmin(dist, dim=1)
                    matches_list.append(min_idx.unsqueeze(-1))

        batch_dict['keypoint_xyz'] = torch.cat(mixed_xyz_list, dim=0)
        batch_dict['keypoint_features'] = torch.cat(mixed_fea_list, dim=0)
        batch_dict['keypoint'] = torch.cat(keypoint_idx_list, dim=0)
        batch_dict['m2_keypoint_xyz'] = m2_clustered_list

        if self.training and len(matches_list) > 0:
            batch_dict['matches'] = torch.cat(matches_list, dim=0)
            batch_dict['vectors'] = batch_dict['vectors_full']

        if self.use_edge or not self.training:
            batch_dict = self.edge_att_net(batch_dict)

        if self.training:
            return self.compute_loss(batch_dict)
        else:
            return batch_dict

    def compute_loss(self, batch_dict):
        loss = 0
        loss_dict = {}
        disp_dict = {}

        m1_loss, loss_dict, disp_dict = self.keypoint_det_net.loss(loss_dict, disp_dict)
        loss += m1_loss

        if hasattr(self.cluster_refine_net, 'loss'):
            c_loss, loss_dict, disp_dict = self.cluster_refine_net.loss(loss_dict, disp_dict)
            loss += c_loss

        m1_prec, m1_rec = self.get_m1_metrics(batch_dict)
        disp_dict['m1_P'] = m1_prec
        disp_dict['m1_R'] = m1_rec

        pred_coords = batch_dict['pred_occ_coords']
        pred_valid_logits = batch_dict['pred_valid_logits']
        gt_coords = batch_dict['occluded_corners_gt']

        raw_m2_loss, m2_acc = self.get_bipartite_loss(pred_coords, pred_valid_logits, gt_coords)

        scaled_m2_loss = raw_m2_loss * 10.0

        loss += scaled_m2_loss

        disp_dict['m2_loss'] = round(scaled_m2_loss.item(), 3)
        disp_dict['m2_acc'] = round(m2_acc, 3)
        m2_prec, m2_rec = self.get_m2_metrics(batch_dict)
        disp_dict['m2_P'] = m2_prec
        disp_dict['m2_R'] = m2_rec

        if self.use_edge:
            m3_loss, loss_dict, disp_dict = self.edge_att_net.loss(loss_dict, disp_dict)
            loss += m3_loss

        return loss, loss_dict, disp_dict

    def get_bipartite_loss(self, pred_coords, pred_valid_logits, gt_coords):
        batch_size = pred_coords.shape[0]
        device = pred_coords.device
        total_acc = 0
        valid_batch_count = 0

        target_labels = torch.zeros((batch_size, pred_coords.shape[1], 1), device=device, dtype=torch.float32)
        batch_loss_coord = []
        batch_loss_chamfer = []

        cost_mats_gpu = []
        valid_gts = []
        max_k = 0

        for i in range(batch_size):
            p = pred_coords[i]
            g = gt_coords[i]
            valid_mask = (g[:, 0] > -5.0)
            valid_g = g[valid_mask]

            valid_gts.append(valid_g)
            k = len(valid_g)
            max_k = max(max_k, k)

            if k == 0:
                cost_mats_gpu.append(None)
                continue

            cost_dist = torch.cdist(p, valid_g, p=2.0)
            cost_mats_gpu.append(cost_dist)

            min_dist_p2g, _ = torch.min(cost_dist, dim=1)

            min_dist_g2p, _ = torch.min(cost_dist, dim=0)

            chamfer_loss = min_dist_p2g.mean() + min_dist_g2p.mean()
            batch_loss_chamfer.append(chamfer_loss)

            min_euc_dists, _ = torch.min(cost_dist, dim=0)
            total_acc += (min_euc_dists < 0.05).float().mean()
            valid_batch_count += 1

        if max_k > 0:
            padded_costs = []
            for c in cost_mats_gpu:
                if c is None:
                    padded_costs.append(torch.zeros((pred_coords.shape[1], max_k), device=device))
                else:
                    pad_size = max_k - c.shape[1]
                    padded_costs.append(torch.nn.functional.pad(c, (0, pad_size), value=1e6))
            all_costs_np = torch.stack(padded_costs).detach().cpu().numpy()
        else:
            all_costs_np = None

        from scipy.optimize import linear_sum_assignment
        for i in range(batch_size):
            valid_g = valid_gts[i]
            p = pred_coords[i]
            k = len(valid_g)

            if k == 0 or all_costs_np is None:
                batch_loss_coord.append(torch.tensor(0.0, device=device))
                batch_loss_chamfer.append(torch.tensor(0.0, device=device))
                continue

            cost_matrix_np = all_costs_np[i, :, :k]
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            row_ind_tensor = torch.tensor(row_ind, device=device, dtype=torch.long)
            col_ind_tensor = torch.tensor(col_ind, device=device, dtype=torch.long)

            matched_dists = torch.norm(p[row_ind_tensor] - valid_g[col_ind_tensor], dim=1)

            valid_matches = matched_dists < 1.0
            valid_row_ind = row_ind_tensor[valid_matches]
            valid_col_ind = col_ind_tensor[valid_matches]

            min_dist_to_gt, _ = torch.min(cost_mats_gpu[i], dim=1)
            target_labels[i, min_dist_to_gt < 0.10, 0] = 1.0

            if len(valid_row_ind) > 0:
                l_coord = torch.nn.functional.smooth_l1_loss(p[valid_row_ind],
                                                             valid_g[valid_col_ind])
                batch_loss_coord.append(l_coord)
            else:
                batch_loss_coord.append(torch.tensor(0.0, device=device))

        p_logits = pred_valid_logits.unsqueeze(-1)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(p_logits, target_labels, reduction='none')
        p_val = torch.sigmoid(p_logits)
        p_t = p_val * target_labels + (1 - p_val) * (1 - target_labels)
        alpha_t = 0.5 * target_labels + 0.5 * (1 - target_labels)
        loss_valid = (alpha_t * (1 - p_t) ** 2.0 * bce_loss).mean()

        loss_coord = torch.mean(torch.stack(batch_loss_coord))
        loss_chamfer = torch.mean(torch.stack(batch_loss_chamfer)) if len(batch_loss_chamfer) > 0 else torch.tensor(0.0,
                                                                                                                    device=device)

        final_loss = 2.0 * loss_valid + 15.0 * loss_coord + 2.0 * loss_chamfer

        final_acc = (total_acc / valid_batch_count).item() if valid_batch_count > 0 else 0.0

        return final_loss, final_acc

    def get_m1_metrics(self, batch_dict, dist_thresh=0.2):
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        if 'm1_keypoint_xyz' not in batch_dict or 'visible_corners_gt' not in batch_dict:
            return 0.0, 0.0

        pred_idx = batch_dict['m1_keypoint_idx'].detach().cpu().numpy().astype(int)
        pred_xyz = batch_dict['m1_keypoint_xyz'].detach().cpu().numpy()
        gt_xyz_all = batch_dict['visible_corners_gt'].detach().cpu().numpy()
        batch_size = batch_dict['batch_size']
        total_tp, total_fp, total_fn = 0, 0, 0

        for b in range(batch_size):
            cur_gt = gt_xyz_all[b]
            cur_gt = cur_gt[cur_gt[:, 0] > -5.0]
            mask = (pred_idx == b)
            cur_pred = pred_xyz[mask]

            if len(cur_pred) == 0 and len(cur_gt) == 0: continue
            elif len(cur_pred) == 0: total_fn += len(cur_gt)
            elif len(cur_gt) == 0: total_fp += len(cur_pred)
            else:
                dist_mat = cdist(cur_pred, cur_gt, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(dist_mat)
                tp = sum(1 for r, c in zip(row_ind, col_ind) if dist_mat[r, c] < dist_thresh)
                total_tp += tp
                total_fp += (len(cur_pred) - tp)
                total_fn += (len(cur_gt) - tp)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return prec, rec

    def get_m2_metrics(self, batch_dict, dist_thresh=0.2):
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        if 'm2_keypoint_xyz' not in batch_dict or 'occluded_corners_gt' not in batch_dict:
            return 0.0, 0.0

        m2_preds = batch_dict['m2_keypoint_xyz']
        gt_coords = batch_dict['occluded_corners_gt'].detach().cpu().numpy()
        batch_size = gt_coords.shape[0]
        total_tp, total_fp, total_fn = 0, 0, 0

        m2_preds_np = [p.detach().cpu().numpy() for p in batch_dict['m2_keypoint_xyz']]

        for b in range(batch_size):
            cur_gt = gt_coords[b]
            cur_gt = cur_gt[cur_gt[:, 0] > -5.0]
            cur_pred = m2_preds_np[b]

            if len(cur_pred) == 0 and len(cur_gt) == 0: continue
            elif len(cur_pred) == 0: total_fn += len(cur_gt)
            elif len(cur_gt) == 0: total_fp += len(cur_pred)
            else:
                dist_mat = cdist(cur_pred, cur_gt, metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(dist_mat)
                tp = sum(1 for r, c in zip(row_ind, col_ind) if dist_mat[r, c] < dist_thresh)
                total_tp += tp
                total_fp += (len(cur_pred) - tp)
                total_fn += (len(cur_gt) - tp)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return prec, rec