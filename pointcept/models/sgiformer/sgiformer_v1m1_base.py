"""
SGIFormer
Xiaoyang slightly modified from: https://github.com/RayYoh/SGIFormer
1. Better batch-wise operation
2. Compatibility to Pointcept backbones
3. No concept of voxel (voxel fundamentally are grid sampled point)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch_scatter import scatter_mean

from pointcept.models.utils.structure import Point
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.builder import MODELS, build_model

from .loss import SGIFormerLoss
from .nms import mask_matrix_nms


def split_offset(value: torch.Tensor, offset: torch.Tensor) -> list[torch.Tensor]:
    ret = []
    start = 0
    for end in offset:
        end_int = int(end)
        ret.append(value[start:end_int])
        start = end_int
    return ret


def identity_inverse_from_offset(
    offset: torch.Tensor, device: torch.device
) -> torch.Tensor:
    bincount = offset2bincount(offset)
    return torch.cat(
        [torch.arange(int(count), device=device) for count in bincount], dim=0
    )


### Decoder ###


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    @staticmethod
    def shift_scale_points(pred_xyz, src_range, dst_range=None):
        """
        pred_xyz: B x N x 3
        src_range: [[B x 3], [B x 3]] - min and max XYZ coords
        dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
        """
        if dst_range is None:
            dst_range = [
                torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
                torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
            ]

        if pred_xyz.ndim == 4:
            src_range = [x[:, None] for x in src_range]
            dst_range = [x[:, None] for x in dst_range]

        assert src_range[0].shape[0] == pred_xyz.shape[0]
        assert dst_range[0].shape[0] == pred_xyz.shape[0]
        assert src_range[0].shape[-1] == pred_xyz.shape[-1]
        assert src_range[0].shape == src_range[1].shape
        assert dst_range[0].shape == dst_range[1].shape
        assert src_range[0].shape == dst_range[1].shape

        src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
        dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
        prop_xyz = (
            ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
        ) + dst_range[0][:, None, :]
        return prop_xyz

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = self.shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2)  # .permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        if self.normalize:
            xyz = self.shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * torch.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2)  # .permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_masks=None, pe=None, query_pe=None):
        """
        query Tensor (b, n_q, d_model)
        """
        outputs = []
        for i in range(len(source)):
            q_pos = query_pe[i] if query_pe is not None else None
            pos = pe[i] if pe is not None else None
            q = self.with_pos_embed(query[i], q_pos)
            k = self.with_pos_embed(source[i], pos)
            v = source[i]
            attn_mask = attn_masks[i] if attn_masks else None
            output, _ = self.attn(q, k, v, attn_mask=attn_mask)
            output = self.dropout(output) + query[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        outputs = []
        for i in range(len(x)):
            pos = pe[i] if pe is not None else None
            q = k = self.with_pos_embed(x[i], pos)
            output, _ = self.attn(q, k, x[i])
            output = self.dropout(output) + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            output = self.net(x[i])
            output = output + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SGIFormerDecoder(nn.Module):
    def __init__(
        self,
        dec_num_layer=3,
        num_sample_query=200,
        num_learn_query=200,
        num_classes=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        attn_mask=True,
        use_score=False,
        alpha=0.4,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_score = use_score
        self.dec_num_layer = dec_num_layer
        self.num_classes = num_classes
        self.d_model = d_model
        self.attn_mask = attn_mask
        self.alpha = alpha

        self.seg_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, num_classes + 1),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            norm_fn(in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, 3),
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.rep_layer = nn.Sequential(
            nn.Linear(d_model, num_sample_query),
            nn.LayerNorm(num_sample_query),
            nn.ReLU(),
        )
        self.query_learn = nn.Embedding(num_learn_query, d_model)

        self.sp_feat_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.x_mask = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())

        self.sp_pos = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=d_model,
            normalize=True,
        )
        self.feat_query_attn_layers = nn.ModuleList([])
        self.feat_self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(self.dec_num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))

            if i < self.dec_num_layer - 1:
                self.feat_query_attn_layers.append(
                    CrossAttentionLayer(d_model, nhead, dropout)
                )
                self.feat_self_attn_layers.append(
                    SelfAttentionLayer(d_model, nhead, dropout)
                )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes + 1)
        )
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )

    def forward_head(self, query_list, sp_mask_feat_list):
        pred_cls_list = []
        pred_mask_list = []
        attn_mask_list = []
        pred_score_list = [] if self.use_score else None

        for i in range(len(query_list)):
            norm_query_ = self.out_norm(query_list[i])
            sp_mask_feat_ = sp_mask_feat_list[i]
            pred_cls_list.append(self.out_cls(norm_query_))
            if self.use_score:
                pred_score_list.append(self.out_score(norm_query_))
            pred_mask_ = torch.einsum("nd, md->nm", norm_query_, sp_mask_feat_)
            if self.attn_mask:
                attn_mask_ = (pred_mask_.sigmoid() < 0.5).bool()
                attn_mask_[torch.where(attn_mask_.sum(-1) == attn_mask_.shape[-1])] = (
                    False
                )
                attn_mask_ = attn_mask_.detach()
                attn_mask_list.append(attn_mask_)
            pred_mask_list.append(pred_mask_)
        attn_mask_list = attn_mask_list if self.attn_mask else None
        return pred_cls_list, pred_score_list, pred_mask_list, attn_mask_list

    def forward(self, input_dict):
        feat = input_dict["feats"]
        coord = input_dict["coord"]
        offset = input_dict["offset"].int()
        use_origin = "inverse" in input_dict and "origin_offset" in input_dict
        origin_offset = (
            input_dict["origin_offset"].int()
            if use_origin
            else input_dict["offset"].int()
        )
        sp_feat = input_dict["sp_feat"]
        sp = input_dict["sp"]

        seg_logits = self.seg_head(feat)
        bias = self.bias_head(feat)
        bincount = offset2bincount(offset)

        # get query
        score = seg_logits.softmax(dim=-1)[:, :-1]
        inverse = (
            input_dict["inverse"]
            if use_origin
            else identity_inverse_from_offset(offset, coord.device)
        )
        inv_list = split_offset(inverse, origin_offset)
        coord_list = split_offset(coord, offset)
        bias_list = split_offset(bias, offset)
        sp_coord_list = [
            scatter_mean((_coord + _bias)[_inv], _sp, dim=0)
            for _coord, _bias, _inv, _sp in zip(coord_list, bias_list, inv_list, sp)
        ]

        feat_proj = self.feat_proj(feat)

        query_list = []
        score_list = score.split(bincount.tolist())
        feat_proj_list = feat_proj.split(bincount.tolist())

        for i in range(len(offset)):
            score_ = score_list[i]
            max_score_, _ = score_.max(dim=-1)
            _, topk_idx = max_score_.topk(
                int(self.alpha * score_.shape[0]), sorted=False
            )
            top_proj_feat_ = feat_proj_list[i][topk_idx, :]
            rep_ = self.rep_layer(top_proj_feat_)
            act_ = torch.softmax(rep_, dim=0)
            query_ = act_.T @ top_proj_feat_
            query_ = torch.cat((query_, self.query_learn.weight), dim=0)
            query_list.append(query_)

        # get pos
        sp_pos_list = []
        for sp_coord_ in sp_coord_list:
            p_min, p_max = sp_coord_.min(0)[0], sp_coord_.max(0)[0]
            pos_emb = self.sp_pos(
                sp_coord_.unsqueeze(0),
                num_channels=self.d_model,
                input_range=(p_min.unsqueeze(0), p_max.unsqueeze(0)),
            )[0]
            sp_pos_list.append(pos_emb)
        sp_feat_list = [self.sp_feat_proj(feat_) for feat_ in sp_feat]
        sp_mask_feat_list = [self.x_mask(feat_) for feat_ in sp_feat_list]

        # decoding
        aux_pred_list = [self.forward_head(query_list, sp_mask_feat_list)]
        attn_mask_list = aux_pred_list[-1][-1]

        for i in range(self.dec_num_layer):
            source_list = [
                sp_feat_ + sp_pos_
                for sp_feat_, sp_pos_ in zip(sp_feat_list, sp_pos_list)
            ]
            # Xiaoyang Note - TODO: Batch-wised Flash Attention with causal mask
            query_list = self.cross_attn_layers[i](
                source_list, query_list, attn_mask_list
            )
            query_list = self.self_attn_layers[i](query_list)
            query_list = self.ffn_layers[i](query_list)

            if i < self.dec_num_layer - 1:
                sp_feat_list = self.feat_query_attn_layers[i](
                    query_list,
                    sp_feat_list,
                    query_pe=sp_pos_list,
                )
                sp_feat_list = self.feat_self_attn_layers[i](
                    sp_feat_list,
                    sp_pos_list,
                )

            aux_pred_list.append(self.forward_head(query_list, sp_mask_feat_list))
            attn_mask_list = aux_pred_list[-1][-1]
        pred_cls_list, pred_score_list, pred_mask_list, _ = aux_pred_list.pop(-1)

        return {
            "cls_list": pred_cls_list,
            "score_list": pred_score_list,
            "mask_list": pred_mask_list,
            "aux_pred_list": [
                {
                    "cls_list": pred_cls_list_,
                    "score_list": pred_score_list_,
                    "mask_list": pred_mask_list_,
                }
                for pred_cls_list_, pred_score_list_, pred_mask_list_, _ in aux_pred_list
            ],
            "seg_logits": seg_logits,
            "bias": bias,
        }


### SGIFormer ###
@MODELS.register_module("SGIFormer-v1m1")
class SGIFormer(nn.Module):
    def __init__(
        self,
        backbone,
        decoder=None,
        criteria=None,
        topk_insts=200,
        score_thr=0.0,
        npoint_thr=100,
        sp_score_thr=0.55,
        nms=True,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.decoder = SGIFormerDecoder(**decoder)
        self.criteria = SGIFormerLoss(**criteria)

        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr
        self.sp_score_thr = sp_score_thr
        self.nms = nms

        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = instance_ignore_index

        if self.decoder.num_classes != self.semantic_num_classes:
            raise ValueError(
                "decoder.num_classes must match semantic_num_classes, "
                f"got {self.decoder.num_classes} and {self.semantic_num_classes}"
            )
        if self.criteria.num_classes != self.semantic_num_classes:
            raise ValueError(
                "criteria.num_classes must match semantic_num_classes, "
                f"got {self.criteria.num_classes} and {self.semantic_num_classes}"
            )

        ignored = sorted([i for i in self.segment_ignore_index if i >= 0])
        total = self.semantic_num_classes + len(ignored)
        kept = [i for i in range(total) if i not in ignored]
        self.register_buffer(
            "class_map", torch.tensor(kept, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def prepare_target(self, data_dict):
        target = dict()
        # only predict instance classes
        segment = data_dict["segment"].clone()
        segment_ignore_index = torch.tensor(
            self.segment_ignore_index, device=segment.device
        )
        segment[torch.isin(segment, segment_ignore_index)] = (
            self.semantic_ignore_index
        )
        for cls in sorted(self.segment_ignore_index, reverse=True):
            if cls == self.semantic_ignore_index:
                continue
            segment[segment >= cls] -= 1

        coord = data_dict["coord"]
        instance = data_dict["instance"]
        if "instance_centroid" in data_dict:
            instance_centroid = data_dict["instance_centroid"]
        else:
            instance_centroid = torch.ones_like(coord) * self.instance_ignore_index
            coord_list = split_offset(coord, data_dict["offset"].int())
            instance_list = split_offset(instance, data_dict["offset"].int())
            centroid_list = split_offset(instance_centroid, data_dict["offset"].int())
            for coord_, instance_, centroid_ in zip(
                coord_list, instance_list, centroid_list
            ):
                valid = instance_ != self.instance_ignore_index
                if valid.any():
                    centroid_per_inst = scatter_mean(
                        coord_[valid], instance_[valid], dim=0
                    )
                    centroid_[valid] = centroid_per_inst[instance_[valid]]

        target["point_info"] = dict(
            segment=segment,
            coord=coord,
            bias=instance_centroid - coord,
            mask=instance != self.instance_ignore_index,
        )

        target["inst_info"] = []
        use_origin = "inverse" in data_dict and "origin_offset" in data_dict
        use_origin = use_origin and "origin_instance" in data_dict
        use_origin = use_origin and "origin_segment" in data_dict
        pt_offset = (
            data_dict["origin_offset"].int()
            if use_origin
            else data_dict["offset"].int()
        )
        pt_ins = split_offset(
            data_dict["origin_instance"] if use_origin else instance, pt_offset
        )
        pt_sem = split_offset(
            data_dict["origin_segment"] if use_origin else data_dict["segment"],
            pt_offset,
        )
        sp = data_dict["sp"]

        for instance_, segment_, sp_ in zip(pt_ins, pt_sem, sp):
            instance_ = instance_.clone()
            segment_ = segment_.clone()

            valid = torch.ones_like(instance_, dtype=torch.bool)
            for label in self.segment_ignore_index:
                valid[segment_ == label] = False
            instance_[~valid] = self.instance_ignore_index
            if valid.any():
                _, inverse = torch.unique(instance_[valid], return_inverse=True)
                instance_[valid] = inverse

            segment_[torch.isin(segment_, segment_ignore_index)] = (
                self.semantic_ignore_index
            )
            for cls in sorted(self.segment_ignore_index, reverse=True):
                if cls == self.semantic_ignore_index:
                    continue
                segment_[segment_ >= cls] -= 1

            inst_mask = instance_.clone()
            if torch.sum(inst_mask == self.instance_ignore_index) != 0:
                inst_mask[inst_mask == self.instance_ignore_index] = (
                    torch.max(inst_mask) + 1
                )
                inst_mask = torch.nn.functional.one_hot(inst_mask.long())[:, :-1]
            else:
                inst_mask = torch.nn.functional.one_hot(inst_mask.long())

            if inst_mask.shape[1] != 0:
                inst_sp_mask_ = scatter_mean(inst_mask.T.float(), sp_, dim=-1) > 0.5
            else:
                inst_sp_mask_ = inst_mask.new_zeros(
                    (0, int(sp_.max().item()) + 1), dtype=torch.bool
                )

            insts = instance_.unique()
            insts = insts[insts != self.instance_ignore_index]
            inst_cls_ = insts.new_zeros(len(insts))
            for i, inst_id in enumerate(insts):
                inst_cls_[i] = segment_[instance_ == inst_id][0]

            target["inst_info"].append(
                dict(
                    cls=inst_cls_,
                    mask=inst_sp_mask_,
                )
            )
        return target

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.backbone(Point(data_dict)))

        vx_offset = data_dict["offset"].int()
        use_origin = "inverse" in data_dict and "origin_offset" in data_dict
        pt_offset = data_dict["origin_offset"].int() if use_origin else vx_offset
        feats = self.backbone(Point(data_dict))

        data_dict["feats"] = feats

        inverse = (
            data_dict["inverse"]
            if use_origin
            else identity_inverse_from_offset(vx_offset, data_dict["coord"].device)
        )
        inv = split_offset(inverse, pt_offset)
        sp_raw = split_offset(data_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp_raw]

        vx_feat = split_offset(feats, vx_offset)
        vx_coord = split_offset(data_dict["coord"], vx_offset)
        vx_grid_coord = split_offset(data_dict["grid_coord"], vx_offset)

        data_dict["sp_feat"] = [
            scatter_mean(_feat[_inv], _sp, dim=0)
            for _feat, _inv, _sp in zip(vx_feat, inv, sp)
        ]
        data_dict["sp_coord"] = [
            scatter_mean(torch.floor(_coord[_inv] * 50), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_coord, inv, sp)
        ]
        data_dict["sp_grid_coord"] = [
            scatter_mean(_coord[_inv].float(), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_grid_coord, inv, sp)
        ]
        data_dict["sp"] = sp

        # assert isinstance(point, Point)
        # while "pooling_parent" in point.keys():
        #     assert "pooling_inverse" in point.keys()
        #     parent = point.pop("pooling_parent")
        #     inverse = point.pop("pooling_inverse")
        #     parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        #     point = parent
        # # Xiaoyang Note: here we use same technology as PTv3 serialization
        # # to fused batched superpoint feature
        # sp, cluster = torch.unique(
        #     point.batch << 48 | point.superpoint,
        #     return_inverse=True,
        # )
        # point["sp_feat"] = torch_scatter.scatter(
        #     point.feat, cluster, dim=0, reduce="mean"
        # )
        # point["sp_batch"] = torch_scatter.scatter(
        #     point.batch, cluster, dim=0, reduce="max"
        # )
        # point["sp_offset"] = batch2offset(point.sp_batch)
        # point["sp_inverse"] = cluster
        pred = self.decoder(data_dict)
        if "segment" in data_dict.keys() and "instance" in data_dict.keys():
            target = self.prepare_target(data_dict)
            return_dict = self.criteria(pred, target)
        else:
            return_dict = dict()

        if not self.training:
            # assume bs=1 for inference
            assert len(pred["cls_list"]) == 1
            pred_cls = pred["cls_list"][0]
            pred_mask = pred["mask_list"][0]

            pred_score = F.softmax(pred_cls, dim=-1)[:, :-1]

            if pred["score_list"] is not None:
                pred_score *= pred["score_list"][0].sigmoid()
            pred_classes = (
                torch.arange(self.semantic_num_classes, device=pred_score.device)
                .unsqueeze(0)
                .repeat(len(pred_cls), 1)
                .flatten(0, 1)
            )
            pred_score, topk_idx = pred_score.flatten(0, 1).topk(
                self.topk_insts, sorted=False
            )
            pred_classes = pred_classes[topk_idx]

            topk_idx = torch.div(
                topk_idx, self.semantic_num_classes, rounding_mode="floor"
            )
            pred_mask = pred_mask[topk_idx]
            pred_mask_sigmoid = pred_mask.sigmoid()

            mask_scores = (pred_mask_sigmoid * (pred_mask > 0)).sum(1) / (
                (pred_mask > 0).sum(1) + 1e-6
            )
            pred_score = pred_score * mask_scores

            if self.nms:
                pred_score, pred_classes, pred_mask_sigmoid, _ = mask_matrix_nms(
                    pred_mask_sigmoid, pred_classes, pred_score, kernel="linear"
                )

            pred_mask_sigmoid = pred_mask_sigmoid[:, sp[0]]
            pred_mask = pred_mask_sigmoid > self.sp_score_thr

            # score_thr
            score_mask = pred_score > self.score_thr
            pred_score = pred_score[score_mask]
            pred_classes = pred_classes[score_mask]
            pred_mask = pred_mask[score_mask]

            # npoint thr
            npoint_mask = pred_mask.sum(1) > self.npoint_thr
            pred_score = pred_score[npoint_mask]
            pred_classes = pred_classes[npoint_mask]
            pred_mask = pred_mask[npoint_mask]

            pred_mask = pred_mask.cpu().detach().numpy()
            pred_classes = self.class_map[pred_classes].cpu().detach().numpy()

            sort_score, sort_index = pred_score.sort(descending=True)
            sort_index = sort_index.cpu().detach().numpy()
            sort_score = sort_score.cpu().detach().numpy()

            sort_classes = pred_classes[sort_index]
            sorted_mask = pred_mask[sort_index]

            return_dict["pred_scores"] = sort_score
            return_dict["pred_masks"] = sorted_mask
            return_dict["pred_classes"] = sort_classes

        return return_dict
