import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention import LocalAttention
from timm.models.layers import DropPath, to_2tuple
from timm.models.layers import trunc_normal_
import math

class HPFS(nn.Module):
    def __init__(self, args, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=6, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', device='cpu', num_head=8,
                 style_dim=-1, emo_dim=-1, source_audio_dim=-1, audio_feat_dim_latent=-1, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.encoder_layers = args.encoder_layers

        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        
        self.source_audio_dim = source_audio_dim
        self.audio_feat = audio_feat
        self.emb_trans_dec = emb_trans_dec

        self.n_seed = n_seed
        self.speaker_dim = args.speaker_dim
        self.cond_mode = cond_mode
        print('USE WAVLM')
        self.audio_feat_dim = audio_feat_dim_latent
        self.WavEncoder = WavEncoder(self.source_audio_dim, self.audio_feat_dim)

        self.multimodal_encoder = MultimodalEncoderLayer(
                    latent_dim=latent_dim,
                    audio_latent_dim=self.audio_feat_dim,
                    ffn_dim=self.ff_size,
                    num_head=self.num_heads,
                    dropout=self.dropout,
                    activation=self.activation,
                    encoder_layers=self.encoder_layers
                )
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.num_head = num_heads

        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.outTransEncoder = nn.ModuleList()
        for i in range(self.num_layers - self.encoder_layers):
            self.outTransEncoder.append(
                        nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
                    )
        print('EMBED STYLE BEGIN TOKEN')
        self.speaker_embedding =   nn.Sequential(
                nn.Embedding(style_dim, self.speaker_dim),
                nn.Linear(self.speaker_dim, self.speaker_dim), 
            )
        self.emotion_f = emo_dim    
        self.emotion_num = args.emo_num      
        self.emotion_embedding =   nn.Sequential(
            nn.Embedding(self.emotion_num, self.emotion_f),
            nn.Linear(self.emotion_f, self.emotion_f) 
        )
        self.style_embed = nn.Linear(self.emotion_f * self.speaker_dim, self.latent_dim)
        self.down_proj_style = nn.Linear(self.latent_dim, self.audio_feat_dim)
        self.down_proj_t = nn.Linear(self.latent_dim, self.audio_feat_dim)

        self.rel_pos_audio = SinusoidalEmbeddings(self.audio_feat_dim // self.num_head)
        self.rel_pos_seq = SinusoidalEmbeddings(self.latent_dim // self.num_head)


        self.final_process = FinalProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.input_process3 = nn.Linear(self.latent_dim*3, self.latent_dim)

        self.motion_repr = str(getattr(args, 'motion_repr', '')).lower()
        default_part_routing = (self.motion_repr == 'smplx' and self.njoints == 265 and self.nfeats == 1)
        self.use_part_routing = bool(getattr(args, 'use_part_routing', default_part_routing))
        self.use_part_loss = bool(getattr(args, 'use_part_loss', self.use_part_routing))
        self.router_reg_lambda = float(getattr(args, 'router_reg_lambda', 0.0))
        self.router_temp = float(getattr(args, 'router_temp', 1.0))
        self.use_global_part_residual = bool(
            getattr(args, 'use_global_part_residual', self.use_part_routing)
        )
        self.part_residual_scale = float(getattr(args, 'part_residual_scale', 1.0))
        self.routing_warmup_steps = max(int(getattr(args, 'routing_warmup_steps', 0)), 0)
        self.part_loss_warmup_steps = max(int(getattr(args, 'part_loss_warmup_steps', 0)), 0)
        self.current_train_step = 0

        self.body_cond_scale = float(getattr(args, 'body_cond_scale', 1.0))
        self.hand_cond_scale = float(getattr(args, 'hand_cond_scale', 1.0))
        self.face_cond_scale = float(getattr(args, 'face_cond_scale', 1.0))

        audio_split_dim = int(getattr(args, 'audio_split_dim', 1133))
        if audio_split_dim <= 0 or audio_split_dim >= self.source_audio_dim:
            self.audio_split_dim = self.source_audio_dim
            self.use_dual_audio_condition = False
        else:
            self.audio_split_dim = audio_split_dim
            self.use_dual_audio_condition = True

        semantic_in_dim = (self.source_audio_dim - self.audio_split_dim) if self.use_dual_audio_condition else self.audio_split_dim
        self.cond_rhythm = nn.Linear(self.audio_split_dim, self.latent_dim)
        self.cond_semantic = nn.Linear(semantic_in_dim, self.latent_dim)
        self.cond_affect = nn.Linear(self.latent_dim, self.latent_dim)

        self.latest_router_reg = torch.tensor(0.0)

        if self.use_part_routing:
            if self.nfeats != 1:
                raise ValueError("Part routing currently supports nfeats=1 only.")

            body_idx, hand_idx, face_idx = self._build_default_part_indices(self.njoints)
            self.register_buffer('body_idx', body_idx, persistent=False)
            self.register_buffer('hand_idx', hand_idx, persistent=False)
            self.register_buffer('face_idx', face_idx, persistent=False)

            self.loss_w_body = float(getattr(args, 'loss_w_body', 1.0))
            self.loss_w_hand = float(getattr(args, 'loss_w_hand', 1.0))
            self.loss_w_face = float(getattr(args, 'loss_w_face', 1.0))

            self.router_body = nn.Linear(self.latent_dim, 3)
            self.router_hand = nn.Linear(self.latent_dim, 3)
            self.router_face = nn.Linear(self.latent_dim, 3)

            self.body_head = nn.Linear(self.latent_dim, int(self.body_idx.numel()))
            self.hand_head = nn.Linear(self.latent_dim, int(self.hand_idx.numel()))
            self.face_head = nn.Linear(self.latent_dim, int(self.face_idx.numel()))

            body_prior = self._parse_prior(
                getattr(args, 'router_prior_body', [0.60, 0.20, 0.20]),
                [0.60, 0.20, 0.20]
            )
            hand_prior = self._parse_prior(
                getattr(args, 'router_prior_hand', [0.20, 0.60, 0.20]),
                [0.20, 0.60, 0.20]
            )
            face_prior = self._parse_prior(
                getattr(args, 'router_prior_face', [0.20, 0.20, 0.60]),
                [0.20, 0.20, 0.60]
            )
            self.register_buffer('body_prior', body_prior, persistent=False)
            self.register_buffer('hand_prior', hand_prior, persistent=False)
            self.register_buffer('face_prior', face_prior, persistent=False)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    @staticmethod
    def _parse_prior(raw_prior, fallback):
        if isinstance(raw_prior, str):
            try:
                raw_prior = [float(i.strip()) for i in raw_prior.split(',')]
            except Exception:
                raw_prior = fallback
        if not isinstance(raw_prior, (list, tuple)) or len(raw_prior) != 3:
            raw_prior = fallback
        prior = torch.tensor(raw_prior, dtype=torch.float32)
        prior = torch.clamp(prior, min=1e-6)
        prior = prior / prior.sum()
        return prior

    @staticmethod
    def _build_default_part_indices(njoints):
        if njoints < 265:
            all_idx = torch.arange(njoints, dtype=torch.long)
            return all_idx, all_idx, all_idx

        body = list(range(0, 66))
        hand = list(range(75, 165))
        face = list(range(66, 75)) + list(range(165, 265))
        body_idx = torch.tensor(body, dtype=torch.long)
        hand_idx = torch.tensor(hand, dtype=torch.long)
        face_idx = torch.tensor(face, dtype=torch.long)
        return body_idx, hand_idx, face_idx

    def _router_kl(self, alpha, prior):
        prior = prior.view(1, 1, 3).to(alpha.device)
        alpha = torch.clamp(alpha, min=1e-6)
        return torch.sum(alpha * (torch.log(alpha) - torch.log(prior)), dim=-1).mean()

    def set_train_step(self, step):
        self.current_train_step = max(int(step), 0)

    def _warmup_scale(self, warmup_steps):
        if warmup_steps <= 0:
            return 1.0
        return min(float(self.current_train_step) / float(warmup_steps), 1.0)

    def _build_condition_tokens(self, y, embed_style):
        # y["audio"]: [bs, nframes, source_audio_dim], where source_audio_dim = audio + text.
        audio_input = y['audio']
        rhythm_input = audio_input[:, :, :self.audio_split_dim]
        semantic_input = audio_input[:, :, self.audio_split_dim:] if self.use_dual_audio_condition else rhythm_input

        c_rhythm = self.cond_rhythm(rhythm_input).permute(1, 0, 2)
        c_semantic = self.cond_semantic(semantic_input).permute(1, 0, 2)
        c_affect = self.cond_affect(embed_style).repeat(c_rhythm.shape[0], 1, 1)
        return c_rhythm, c_semantic, c_affect

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def _mask_cond(self, cond, force_mask=False):
        bs, d, l = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.zeros((bs,), device = cond.device).float().uniform_(0, 1) < self.cond_mask_prob
            cond = torch.where(mask.unsqueeze(1).unsqueeze(2), self.cond_mask_emb.repeat(cond.shape[1],1).unsqueeze(0), cond)
            return cond
        else:
            return cond


    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        """

        bs, njoints, nfeats, nframes = x.shape      # 64, 2052, 1, 150
        emb_t = self.embed_timestep(timesteps)  # (1, bs, latent dim)

        # force_mask = y.get('uncond', False)  # False
        # embed_style = self.mask_cond(self.speaker_embedding(y['style']), force_mask=force_mask)  # (bs, latent dim)
        embed_id = self.speaker_embedding(y['style'])
        embed_emo = self.emotion_embedding(y['emotion'][:,0]).unsqueeze(2)
        
        style_m = torch.einsum('bnh,bhm->bnm', embed_emo, embed_id) # B 12 32
        style_m = style_m.reshape(bs, -1)
        
        force_mask = y.get('uncond', False)  # False
        embed_style = self.mask_cond(self.style_embed(style_m), force_mask=force_mask).unsqueeze(0)  # (1, bs, latent dim)

        # audio
        down_style = self.down_proj_style(embed_style)
        down_t = self.down_proj_t(emb_t)
        # enc_audio = self._mask_cond(self.WavEncoder(y['audio']).permute(1, 0, 2)) # nframes, bs, dim
        enc_audio = self.WavEncoder(y['audio'].permute(1, 0, 2))
        audio_style = torch.cat((down_style, enc_audio, down_t), axis=0)
        audio_style = audio_style.permute(1, 0, 2)  # (bs, len, dim)
        audio_style = audio_style.view(bs, nframes + 2, self.num_head, -1)
        audio_style = audio_style.permute(0, 2, 1, 3)
        audio_style = audio_style.reshape(bs * self.num_head, nframes + 2, -1)
        pos_emb = self.rel_pos_audio(audio_style) 
        audio_style, _ = apply_rotary_pos_emb(audio_style, audio_style, pos_emb)
        audio_style_rpe = audio_style.reshape(bs, self.num_head, nframes + 2, -1)
        audio_style = audio_style_rpe.permute(0, 2, 1, 3)  # [seqlen+2, bs, d]
        audio_style = audio_style.view(bs, nframes + 2, -1)
        audio_style = audio_style.permute(1, 0, 2)

        # audio_seq = self.input_process2(audio_seq)
        # noise gesture
        x = x.reshape(bs, njoints * nfeats, 1, nframes)
        gesture = self.input_process(x)  # [seqlen, bs, d]
        xseq = torch.cat((embed_style, gesture, emb_t), axis=0) # (nframes + 2, bs, dim)
        xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
        xseq = xseq.view(bs, nframes + 2, self.num_head, -1)
        xseq = xseq.permute(0, 2, 1, 3)
        xseq = xseq.reshape(bs * self.num_head, nframes + 2, -1)
        pos_emb = self.rel_pos_seq(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 2, -1)
        xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+2, bs, d]
        xseq = xseq.view(bs, nframes + 2, -1)
        xseq = xseq.permute(1, 0, 2)

        # fusion
        fusion_seq, style_b, time_b, xseq_out, audio_out = self.multimodal_encoder(xseq, audio_style)

        # fusion_seq = fusion_seq[1:-1]
        fusion_seq = torch.cat((xseq_out[1:-1], audio_out[1:-1], fusion_seq[1:-1]), axis=2)
        fusion_seq = self.input_process3(fusion_seq)
        fusion_seq = torch.cat((style_b + time_b, fusion_seq), axis=0)
        out_fusion_seq = fusion_seq
        for module in self.outTransEncoder:
            out_fusion_seq = module(fusion_seq)
            fusion_seq = out_fusion_seq

        style_output = out_fusion_seq[3:]
        if not self.use_part_routing:
            self.latest_router_reg = style_output.new_zeros(())
            output = self.final_process(style_output)  # [bs, njoints, nfeats, nframes]
            return output

        global_output = None
        if self.use_global_part_residual:
            global_output = self.final_process(style_output)

        routing_scale = self._warmup_scale(self.routing_warmup_steps)

        c_rhythm, c_semantic, c_affect = self._build_condition_tokens(y, embed_style.squeeze(0))
        cond_stack = torch.stack((c_rhythm, c_semantic, c_affect), dim=2)  # [T, B, 3, D]

        alpha_body = F.softmax(self.router_body(style_output) / self.router_temp, dim=-1)
        alpha_hand = F.softmax(self.router_hand(style_output) / self.router_temp, dim=-1)
        alpha_face = F.softmax(self.router_face(style_output) / self.router_temp, dim=-1)

        body_cond = torch.sum(alpha_body.unsqueeze(-1) * cond_stack, dim=2)
        hand_cond = torch.sum(alpha_hand.unsqueeze(-1) * cond_stack, dim=2)
        face_cond = torch.sum(alpha_face.unsqueeze(-1) * cond_stack, dim=2)

        body_feat = style_output + (self.body_cond_scale * routing_scale) * body_cond
        hand_feat = style_output + (self.hand_cond_scale * routing_scale) * hand_cond
        face_feat = style_output + (self.face_cond_scale * routing_scale) * face_cond

        body_out = self.body_head(body_feat)
        hand_out = self.hand_head(hand_feat)
        face_out = self.face_head(face_feat)

        merged = style_output.new_zeros((style_output.shape[0], style_output.shape[1], self.njoints))
        merged[:, :, self.body_idx] = body_out
        merged[:, :, self.hand_idx] = hand_out
        merged[:, :, self.face_idx] = face_out

        if self.router_reg_lambda > 0:
            self.latest_router_reg = (
                self._router_kl(alpha_body, self.body_prior) +
                self._router_kl(alpha_hand, self.hand_prior) +
                self._router_kl(alpha_face, self.face_prior)
            ) / 3.0
        else:
            self.latest_router_reg = style_output.new_zeros(())

        part_output = merged.permute(1, 2, 0).unsqueeze(2)
        if self.use_global_part_residual and global_output is not None:
            part_scale = self.part_residual_scale
            if self.routing_warmup_steps > 0:
                part_scale = part_scale * routing_scale
            output = global_output + part_scale * part_output
        else:
            output = part_output

        return output

class MultimodalEncoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 audio_latent_dim=512,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 activation="gelu",
                 encoder_layers=2):
        super().__init__()
        audioTransEncoderLayer = nn.TransformerEncoderLayer(d_model=audio_latent_dim,
                                                            nhead=num_head,
                                                            dim_feedforward=ffn_dim,
                                                            dropout=dropout,
                                                            activation=activation)

        self.audioTransEncoder = nn.TransformerEncoder(audioTransEncoderLayer,
                                                        num_layers=encoder_layers)
        
        xseqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                            nhead=num_head,
                                                            dim_feedforward=ffn_dim,
                                                            dropout=dropout,
                                                            activation=activation)

        self.xseqTransEncoder = nn.TransformerEncoder(xseqTransEncoderLayer,
                                                        num_layers=encoder_layers)
        self.crossattn = TemporalCrossAttention(latent_dim=latent_dim, condition_latent_dim=latent_dim, num_head=num_head, dropout=dropout)
        self.ffn = FFN(latent_dim, ffn_dim, dropout)
        self.input_process2 = nn.Linear(audio_latent_dim, latent_dim)

    def forward(self, x, audio):

        audio_seq = self.audioTransEncoder(audio)
        audio_seq = self.input_process2(audio_seq)
        xseq = self.xseqTransEncoder(x) # [seqlen+2, bs, d]

        # specific token
        style_1, time_1, style_2, time_2 = audio_seq[0].unsqueeze(0) , audio_seq[-1].unsqueeze(0) , xseq[0].unsqueeze(0) , xseq[-1].unsqueeze(0) 
        style_spec = torch.cat((style_1, style_2), axis=0)
        time_spec = torch.cat((time_1, time_2), axis=0)
        # audio_seq = audio_seq[1:-1] 
        # xseq = xseq[1:-1]
        xseq_out = xseq
        audio_out = audio_seq
        xseq = xseq.permute(1, 0, 2)
        audio_seq = audio_seq.permute(1, 0, 2)
        fusion_seq = self.crossattn(xseq, audio_seq)
        fusion_seq = self.ffn(fusion_seq)
        fusion_seq = fusion_seq.permute(1, 0, 2) # [seqlen+2, bs, d]

        # share token
        style_share, time_share = fusion_seq[0].unsqueeze(0) , fusion_seq[-1].unsqueeze(0)

        style_b = torch.cat((style_spec, style_share), axis=0)
        time_b = torch.cat((time_spec, time_share), axis=0)

        return fusion_seq, style_b, time_b, xseq_out, audio_out

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError

class FinalProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, condition_latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.condition_norm = nn.LayerNorm(condition_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(condition_latent_dim, latent_dim)
        self.value = nn.Linear(condition_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, c):
        """
        x: B, T, D
        c: B, N, L
        """
        B, T, D = x.shape
        N = c.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.condition_norm(c)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        # style_c = query[:,0].unsqueeze(1)
        # style = style_c.repeat(1,T-1,1,1)
        # query = query[:,1:]

        key = key.view(B, N, H, -1)

        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        # style_w = torch.einsum('bnhd,bmhd->bnmh', style, key)
        # style_w = F.softmax(style_w, dim=2)

        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.condition_norm(c)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        # style_c = style_c.reshape(B, 1, D)
        # y = torch.cat((style_c, y), axis=1)
        return y

class WavEncoder(nn.Module):
    def __init__(self, source_dim, audio_feat_dim):
        super().__init__()
        self.audio_feature_map = nn.Linear(source_dim, audio_feat_dim)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep
