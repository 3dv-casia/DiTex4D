import os
import json
from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
from PIL import Image
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


def ss_noise_share_inference(shape, alpha, device=None, dtype=None):
    T, C, D, H, W = shape
    
    denom = (1.0 + alpha ** 2) ** 0.5
    scale_shared = alpha / denom
    scale_ind = 1.0 / denom

    shared = torch.randn(1, C, D, H, W, device=device, dtype=dtype) * scale_shared

    ind = torch.randn(T, C, D, H, W, device=device, dtype=dtype) * scale_ind

    noise = shared + ind
    
    return noise

def slat_noise_share_inference(coords, feats_shape, num_frames, alpha):
    """
    SparseTensor Time-Correlated Noise (Base + Residual)
    
    Args:
        coords: Tensor [N, 4] -> (Global_Frame_Idx, X, Y, Z)
        feats_shape: Tuple/List -> (N, C), e.g., (Number_of_points, Channels)
        num_frames: int
        alpha: float or Tensor
        
    Returns:
        final_noise: Tensor [N, C]
    """
    device = coords.device
    dtype = torch.float32 
    
    point_video_ids = coords[:, 0] // num_frames

    alpha_t = torch.tensor(float(alpha), device=device, dtype=dtype)

    denom = torch.sqrt(1.0 + alpha_t * alpha_t)
    scale_shared = (alpha_t / denom)
    scale_ind = (1.0 / denom)

    ind = torch.randn(feats_shape, device=device, dtype=dtype) * scale_ind

    spatial_keys = torch.cat([point_video_ids.unsqueeze(1), coords[:, 1:]], dim=1)
    unique_keys, inverse_indices = torch.unique(spatial_keys, dim=0, return_inverse=True)

    num_channels = feats_shape[1]
    
    unique_noise = torch.randn(unique_keys.shape[0], num_channels, device=device, dtype=dtype)
    shared = unique_noise[inverse_indices] * scale_shared
    final_noise = shared + ind
    
    return final_noise


class TrellisStaticTextTo4DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisStaticTextTo4DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisStaticTextTo4DPipeline, TrellisStaticTextTo4DPipeline).from_pretrained(path)
        new_pipeline = TrellisStaticTextTo4DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_cond(self, prompt: List[str], num_samples: int = 1) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text([prompt])
        neg_cond = self.text_cond_model['null_cond']
        cond = cond.unsqueeze(1).repeat(1, num_samples, 1, 1).flatten(0, 1)
        neg_cond = neg_cond.unsqueeze(1).repeat(1, num_samples, 1, 1).flatten(0, 1)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        autoregression: bool = False,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        # noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        noise_shape = (num_samples, flow_model.in_channels, reso, reso, reso)
        noise = ss_noise_share_inference(noise_shape, alpha=1, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        if autoregression:
            return coords, z_s
        return coords, None

    def decode_slat(
            self,
            slat: sp.SparseTensor,
            num_frames: int,
            formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        ) -> dict:
            """
            Decode the structured latent.

            Args:
                slat (sp.SparseTensor): The structured latent.
                formats (List[str]): The formats to decode the structured latent to.

            Returns:
                dict: The decoded structured latent.
            """
            ret = {}
            if 'mesh' in formats:
                raise NotImplementedError("Mesh decoding is not implemented yet.")
                ret['mesh'] = self.models['slat_decoder_mesh'](slat)
            if 'gaussian' in formats:
                ret['gaussian'] = self.models['slat_decoder_gs'](slat, num_frames)
            if 'radiance_field' in formats:
                raise NotImplementedError("Radiance field decoding is not implemented yet.")
                ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
            return ret

    def get_merge_coords(self, coords, static):
        ff = static.coords[static.coords[:, 0] == 0]
        ori = coords[coords[:, 0] != 0]
        merge_coords = torch.cat([ff, ori], dim=0)
        return merge_coords
    
    def get_slat_mask(self, x_0, num_frames):
        global_batch_idx = x_0.coords[:, 0]
        is_first_frame = (global_batch_idx % num_frames) == 0
        mask_feat = is_first_frame.float().view(-1, 1)
        if x_0.feats.device != mask_feat.device:
            mask_feat = mask_feat.to(x_0.feats.device)
        return x_0.replace(mask_feat)
    
    def get_slat_static(self, static, x_0, num_frames):
        new_feats = torch.randn_like(x_0.feats)
        x_coords = x_0.coords
        s_coords = static.coords
        s_feats = static.feats
        def compute_hash(coords):
            c = coords.long()
            return (c[:, 0] << 45) | (c[:, 1] << 30) | (c[:, 2] << 15) | c[:, 3]
        x_keys = compute_hash(x_coords)
        s_keys = compute_hash(s_coords)

        is_start_frame_x = (x_coords[:, 0] % num_frames) == 0
        is_start_frame_s = (s_coords[:, 0] % num_frames) == 0
        x_start_keys = x_keys[is_start_frame_x]
        s_start_keys = s_keys[is_start_frame_s]
        if x_start_keys.shape[0] != s_start_keys.shape[0]:
            raise ValueError(f"Mismatch in first frame point count! x_0: {x_start_keys.shape[0]}, static: {s_start_keys.shape[0]}")
        
        s_keys_sorted, argsort_idx = torch.sort(s_keys)
        s_feats_sorted = s_feats[argsort_idx]
        idx_in_s = torch.searchsorted(s_keys_sorted, x_keys)
        idx_in_s = torch.clamp(idx_in_s, max=len(s_keys_sorted) - 1)
        is_match = (s_keys_sorted[idx_in_s] == x_keys)
        if is_match.any():
            new_feats[is_match] = s_feats_sorted[idx_in_s[is_match]]
        return x_0.replace(new_feats)
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        merge_coords = self.get_merge_coords(coords, sampler_params['static'])
        noise_shape = (merge_coords.shape[0], flow_model.in_channels)
        # feats=slat_noise_share_inference(coords, noise_shape, num_frames=sampler_params["num_frames"], alpha=1).to(self.device)
        noise = sp.SparseTensor(
            feats=slat_noise_share_inference(merge_coords, noise_shape, num_frames=sampler_params["num_frames"], alpha=1).to(self.device),
            coords=merge_coords,
        )
        sampler_params["static"] = self.get_slat_static(sampler_params['static'], noise, sampler_params["num_frames"])
        sampler_params["mask"] = self.get_slat_mask(noise, sampler_params["num_frames"])
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def decompose_coords(self, coords, num_samples):
        seq_len = torch.bincount(coords[:, 0], minlength=num_samples)
        offset = torch.cumsum(seq_len, dim=0)
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(num_samples)]
        coords = coords[:, 1:]
        coords = [coords[layout[i]] for i in range(num_samples)]
        return coords


    @torch.no_grad()
    def run(
        self,
        prompt: str,
        static_ss: torch.Tensor,
        static_slat: torch.Tensor,
        clip: torch.Tensor,
        num_frames: int = 16,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['gaussian'],
        coords: Optional[torch.Tensor] = None,
        autoregression: bool = False,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """

        mask = torch.ones_like(static_ss[:1])
        mask = mask.unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1)
        mask[:, 1:] = 0
        cond = self.get_cond(prompt, num_frames)
        torch.manual_seed(seed)
        if coords is None:
            sparse_structure_sampler_params["num_frames"] = num_frames
            sparse_structure_sampler_params["static"] = static_ss.unsqueeze(0).unsqueeze(1).repeat(1, num_frames, 1, 1, 1, 1).flatten(0, 1)
            sparse_structure_sampler_params["mask"] = mask.flatten(0, 1)
            sparse_structure_sampler_params["clip_cond"] = clip.unsqueeze(0).unsqueeze(1).repeat(1, num_frames, 1, 1).flatten(0, 1)
            coords, ss_latent = self.sample_sparse_structure(cond, num_frames, sparse_structure_sampler_params, autoregression=autoregression)

        slat_sampler_params["num_frames"] = num_frames
        slat_sampler_params["static"] = static_slat 
        slat_sampler_params["clip_cond"] = clip.unsqueeze(0).unsqueeze(1).repeat(1, num_frames, 1, 1).flatten(0, 1)
        slat = self.sample_slat(cond, coords, slat_sampler_params)

        samples = self.decode_slat(slat, num_frames, formats)
        coords = self.decompose_coords(coords, num_frames)

        if autoregression:
            return coords, samples, ss_latent[-1], slat
        return coords, samples
    

    @torch.no_grad()
    def run_ss(
        self,
        prompt: str,
        num_frames: int = 16,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        return_images: bool = False,
    ) -> list[torch.Tensor]:
        
        cond = self.get_cond(prompt, num_frames)
        torch.manual_seed(seed)
        sparse_structure_sampler_params["num_frames"] = num_frames
        coords = self.sample_sparse_structure(cond, num_frames, sparse_structure_sampler_params)

        coords = self.decompose_coords(coords, num_frames)
        return coords