"""
Modified Flux attention blocks with mask support
Based on Fluxtapoz implementation
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from comfy.ldm.flux.layers import (
    SingleStreamBlock as OriginalSingleStreamBlock, 
    DoubleStreamBlock as OriginalDoubleStreamBlock
)
from comfy.ldm.modules.attention import optimized_attention


def apply_rope_single(xq: Tensor, freqs_cis: Tensor):
    """Apply rotary position embedding"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, skip_rope: bool = False, k_pe=None, mask=None) -> Tensor:
    """
    Modified attention function with mask support
    
    Key modification:
    - Added mask parameter
    - Passes mask to optimized_attention
    """
    if not skip_rope:
        q_pe = pe
        q = apply_rope_single(q, q_pe)

    if k_pe is None:
        k_pe = pe   
    k = apply_rope_single(k, k_pe)
    
    heads = q.shape[1]
    
    # ✅ MASK SUPPORT ADDED
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)

    return x


class DoubleStreamBlock(OriginalDoubleStreamBlock):
    """
    Modified DoubleStreamBlock with mask function support
    
    Key modification:
    - Checks for mask_fn in transformer_options
    - If found, generates mask and passes to attention
    """
    
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, transformer_options={}):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Post-Q hook (for other extensions)
        post_q_fn = transformer_options.get('patches_replace', {}).get('double', {}).get(('post_q', self.idx), None) 
        if post_q_fn is not None:
            q = post_q_fn(q, transformer_options)

        # ✅ MASK FUNCTION SUPPORT
        mask_fn = transformer_options.get('patches_replace', {}).get('double', {}).get(('mask_fn', self.idx), None) 
        mask = None
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, txt.shape[1])

        # RFEdit support (for other extensions)
        rfedit = transformer_options.get('rfedit', {})
        if rfedit.get('process', None) is not None and rfedit.get('double_layers', {}).get(str(self.idx), False):
            pred = rfedit['pred']
            step = rfedit['step']
            if rfedit['process'] == 'forward':
                rfedit['bank'][step][pred][self.idx] = v.cpu()
            elif rfedit['process'] == 'reverse' and self.idx in rfedit['bank'][step][pred]:
                v = rfedit['bank'][step][pred][self.idx].to(v.device)

        # ✅ ATTENTION WITH MASK
        attn = attention(q, k, v, pe=pe, mask=mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlock(OriginalSingleStreamBlock):
    """
    Modified SingleStreamBlock with mask function support
    
    Key modification:
    - Checks for mask_fn in transformer_options
    - If found, generates mask and passes to attention
    """
    
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, transformer_options={}) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Post-Q hook (for other extensions)
        post_q_fn = transformer_options.get('patches_replace', {}).get('single', {}).get(('post_q', self.idx), None) 
        if post_q_fn is not None:
            q = post_q_fn(q, transformer_options)

        # ✅ MASK FUNCTION SUPPORT
        mask_fn = transformer_options.get('patches_replace', {}).get('single', {}).get(('mask_fn', self.idx), None) 
        mask = None
        if mask_fn is not None:
            mask = mask_fn(q, transformer_options, transformer_options['txt_size'])

        # RFEdit support (for other extensions)
        rfedit = transformer_options.get('rfedit', {})
        if rfedit.get('process', None) is not None and rfedit.get('single_layers', {}).get(str(self.idx), False):
            pred = rfedit['pred']
            step = rfedit['step']
            if rfedit['process'] == 'forward':
                rfedit['bank'][step][pred][self.idx] = v.cpu()
            elif rfedit['process'] == 'reverse' and self.idx in rfedit['bank'][step][pred]:
                v = rfedit['bank'][step][pred][self.idx].to(v.device)

        # ✅ ATTENTION WITH MASK
        attn = attention(q, k, v, pe=pe, mask=mask)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output

        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        
        return x


def inject_blocks(diffusion_model):
    """
    Replace block classes with modified versions
    
    WARNING: This modifies the instance's classes!
    """
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = DoubleStreamBlock
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = SingleStreamBlock
        block.idx = i

    return diffusion_model