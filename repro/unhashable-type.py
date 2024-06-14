#!python3
import os
import torch
import nemo
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import \
    NevaWordEmbeddingMixin
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters \
    import MultimodalProjectorAdapterConfig
from nemo.collections.common.parts.utils import extend_instance
from nemo.core import adapter_mixins
from transformers import CLIPVisionModel

clip = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float
).cuda()

media_start_id = 32004
media_end_id = 32005
class_token_length = 1
vision_select_layer = -2
use_im_start_end = False
class Derived(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    """Required due to how the mixin works"""
    def __init__(self):
        super().__init__()
    def forward(self, input_ids, **kwargs):
        v = torch.empty((2,384,5120), device='cuda:0', dtype=torch.float)
        return v

mdl = Derived()
extend_instance(mdl, NevaWordEmbeddingMixin)
mdl.init_vision(
    clip,
    media_start_id,
    media_end_id,
    vision_select_layer,
    class_token_length,
    use_im_start_end,
)
#mdl.to(device='cuda', dtype=torch.float)
mdl.add_adapter(
    "mm_projector_adapter",
    cfg=MultimodalProjectorAdapterConfig("linear", in_features=1024, out_features=5120,
                               bias=False)
)
#for param in mdl.parameters():
#    param.to(device='cuda', dtype=torch.float)
#mdl.set_enabled_adapters(name="mm_projector_adapter", enabled=True)

input_ids = torch.empty((2, 384), device='cuda:0', dtype=torch.float)
media = torch.empty((2,1,1,3,224,224), device='cuda:0', dtype=torch.float)
mdl.set_media(media)

# First make sure that the above setup would run without thunder, so we can be
# sure that what we are hitting is a *thunder* bug.
try:
    mdl.forward(input_ids)
    input_ids = torch.empty((2, 384), device='cuda:0', dtype=torch.float)
except:
    print("code failed even without thunder")
    import sys
    sys.exit(1)

import thunder
fwd = thunder.jit(mdl.forward)
x = fwd(input_ids)
