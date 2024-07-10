#!python3
import torch
import thunder
import einops

def foo(input_ids, inputs_embeds):
  batch_size, sequence_length, hidden_size = inputs_embeds.shape

  media_features = torch.randn((2,1,1,256,5120), dtype=torch.float16)
  num_images_per_sample = media_features.size(1)
  num_patches = media_features.size(3) * media_features.size(2)

  media_end_id = 32005
  sorted_media_end_positions_mask, media_end_positions_mask_sort_idx = (
      # NOTE: to(torch.long) is needed because PyTorch does not have sort for boolean tensors on CUDA
      (input_ids == media_end_id).to(torch.long).sort(dim=-1, descending=True, stable=True)
  )

  padded_media_indices = torch.where(
    sorted_media_end_positions_mask.to(torch.bool),
    media_end_positions_mask_sort_idx - num_patches + 1,
    sequence_length
  )
  padded_media_indices = padded_media_indices.unsqueeze(-1) + torch.arange(
    num_patches, device=padded_media_indices.device
  ).repeat(*padded_media_indices.shape, 1)
  padded_media_indices = padded_media_indices.reshape(batch_size, -1)
  padded_media_indices = einops.repeat(padded_media_indices, 'b s -> b s h', h=hidden_size)

  second = torch.zeros((batch_size, num_patches, hidden_size), device=inputs_embeds.device)
  # Note: thunder can be made to work by explicitly setting the dtype:
  #   second = torch.zeros((batch_size, num_patches, hidden_size), dtype=torch.float32, device=inputs_embeds.device)
  #print(f"ii dt:shape={inputs_embeds.dtype}:{inputs_embeds.shape}")
  #print(f"2nd dt:shape={second.dtype}:{second.shape}")
  updated_input_embeds = torch.cat(
    (inputs_embeds, second), dim=1
  )
  return updated_input_embeds

at = torch.zeros((2,384), dtype=torch.int64)
bt = torch.randn((2,384, 5120), dtype=torch.float32)
ct = torch.randn((2,1,1,3,224,224), dtype=torch.float32)

foo(at, bt)

thfoo = thunder.jit(foo)
thfoo(at, bt)
