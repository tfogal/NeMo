#!python3
import functools
import sys
import traceback
import torch
import torch._inductor
import torch._inductor.cudagraph_trees
import torch._inductor.compile_fx

def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  t0 = a + 0.4242
  t1 = t0 * b
  t2 = torch.div(t1, 1.48)
  return t2

def cugraph(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  keywords = {
    "device_index": 0,
    "stack_traces": [],
    "is_backward": False,
    "is_inference": False,
  }
  try: # ensure we only get *our* traceback when it dies
    graph_it = torch._inductor.compile_fx.cudagraphify(
      gm,
      **keywords,
    )
    print(f"args: {args}")
    #print(help(graph_it))
    rv = graph_it(args) # force it to graph now.
    #return rv
    return rv
  except Exception as e:
    traceback.print_exception(e)
    sys.exit(42)

fqn = torch.compile(foo, backend=cugraph)
fqn(
  torch.randn((2), device='cuda'),
  torch.randn((2), device='cuda')
)
