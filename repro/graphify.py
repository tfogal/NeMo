#!python3
import functools
import sys
import traceback
import torch
import torch._inductor
import torch._inductor.cudagraph_trees
import torch._inductor.compile_fx

#def foo(args: list[torch.Tensor]) -> torch.Tensor:
#  a = args[0]
#  b = args[1]
#  t0 = a + 0.4242
#  t1 = t0 * b
#  t2 = torch.div(t1, 1.48)
#  return t2

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
    graph_it = torch._inductor.cudagraph_trees.cudagraphify(
      gm
      inputs=args,
      static_input_idxs=[],
      **keywords,
    )
    #print(help(graph_it))
    rv = graph_it(args) # force it to graph now.
    #return rv
    return rv
  except Exception as e:
    traceback.print_exception(e)
    sys.exit(42)

def cugraph2(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  try: # ensure we only get *our* traceback when it dies
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      o = gm(*args)
    def wrapper(*args, **kwargs):
      g.replay()
      return o

    return wrapper
  except Exception as e:
    traceback.print_exception(e)
    sys.exit(42)
  return None

fqn = torch.compile(foo, backend=cugraph)
inputs = [
  torch.randn((2), device='cuda'),
  torch.randn((2), device='cuda')
]
fqn(*inputs)
