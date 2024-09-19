# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import functools
import os
import typing
import warnings

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.lightning.base import teardown
import thunder
import thunder.dynamo
import thunder.transforms
import thunder.transforms.cudagraph
import torch
import torch._dynamo
import torch._inductor.compile_fx
from torch.profiler import profile, ProfilerActivity
from torch.nn.modules.module import _addindent
from thunder.examine import examine
# See https://github.com/pytorch/pytorch/issues/104674
torch._dynamo.config.optimize_ddp = False
# Workaround an autograd issue.
torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)

# These are meant to match the 'target' property of torch.fx.Nodes. For the
# most part the targets are functions and methods, but they are occasionally
# torch types.
unsupported: set[typing.Any] = set([
  # See https://github.com/Lightning-AI/lightning-thunder/issues/824
  torch.ops._enter_autocast,
  torch.ops._exit_autocast,
  # This type is used for e.g. all_reduce in the torch graph. We don't support
  # capturing those in thunder, and probably won't for some time.
  torch._ops.OpOverloadPacket,
])

def thunder_supported(gm: torch.fx.GraphModule) -> bool:
  """Returns true if thunder supports the graph."""
  for node in gm.graph.nodes:
    if node.op == "call_function" and (node.target in unsupported \
       or type(node.target) in unsupported):
      return False
  return True

#import thunder._experimental
#thunder_supported = thunder._experimental.thunder_supported

def arg_like_tensor_integer(arg: torch.Tensor, f: typing.TextIO):
  """Creates a new argument like the given tensor, which must be an integer
  type. This is separated out because randn() does not work for integer
  types."""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  assert arg.dtype in itypes
  minmax: tuple[torch.Tensor,torch.Tensor] = torch.aminmax(arg)
  # Sometimes the tensor can just be all the same value, in which case
  # randint() will complain that there's no "range" to the low/high params.
  # So we use a completely different method for tensor generation, then.
  if minmax[0].cpu().item() == minmax[1].cpu().item():
    meta = f"data={minmax[0].cpu().item()}, dtype={arg.dtype},"
    meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
    print(f"  torch.tensor({meta}).broadcast_to({arg.shape}),", file=f)
    return
  meta = f"low={minmax[0].cpu().item()}, high={minmax[1].cpu().item()},"
  meta = f"{meta} size={arg.shape},"
  meta = f"{meta} dtype={arg.dtype}, layout={arg.layout},"
  meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
  print(f"  torch.randint({meta}),", file=f)

def arg_like_tensor(arg: torch.Tensor, f: typing.TextIO):
  """Creates a new argument like the given tensor"""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  assert arg.dtype not in itypes
  minmax: tuple[torch.Tensor,torch.Tensor] = torch.aminmax(arg)
  meta = f"size={arg.shape},"
  meta = f"{meta} dtype={arg.dtype}, layout={arg.layout},"
  meta = f"{meta} device=\"{arg.device}\", requires_grad={arg.requires_grad}"
  print(f"  torch.randn({meta}),", file=f)

def arg_like(arg: typing.Any, f: typing.TextIO):
  """Creates a new argument that is similar to the given arg."""
  itypes = (torch.int, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5,
            torch.uint6, torch.uint7, torch.uint8, torch.uint16, torch.uint32)
  if isinstance(arg, torch.Tensor) and arg.dtype in itypes:
      arg_like_tensor_integer(arg, f)
  elif isinstance(arg, torch.Tensor):
      arg_like_tensor(arg, f)
  else:
      # Assume it's a literal that we can just print directly.
      print(f"  {arg},", file=f)

def _readable(
    module,
    module_name,
    print_output=False,
    include_stride=True,
    include_device=True,
    colored=False,
):
    """Modified from torch. This is basically print_readable but it sets
    verbose=False (torch hardcodes it to True)."""
    graph = module.graph
    assert graph is not None and isinstance(graph, torch.fx.Graph), "print_readable must be used on a module with a graph"

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=False,
        include_stride=include_stride,
        include_device=include_device,
        colored=colored,
    )
    module_code = verbose_python_code.src
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 4)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if hasattr(submodule, "graph"):
            submodule_code_list.append(
                _readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 4)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output

def reproducer(gm: torch.fx.GraphModule, args, num: int):
  assert num >= 0
  # Ideally we'd use print_readable, but we want verbose=False and there's no
  # way to set that with print_readable.
  readable = _readable(gm, "DynamoModule", print_output=False)
  #readable = gm.graph.python_code(
  #  root_module="self",
  #  verbose=False,
  #  include_stride=True,
  #  include_device=True,
  #  colored=False
  #).src.lstrip("\n")
  # Fixes for python_code not producing valid code.
  readable = readable.replace("device = device(type",
                              "device=torch.device(type")
  readable = readable.replace("einops_einops_rearrange",
                              "einops.einops.rearrange")
  readable = readable.replace("einops_einops_reduce",
                              "einops.einops.reduce")
  readable = readable.replace("einops_einops_repeat",
                              "einops.einops.repeat")
  with open(f"/tmp/g{num}.py", "w") as f:
    print("import os\n", file=f)
    print("import einops", file=f)
    print("import torch", file=f)
    print("import thunder\n", file=f)
    print("import thunder.transforms.cudagraph\n", file=f)
    print("_execs = [", file=f)
    print("  thunder.extend.get_executor(\"nvfuser\"),", file=f)
    print("  thunder.extend.get_executor(\"sdpa\"),", file=f)
    print("  thunder.extend.get_executor(\"cudnn\"),", file=f)
    print("]\n", file=f)
    print(f"def test_g{num}():", file=f)
    print("   ", _addindent(readable, 4), file=f)
    print("    inputs = [", file=f)
    for a in args:
      print("    ", end="", file=f)
      arg_like(a, f)
    print("    ]", file=f)
    print("    backend = os.getenv(\"BACKEND\")", file=f)
    print("    if backend == None or backend == \"thunder\":", file=f)
    print("        fqn = thunder.jit(DynamoModule())", file=f)
    print("    elif backend == \"thunder-no-t.c\":", file=f)
    print("        fqn = thunder.jit(DynamoModule(), executors=_execs)", file=f)
    print("    elif backend == \"t.c\":", file=f)
    print("        fqn = torch.compile(DynamoModule())", file=f)
    print("    elif backend == \"dynamo-eager\":", file=f)
    print("        fqn = torch.compile(DynamoModule(), backend=\"eager\")", file=f)
    print("    elif backend == \"thunder-cugraph\":", file=f)
    print("        xform = thunder.transforms.cudagraph.CUDAGraphTransform()",
          file=f)
    print("        fqn = thunder.jit(DynamoModule(), transform=[xform])",
          file=f)
    print(f"    post_graph = os.getenv(\"POST_GRAPH\", \"0\")", file=f)
    print(f"    if int(post_graph) > 0:", file=f)
    print(f"        fqn = torch.cuda.make_graphed_callables(", file=f)
    print(f"          fqn, inputs,", file=f)
    print(f"          num_warmup_iters=1, allow_unused_input=True", file=f)
    print(f"        )", file=f)
    print(f"    torch.cuda.nvtx.range_push(\"g{num} compilation\")", file=f)
    print(f"    fqn(*inputs) # run once to force compilation", file=f)
    print(f"    torch.cuda.nvtx.range_pop()", file=f)
    print(f"    torch.cuda.nvtx.range_push(\"g{num} warmups\")", file=f)
    print(f"    for i in range(3): # warmup runs", file=f)
    print(f"        fqn(*inputs)", file=f)
    print(f"    torch.cuda.synchronize()", file=f)
    print(f"    torch.cuda.nvtx.range_pop()", file=f)
    print(f"    torch.cuda.nvtx.range_push(\"g{num}\")", file=f)
    print(f"    fqn(*inputs)", file=f)
    print(f"    torch.cuda.synchronize()", file=f)
    print(f"    torch.cuda.nvtx.range_pop()", file=f)
    print(f"\ntest_g{num}()", file=f)

num_graphs = 0
thunder_graphs = 0
empty_graphs = 0
try:
  os.unlink("/tmp/graphs.txt") # ensure it's new per-run
except NotImplementedError: pass
except FileNotFoundError: pass

def logging_thunder_backend(gm: torch.fx.GraphModule, args):
  gm.real_recompile()
  global num_graphs
  reproducer(gm, args, num_graphs)
  with open("/tmp/graphs.txt", "a") as f:
    print(f"graph {num_graphs}: {len(args)} args", file=f)
    for a in args:
      print("\t", end="", file=f)
      arg_like(a, f)
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      fqn = thunder.jit(gm)
      fqn(*args)
      fname = f"/tmp/graph-{thunder_graphs}.log.txt"
      with open(fname, "w") as f:
        src: str = repr(thunder.last_traces(fqn)[-1])
        src: str = "".join([src, "\n"])
        f.write(src)
      thunder_graphs = thunder_graphs + 1
      return fqn
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
      fname = f"/tmp/unsupported-graph-{num_graphs}.log.txt"
      with open(fname, "w") as f:
        f.write(f"graph {num_graphs}:\n")
        print(gm.graph, file=f)
  except Exception as e:
    print(f"broke: {e}")
    print(f"because of input: {gm.graph}")
    return gm
  return gm

import thunder.extend
#_execs: tuple[thunder.extend.Executor] = thunder.extend.get_always_executors() + (
_execs: tuple[thunder.extend.Executor] = (
  thunder.extend.get_executor("python"),
  thunder.extend.get_executor("torch"),
  thunder.extend.get_executor("nvfuser"),
  thunder.extend.get_executor("torchcompile_cat"),
  thunder.extend.get_executor("sdpa"),
  thunder.extend.get_executor("cudnn"),
)

def cugraph(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  try:
    import torch._inductor
    import torch._inductor.cudagraph_trees
    keywords = {
      "device_index": 0,
      "stack_traces": [],
      "is_backward": False,
      "is_inference": False,
    }
    #graph_it = torch._inductor.cudagraph_trees.cudagraphify_impl(
    graph_it = torch._inductor.compile_fx.cudagraphify(
      gm,
      **keywords,
    )
    rv = graph_it(args) # force it to graph now.
    return rv
  except Exception as e:
    import traceback
    traceback.print_exception(e)
    print(f"exception:\n{e}")
    import sys
    sys.exit(42)
  return rv

  g = torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
    outs = gm(args)

  def g_replay(args: list[torch.Tensor], **kwargs):
    torch.cuda.nvtx.range_push("replay")
    g.replay()
    torch.cuda.nvtx.range_pop()
    return outs
  return g_replay

def thunder_backend(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      torch.cuda.nvtx.range_push("forced compilation")
      fqn = thunder.jit(gm)
      fqn(*args) # force compilation to happen now
      torch.cuda.nvtx.range_pop()
      thunder_graphs = thunder_graphs + 1
      return fqn
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    print(f"broke: {e}")
    print(f"because of input: {gm.graph}")
    return gm
  return gm

def thunder_graph_backend5(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      #with torch.profiler.record_function("thunder's jit"):
      torch.cuda.nvtx.range_push("thunder.jit'd function")
      fqn = thunder.jit(gm)
      #fqn = thunder.jit(gm, executors=_execs)
      #g = torch.cuda.CUDAGraph()
      #with torch.cuda.graph(g):
      fqn = torch._inductor.compile_fx.cudagraphify(fqn,
        device_index=0,
        stack_traces=[],
        is_backward=False,
        is_inference=False,
      )
      fqn(*args)
      torch.cuda.nvtx.range_pop()
      thunder_graphs = thunder_graphs + 1
      return fqn
      #def graph_replay(args: list[torch.Tensor], **kwargs = {}):
      #  torch.cuda.nvtx.range_push("replay graph")
      #  g.replay()
      #  torch.cuda.nvtx.range_pop()
      #return graph_replay
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    print(f"broke: {e}")
    print(f"because of input: {gm.graph}")
    return gm
  return gm

def graph_is_empty(warns) -> bool:
  return any("Graph is empty" in str(w.message) for w in warns)

def use_cuda_graph(graph_id: int) -> bool:
  """whether or not the graph should use cuda graphs"""
  #return graph_id < 35 or graph_id > 45
  return graph_id == 37

def thunder_graph_backend(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      #with torch.profiler.record_function("thunder's jit"):
      fqn = thunder.jit(gm)
      g = torch.cuda.CUDAGraph()
      fqn(*args) # run once outside the graph to make the allocator happy?
      if not use_cuda_graph(copy.deepcopy(thunder_graphs)):
        thunder_graphs = thunder_graphs + 1
        return fqn

      cur_graph = copy.deepcopy(thunder_graphs)
      with warnings.catch_warnings(record=True) as warns:
        torch.cuda.nvtx.range_push(f"TFX capture graph {cur_graph}")
        print(f"TJF capture graph {cur_graph} ..")
        with torch.cuda.graph(g):
          outputs = fqn(*args)
        torch.cuda.nvtx.range_pop()
      # don't bother using replay for an empty graph.
      if graph_is_empty(warns):
        global empty_graphs
        empty_graphs = empty_graphs + 1
        torch.cuda.nvtx.mark(f"TFX graph {cur_graph} empty")
        return fqn

      thunder_graphs = thunder_graphs + 1
      def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(f"TFX graph {cur_graph} run")
        g.replay()
        torch.cuda.nvtx.range_pop()
        return outputs
      return wrapper
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    print(f"broke: {e}")
    print(f"because of input: {gm.graph}")
    return gm
  return gm

def thunder_graph_backend2(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      fqn = thunder.jit(gm)
      torch.cuda.nvtx.range_push(f"ThunderFX {thunder_graphs}")
      fqn(*args) # run once outside the graph to make the allocator happy?
      torch.cuda.nvtx.range_pop()

      if not use_cuda_graph(copy.deepcopy(thunder_graphs)):
        torch.cuda.nvtx.mark(f"TFX no-graphs {thunder_graphs}")
        graph_id = copy.deepcopy(thunder_graphs)
        def wrapper_no_graphs(*args):
          torch.cuda.nvtx.range_push(f"TFX no-graph run {graph_id}")
          o = fqn(*args)
          torch.cuda.nvtx.range_pop()
          return o
        thunder_graphs = thunder_graphs + 1
        return wrapper_no_graphs
      torch.cuda.nvtx.mark(f"TFX GRAPH {thunder_graphs}")

      wrapper = torch.cuda.make_graphed_callables(
        fqn, args,
        num_warmup_iters=1,
        allow_unused_input=True
      )
      graph_id = copy.deepcopy(thunder_graphs)
      def wrapper_of_wrapper(*args):
        torch.cuda.nvtx.range_push(f"TFX run {graph_id}")
        o = wrapper(*args)
        torch.cuda.nvtx.range_pop()
        return o
      thunder_graphs = thunder_graphs + 1

      return wrapper_of_wrapper
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    import traceback
    print(f"broke: {e}")
    traceback.print_exception(e)
    print(f"because of input: {gm.graph}")
    return gm
  return gm

def thunder_graph_backend3(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      fqn = thunder.jit(gm)
      torch.cuda.nvtx.range_push(f"ThunderFX {thunder_graphs}")
      fqn(*args) # run once outside the graph to make the allocator happy?
      torch.cuda.nvtx.range_pop()

      if use_cuda_graph(copy.deepcopy(thunder_graphs)):
        graph_id = copy.deepcopy(thunder_graphs)
        reproducer(gm, args, graph_id)
      thunder_graphs = thunder_graphs + 1
      return fqn
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    import traceback
    print(f"broke: {e}")
    traceback.print_exception(e)
    print(f"because of input: {gm.graph}")
    return gm
  return gm

# use thunder's transform
def thunder_graph_backend4(gm: torch.fx.GraphModule, args: list[torch.Tensor], **kwargs):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      if use_cuda_graph(thunder_graphs):
        xform = thunder.transforms.cudagraph.CUDAGraphTransform()
        fqn = thunder.jit(gm, transforms=[xform])
      else:
        fqn = thunder.jit(gm)
      torch.cuda.nvtx.range_push(f"ThunderFX {thunder_graphs} warmup")
      fqn(*args) # run once to force compilation
      torch.cuda.nvtx.range_pop()

      graph_id = copy.deepcopy(thunder_graphs)
      def wrapper(*args):
        torch.cuda.nvtx.range_push(f"TFX graph run {graph_id}")
        o = fqn(*args)
        torch.cuda.nvtx.range_pop()
        return o
      thunder_graphs = thunder_graphs + 1
      return wrapper
    else:
      print(f"Thunder reports graph is not supported, so skipping.")
  except Exception as e:
    import traceback
    print(f"broke: {e}")
    traceback.print_exception(e)
    print(f"because of input: {gm.graph}")
    return gm
  return gm

def old_thunder_backend(gm: torch.fx.GraphModule, args):
  gm.real_recompile()
  global num_graphs
  global thunder_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      #fqn = thunder.jit(gm, executors=_execs)
      fqn = thunder.jit(gm)
      thunder_graphs = thunder_graphs + 1
      return fqn
    else:
      print("Falling back to eager for unsupported graph.")
      #return gm
  except Exception as e:
    print("Unsupported graph: {e}. Falling back to eager.")
    return gm
  return gm

def thunder_examine(gm: torch.fx.GraphModule, args):
  print(f"Compiling gm: {gm.graph}\nthunder support: {thunder_supported(gm)}")
  gm.real_recompile()
  if thunder_supported(gm):
    try: # Examine may raise an error
        thunder.examine.examine(gm, *args)
    except Exception as e:
        print(f"Hit problem with examine:\n{e}")
  return gm


def count_graphs(gm: torch.fx.GraphModule, args):
  #gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  return gm

@hydra_runner(config_path="conf", config_name="neva_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronNevaModel(cfg.model, trainer)
    use_thunder: str = os.getenv("NEMO_THUNDER_NEVA")
    if use_thunder is not None and use_thunder.strip() == "thunder":
        model.model = thunder.jit(model.model)
    elif use_thunder is not None and use_thunder.strip() == "inductor":
        model.model = torch.compile(model.model, mode="reduce-overhead")
    elif use_thunder is not None and use_thunder.strip() == "dynamo":
        # The dynamic=False is critical because we end up with SymInts
        # in the trace otherwise, which we cannot yet handle.
        #model.model = torch.compile(backend=thunder_backend, dynamic=False)(model.model)
        model.model = torch.compile(backend=logging_thunder_backend, dynamic=False)(model.model)
        #model.model = torch.compile(model.model, backend=thunder_graph_backend2, dynamic=False, mode="default")
    elif use_thunder is not None and use_thunder.strip() == "examine":
        model.model = torch.compile(backend=thunder_examine)(model.model)
    elif use_thunder is not None and use_thunder.strip() == "tc":
        #model.model = torch.compile(model.model)
        model.model = torch.compile(model.model, backend=cugraph)
    elif use_thunder is not None:
        raise ValueError(f"unknown NEMO_THUNDER_NEVA setting: {use_thunder}")

    try:
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        trainer.fit(model)
        #prof.export_chrome_trace(f"/tmp/nevaprofile-1.json")
    finally:
        teardown(trainer)
        global num_graphs
        global thunder_graphs
        global empty_graphs
        print(f"{thunder_graphs=} ({empty_graphs} empty) / {num_graphs=}")


if __name__ == '__main__':
    main()
