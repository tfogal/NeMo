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
import functools
import os
import typing

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.lightning.base import teardown
import thunder
import torch
import torch._dynamo
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

num_graphs = 0
thunder_graphs = 0
thunder_graphs_list = []

def thunder_backend(gm: torch.fx.GraphModule, args):
  gm.real_recompile()
  global num_graphs
  num_graphs = num_graphs + 1
  try:
    if thunder_supported(gm):
      global thunder_graphs
      global thunder_graphs_list
      thunder_graphs = thunder_graphs + 1
      # import pdb; pdb.set_trace()
      x = thunder.jit(gm)
      thunder_graphs_list.append(x)
      # out = x(args)
      # print(thunder.last_traces(x)[-1])
      return x
      # return thunder.jit(gm)
  except e:
    print("broke: {e}")
    print("because of input: {gm.graph}")
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
        model.model = torch.compile(model.model)
    elif use_thunder is not None and use_thunder.strip() == "dynamo":
        # The dynamic=False is critical because we end up with SymInts
        # in the trace otherwise, and this ends up dying on us.
        model.model = torch.compile(backend=thunder_backend, dynamic=False)(model.model)
    elif use_thunder is not None and use_thunder.strip() == "examine":
        model.model = torch.compile(backend=thunder_examine)(model.model)
    elif use_thunder is not None:
        raise ValueError(f"unknown NEMO_THUNDER_NEVA setting: {use_thunder}")

    try:
        trainer.fit(model)
    finally:
        # Grab the execution traces
        # fwd_trace = thunder.last_traces(model.model)[-1]
        # bwd_trace = thunder.last_backward_traces(model.model)[-1]

        # print("forward")
        # print(fwd_trace)
        # # for k, v in fwd_trace.python_ctx().items():
        # #     if 'nvFusion' in k:
        # #         print(v.last_used)
            
        # print("backward")
        # print(bwd_trace)
        # # for k, v in bwd_trace.python_ctx().items():
        # #     if 'nvFusion' in k:
        # #         print(v.last_used)
        # import pdb; pdb.set_trace()
        teardown(trainer)
        global num_graphs
        global thunder_graphs
        print(f"{thunder_graphs=} / {num_graphs=}")


if __name__ == '__main__':
    main()
