# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        # Propagate MASTER_ADDR/PORT to remote workers so K8s-injected values
        # (e.g. a pod hostname or WORLD_SIZE=1) don't override the correct ones.
        "MASTER_ADDR": "127.0.0.1",  # overridden by os.environ at runtime
        "MASTER_PORT": "29500",      # overridden by os.environ at runtime
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # For each key, prefer the value from os.environ (set by the caller/job) over the
    # hardcoded default.  Do NOT remove the key — remote Ray actors on other nodes do
    # not inherit os.environ, so the key must stay in runtime_env to be propagated.
    for key in list(runtime_env["env_vars"].keys()):
        env_val = os.environ.get(key)
        if env_val is not None:
            runtime_env["env_vars"][key] = env_val

    # Propagate LD_LIBRARY_PATH so all remote actors load the same NCCL binary.
    # Without this, workers on nodes with system torch paths in LD_LIBRARY_PATH
    # may load a different libnccl.so.2 version, causing bootstrap size mismatch.
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        runtime_env["env_vars"]["LD_LIBRARY_PATH"] = ld_library_path

    return runtime_env
