#!/bin/bash
# 两台机器分别运行此脚本测试 NCCL 跨节点连通性
# 机器A: RANK=0 MASTER_ADDR=10.144.206.227 bash test_nccl.sh
# 机器B: RANK=1 MASTER_ADDR=10.144.206.227  bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/test_nccl.sh
#
# 超时后自动退出（默认 60 秒），无需手动 Ctrl+C

RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-10.144.206.227}
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-2}
TIMEOUT=${TIMEOUT:-60}

export MASTER_ADDR MASTER_PORT WORLD_SIZE RANK
# 把 conda NCCL 强制置顶，防止 K8s 注入的系统 torch 路径里有不同版本的 libnccl
_CONDA=${CONDA_PREFIX:-/root/miniconda3/envs/verl_qwen3}
_NCCL_LIB="${_CONDA}/lib/python3.12/site-packages/nvidia/nccl/lib"
_TORCH_LIB="${_CONDA}/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${_NCCL_LIB}:${_TORCH_LIB}"
unset _CONDA _NCCL_LIB _TORCH_LIB

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
# 清除 Ray/vLLM 在 rank1 上遗留的所有 IB/channel 参数
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_SPLIT_DATA_ON_QPS
unset NCCL_IB_TC
unset NCCL_IB_SL
unset NCCL_IB_TIMEOUT
unset NCCL_IB_HCA
unset NCCL_IB_GID_INDEX
unset NCCL_MIN_NCHANNELS
unset NCCL_NVLS_ENABLE
unset NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export NCCL_DEBUG_FILE="/tmp/nccl_rank${RANK}.log"
export GLOO_SOCKET_IFNAME=eth0

echo "[test_nccl] RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${WORLD_SIZE} timeout=${TIMEOUT}s"
echo "[test_nccl] python: $(which python3)"
echo "[test_nccl] eth0 addrs: $(hostname -I 2>/dev/null || echo 'n/a')"

# 检查系统路径中是否存在 NCCL net plugin（这是 2x bootstrap 大小的常见原因）
echo "[test_nccl] === NCCL plugin scan ==="
for dir in /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu \
           /usr/local/lib/python3.12/dist-packages/torch/lib \
           "${_CONDA:-/root/miniconda3/envs/verl_qwen3}/lib/python3.12/site-packages/torch/lib"; do
    found=$(ls "${dir}"/libnccl* 2>/dev/null | tr '\n' ' ')
    [ -n "$found" ] && echo "[test_nccl] ${dir}: ${found}"
done
echo "[test_nccl] NCCL_PLUGIN_PATH=${NCCL_PLUGIN_PATH:-<not set>}"
echo "[test_nccl] ==============================="

# 用 timeout 命令包裹 python，超时自动 SIGKILL
timeout "${TIMEOUT}" python3 << 'PY'
import os, subprocess, torch, torch.distributed as dist

rank = int(os.environ["RANK"])
print(f"[py] torch={torch.__version__} nccl={torch.cuda.nccl.version()} cuda={torch.version.cuda}", flush=True)
print(f"[py] MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}", flush=True)
print(f"[py] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')[:200]}", flush=True)
print(f"[py] NCCL_NET_PLUGIN={os.environ.get('NCCL_NET_PLUGIN','<not set>')}", flush=True)
try:
    ldc = subprocess.check_output(['ldconfig','-p'], stderr=subprocess.DEVNULL).decode()
    for l in ldc.splitlines():
        if 'libnccl-net' in l: print(f"[py] ldconfig: {l.strip()}", flush=True)
except Exception as e:
    print(f"[py] ldconfig error: {e}", flush=True)

# 确认运行时实际加载的 libnccl.so.2（读 /proc/self/maps）
# torch import 会触发 NCCL 加载，因此在 import torch 之后读
with open("/proc/self/maps") as f:
    for line in f:
        if "nccl" in line.lower():
            path = line.strip().split()[-1]
            print(f"[py] maps:nccl => {path}", flush=True)
            # md5 of the actual loaded lib
            try:
                md5 = subprocess.check_output(["md5sum", path], stderr=subprocess.DEVNULL).decode().split()[0]
                print(f"[py] md5({path.split('/')[-1]}) = {md5}", flush=True)
            except Exception:
                pass
            break
    else:
        print("[py] maps: no nccl entry found (not yet loaded)", flush=True)

# 列出系统上所有 libnccl.so*
try:
    out = subprocess.check_output(
        ["find", "/", "-name", "libnccl.so*", "-not", "-path", "*/proc/*"],
        stderr=subprocess.DEVNULL, timeout=10
    ).decode().strip()
    print(f"[py] all libnccl files:\n{out}", flush=True)
except Exception as e:
    print(f"[py] find nccl: {e}", flush=True)

# ld.so.preload
try:
    preload = open("/etc/ld.so.preload").read().strip()
    print(f"[py] /etc/ld.so.preload: {preload}", flush=True)
except FileNotFoundError:
    print("[py] /etc/ld.so.preload: not present", flush=True)

dist.init_process_group("nccl", init_method="env://")
print(f"[py] rank {rank}: process group initialized", flush=True)

# 在第一个 collective 之前再扫一次 maps，看 ncclCommInitRank 是否 dlopen 了额外 plugin
with open("/proc/self/maps") as f:
    seen = set()
    for line in f:
        if "nccl" in line.lower():
            p = line.strip().split()[-1]
            if p not in seen:
                seen.add(p)
                print(f"[py] pre-allreduce maps:nccl => {p}", flush=True)

t = torch.full((1,), float(rank)).cuda()
dist.all_reduce(t)
expected = 1.0
ok = abs(t.item() - expected) < 1e-5
print(f"[py] rank {rank}: all_reduce={t.item()} -> {'OK' if ok else 'WRONG'}", flush=True)

dist.destroy_process_group()
print(f"[py] rank {rank}: DONE", flush=True)
PY

EXIT=$?
if [ $EXIT -eq 124 ]; then
    echo "[test_nccl] TIMEOUT after ${TIMEOUT}s — NCCL hung, no cross-node connectivity"
elif [ $EXIT -ne 0 ]; then
    echo "[test_nccl] FAILED with exit code $EXIT"
else
    echo "[test_nccl] SUCCESS"
fi
