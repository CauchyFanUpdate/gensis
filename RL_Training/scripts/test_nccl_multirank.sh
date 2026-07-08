#!/bin/bash
# 多 rank 跨节点 NCCL 测试 —— 复现训练中 FSDP sub-PG 场景
#
# 机器A (master, NODE_RANK=0):
#   NODE_RANK=0 MASTER_ADDR=10.144.206.227 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/test_nccl_multirank.sh
#
# 机器B (worker, NODE_RANK=1):
#   NODE_RANK=1 MASTER_ADDR=10.144.206.227 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/test_nccl_multirank.sh
#
# 可选参数（环境变量）：
#   N_GPUS         每节点 GPU 数，默认 8
#   NNODES         节点总数，默认 2
#   MASTER_PORT    默认 29600（避免和训练端口冲突）
#   TIMEOUT        每步超时秒数，默认 120

NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-10.144.206.227}
MASTER_PORT=${MASTER_PORT:-29600}
NNODES=${NNODES:-2}
N_GPUS=${N_GPUS:-8}
TIMEOUT=${TIMEOUT:-120}

# ── conda NCCL 置顶 ──────────────────────────────────────────────
_CONDA=${CONDA_PREFIX:-/root/miniconda3/envs/verl_qwen3}
_NCCL_LIB="${_CONDA}/lib/python3.12/site-packages/nvidia/nccl/lib"
_TORCH_LIB="${_CONDA}/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${_NCCL_LIB}:${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
unset _CONDA _NCCL_LIB _TORCH_LIB

# ── NCCL 强制 Socket（与训练一致）────────────────────────────────
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=eth0
unset NCCL_IB_QPS_PER_CONNECTION NCCL_IB_SPLIT_DATA_ON_QPS NCCL_IB_TC
unset NCCL_IB_SL NCCL_IB_TIMEOUT NCCL_IB_HCA NCCL_IB_GID_INDEX
unset NCCL_MIN_NCHANNELS NCCL_NVLS_ENABLE NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV,BOOTSTRAP

echo "========================================"
echo "[test_multirank] NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[test_multirank] ${NNODES} nodes x ${N_GPUS} GPUs = $((NNODES * N_GPUS)) total ranks"
echo "[test_multirank] python: $(which python3)"
echo "========================================"

# Python 脚本写到临时文件（heredoc 内不能直接用 torchrun -m）
PY_SCRIPT=$(mktemp /tmp/test_nccl_multirank_XXXXXX.py)
cat > "$PY_SCRIPT" << 'PYEOF'
import os, torch, torch.distributed as dist

rank       = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
n_gpus     = world_size // int(os.environ.get("NNODES", 2))

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def log(msg):
    print(f"[rank{rank:02d}/local{local_rank}] {msg}", flush=True)

log(f"torch={torch.__version__} nccl={torch.cuda.nccl.version()} cuda={torch.version.cuda}")
log(f"NCCL_NET_PLUGIN={os.environ.get('NCCL_NET_PLUGIN','<not set>')}")
log(f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE','<not set>')}")
log(f"NCCL_NET={os.environ.get('NCCL_NET','<not set>')}")

# ── 步 1: 初始化全局 PG（与训练 initialize_global_process_group_ray 一致）────
log("Step 1: init_process_group nccl ...")
dist.init_process_group("nccl", init_method="env://")
log("Step 1: global PG OK")

# 全局 all_reduce 验证主 PG 可用
t = torch.ones(1, device=device) * rank
dist.all_reduce(t)
expected = world_size * (world_size - 1) / 2
ok = abs(t.item() - expected) < 1e-3
log(f"Step 1: global all_reduce = {t.item()} (expected {expected}) -> {'OK' if ok else 'WRONG'}")
assert ok, "global all_reduce failed"

# ── 步 2: 创建跨节点 sub-PG（模拟 FSDP DP group）────────────────────────────
# 每个 local_rank 和另一个节点上相同 local_rank 组成一组
# e.g. 16 ranks, 8/node: groups = [0,8],[1,9],[2,10],...,[7,15]
log("Step 2: creating cross-node sub-PGs (simulates FSDP DP groups) ...")
cross_node_groups = []
for i in range(n_gpus):
    ranks_in_group = list(range(i, world_size, n_gpus))
    pg = dist.new_group(ranks=ranks_in_group, backend="nccl")
    if rank in ranks_in_group:
        cross_node_groups.append((ranks_in_group, pg))

log(f"Step 2: {len(cross_node_groups)} cross-node sub-PG(s) created — checking ...")

for ranks_in_group, pg in cross_node_groups:
    t = torch.ones(1, device=device) * rank
    dist.all_reduce(t, group=pg)
    expected = sum(ranks_in_group)
    ok = abs(t.item() - expected) < 1e-3
    log(f"Step 2: cross-node all_reduce group={ranks_in_group} result={t.item()} -> {'OK' if ok else 'WRONG'}")
    assert ok, f"cross-node sub-PG all_reduce failed for group {ranks_in_group}"

# ── 步 3: sub-PG 内 broadcast（模拟 FSDP broadcast state dict）─────────────
log("Step 3: broadcast within cross-node sub-PG (simulates FSDP _broadcast_state_dict) ...")
for ranks_in_group, pg in cross_node_groups:
    t = torch.zeros(4, device=device)
    if rank == ranks_in_group[0]:
        t = torch.arange(4, dtype=torch.float32, device=device)
    dist.broadcast(t, src=ranks_in_group[0], group=pg)
    ok = torch.allclose(t, torch.arange(4, dtype=torch.float32, device=device))
    log(f"Step 3: broadcast group={ranks_in_group} -> {'OK' if ok else f'WRONG: {t}'}")
    assert ok, f"broadcast failed for group {ranks_in_group}"

dist.destroy_process_group()
log("ALL TESTS PASSED")
PYEOF

# ── 用 torchrun 启动 ──────────────────────────────────────────────
timeout "$TIMEOUT" torchrun \
    --nproc_per_node="$N_GPUS" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=static \
    "$PY_SCRIPT"

EXIT=$?
rm -f "$PY_SCRIPT"

echo "========================================"
if [ $EXIT -eq 124 ]; then
    echo "[test_multirank] TIMEOUT — NCCL hung on sub-PG bootstrap"
elif [ $EXIT -ne 0 ]; then
    echo "[test_multirank] FAILED (exit=$EXIT)"
    echo "[test_multirank] Check /tmp/nccl_rank*.log for NCCL INFO details"
else
    echo "[test_multirank] SUCCESS — all cross-node sub-PG tests passed"
fi
echo "========================================"
exit $EXIT
