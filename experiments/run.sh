#!/usr/bin/env bash
set -euo pipefail
set -x
export CUDA_VISIBLE_DEVICES=0,1
# 把当前仓库加入 PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

######################
#  模型类型选择区域  #
######################
export MODEL_TYPE="Flux"

# 脚本 / 模型路径 / 推理步数配置
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py ./cfs/dit/FLUX.1-dev 28"
    ["FluxControl"]="flux_control_example.py /cfs/dit/FLUX.1-Depth-dev 28"
    ["HunyuanDiT"]="hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
    ["SDXL"]="sdxl_example.py /cfs/dit/stable-diffusion-xl-base-1.0 30"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

######################
#  通用任务 / Profiler #
######################

# 任务参数（分辨率 / CFG 等）
TASK_ARGS="--height 2048 --width 2048 --no_use_resolution_binning --guidance_scale 3.5"

# prompt 列表文件
PROMPTS_FILE="./experiments/data/prompts.txt"

# 只 profile 第 PROF_STEP 个 denoising step（0-based）
PROF_STEP=10

######################
#   baseline(单卡)   #
######################

# 单卡 baseline：results/base/imgs
BASE_RESULT_ROOT="./results/base"
BASE_IMG_DIR="${BASE_RESULT_ROOT}/imgs"

if [[ ! -d "${BASE_IMG_DIR}" ]] || [[ -z "$(ls -A "${BASE_IMG_DIR}" 2>/dev/null)" ]]; then
    echo "[INFO] Baseline images not found, running 1-GPU baseline to ${BASE_IMG_DIR} ..."
    mkdir -p "${BASE_RESULT_ROOT}"

    base_run_id=0
    while IFS= read -r B_PROMPT || [[ -n "${B_PROMPT}" ]]; do
        # 跳过空行
        if [[ -z "${B_PROMPT}" ]]; then
            continue
        fi
        # 跳过注释
        if [[ "${B_PROMPT}" =~ ^# ]]; then
            continue
        fi

        echo "=============================="
        echo " BASE RUN ${base_run_id}, prompt: ${B_PROMPT}"
        echo "=============================="

        # run_id 用 4 位零填充，保证排序一致：0000, 0001, ...
        BASE_RUN_ID_STR=$(printf "%04d" "${base_run_id}")
        export RUN_ID="${BASE_RUN_ID_STR}"
        export RESULT_ROOT="${BASE_RESULT_ROOT}"

        # 单卡不需要并行参数，也不需要 profiler（只要图片）
        torchrun --nproc_per_node=1 "./experiments/${SCRIPT}" \
            --model "${MODEL_ID}" \
            ${TASK_ARGS} \
            --num_inference_steps "${INFERENCE_STEP}" \
            --warmup_steps 1 \
            --prompt "${B_PROMPT}"

        base_run_id=$((base_run_id + 1))
    done < "${PROMPTS_FILE}"
else
    echo "[INFO] Baseline images already exist under ${BASE_IMG_DIR}, skip baseline generation."
fi

######################
#   并行 / 方法配置  #
######################

# 使用的 GPU 数
N_GPUS=2

# 并行配置（这里是 Ulysses = 2 卡场景）
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1 --num_pipeline_patch 2"

# 方法名（用于结果目录分类；你可以换成 sp / dp / ring 等）
METHOD="pp"

# 根结果目录：results/<method>/<ngpus>
RESULT_ROOT="./results/${METHOD}/${N_GPUS}"
mkdir -p "${RESULT_ROOT}"

######################
#       主循环       #
######################

run_id=0

while IFS= read -r PROMPT || [[ -n "${PROMPT}" ]]; do
    # 跳过空行
    if [[ -z "${PROMPT}" ]]; then
        continue
    fi
    # 跳过注释行（以 # 开头）
    if [[ "${PROMPT}" =~ ^# ]]; then
        continue
    fi

    echo "=============================="
    echo " RUN ${run_id}, prompt: ${PROMPT}"
    echo "=============================="

    # run_id 用 4 位零填充：0000, 0001, ...
    RUN_ID_STR=$(printf "%04d" "${run_id}")
    export RUN_ID="${RUN_ID_STR}"
    export RESULT_ROOT="${RESULT_ROOT}"

    # 本次实验的 trace 目录：results/<method>/<ngpus>/traces
    TRACE_DIR="${RESULT_ROOT}/traces"
    mkdir -p "${TRACE_DIR}"

    ##################################################################
    # 1) 运行模型推理 + torch.profiler（每个 rank 会输出一个 trace_*.json）
    ##################################################################
    torchrun --nproc_per_node="${N_GPUS}" "./experiments/${SCRIPT}" \
        --model "${MODEL_ID}" \
        ${PARALLEL_ARGS} \
        ${TASK_ARGS} \
        --num_inference_steps "${INFERENCE_STEP}" \
        --warmup_steps 1 \
        --prompt "${PROMPT}" \
        --torch_profiler \
        --torch_profiler_step "${PROF_STEP}" \
        --torch_profiler_dir "${TRACE_DIR}"

    ##################################################################
    # 2) 用 analyze_tracing.py 解析当前 run 的 rank0 trace，生成 comm_xxx_0.txt
    #    trace 文件名形如：
    #    trace_dp1_cfg1_ulysses2_ring1_tp1_pp1_patchNone_run0000_rank0_of2.json
    ##################################################################
    TRACE_PATTERN="${TRACE_DIR}/trace_dp*_run${RUN_ID_STR}_rank0_of${N_GPUS}.json"
    # 这里用 ls 让 * 展开成真实文件名，再取第一个
    TRACE_FILE=$(ls ${TRACE_PATTERN} 2>/dev/null | head -n 1 || true)

    if [[ -n "${TRACE_FILE}" && -f "${TRACE_FILE}" ]]; then
        base=$(basename "${TRACE_FILE}")          # trace_xxx.json
        base_noext=${base%.json}                 # trace_xxx
        tmp=${base_noext#trace_}                 # 去掉前缀 trace_
        # dp1_cfg1_ulysses2_ring1_tp1_pp1_patchNone_run0000_rank0_of2
        # 截到 _rank0 之前，得到：
        # dp1_cfg1_ulysses2_ring1_tp1_pp1_patchNone_run0000
        method_tag=${tmp%_rank0*}

        COMM_DIR="${RESULT_ROOT}/comm"
        mkdir -p "${COMM_DIR}"

        img_idx=0
        COMM_FILE="${COMM_DIR}/comm_${method_tag}_${img_idx}.txt"

        python "./experiments/utils/analyze_tracing.py" \
            "${TRACE_FILE}" \
            --device 0 \
            --step "${PROF_STEP}" \
            > "${COMM_FILE}"
    else
        echo "[WARN] Trace file not found for run_id=${run_id}: pattern=${TRACE_PATTERN}"
    fi

    run_id=$((run_id + 1))
done < "${PROMPTS_FILE}"

######################
#   3) 计算质量指标   #
######################

METHOD_IMG_DIR="${RESULT_ROOT}/imgs"
METRIC_OUT="${RESULT_ROOT}/metrics.txt"

if [[ -d "${BASE_IMG_DIR}" ]] && [[ -n "$(ls -A "${BASE_IMG_DIR}" 2>/dev/null)" ]]; then
    if [[ -d "${METHOD_IMG_DIR}" ]] && [[ -n "$(ls -A "${METHOD_IMG_DIR}" 2>/dev/null)" ]]; then
        echo "[INFO] Computing metrics (PSNR / LPIPS / FID) for ${METHOD}/${N_GPUS} vs baseline..."
        python "./experiments/utils/compute_metrics.py" \
            --input_root0 "${BASE_IMG_DIR}" \
            --input_root1 "${METHOD_IMG_DIR}" \
            > "${METRIC_OUT}"
        echo "[INFO] Metrics saved to ${METRIC_OUT}"
    else
        echo "[WARN] No images found under ${METHOD_IMG_DIR}, skip metric compute."
    fi
else
    echo "[WARN] No baseline images found under ${BASE_IMG_DIR}, skip metric compute."
fi
