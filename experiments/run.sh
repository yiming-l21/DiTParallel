#!/usr/bin/env bash
set -euo pipefail
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 把当前仓库加入 PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

######################
#  小工具函数        #
######################

# 统计一个目录中已有的图片数量（支持 png/jpg/jpeg）
count_images() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo 0
        return
    fi
    find "$dir" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l
}

######################
#  模型类型选择区域  #
######################
export MODEL_TYPE="Flux"

# 脚本 / 模型路径 / 推理步数配置
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py /export/home/liuyiming54/flux-dev 28"
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

# 总 prompt 数（忽略空行）  ==== 断点续跑相关 ====
TOTAL_PROMPTS=$(awk 'NF>0{c++} END{print c+0}' "${PROMPTS_FILE}")
echo "[INFO] TOTAL_PROMPTS=${TOTAL_PROMPTS}"

# 只 profile 第 PROF_STEP 个 denoising step（0-based）
PROF_STEP=10

######################
#   baseline(单卡)   #
######################

# 单卡 baseline：results/base/imgs
BASE_RESULT_ROOT="./results/base"
BASE_IMG_DIR="${BASE_RESULT_ROOT}/imgs"
mkdir -p "${BASE_RESULT_ROOT}"

# 统计 baseline 目录已有图片数  ==== 断点续跑相关 ====
EXISTING_BASE_IMAGES=$(count_images "${BASE_IMG_DIR}")
echo "[INFO] Baseline existing images: ${EXISTING_BASE_IMAGES}"

if (( EXISTING_BASE_IMAGES >= TOTAL_PROMPTS )); then
    echo "[INFO] Baseline images already complete (${EXISTING_BASE_IMAGES}/${TOTAL_PROMPTS}), skip baseline generation."
else
    if (( EXISTING_BASE_IMAGES > 0 )); then
        echo "[INFO] Resume baseline: found ${EXISTING_BASE_IMAGES} images, continue from prompt index $((EXISTING_BASE_IMAGES+1))."
    else
        echo "[INFO] Baseline images not found, start from scratch."
    fi

    mkdir -p "${BASE_IMG_DIR}"

    # 生成一个临时 prompts 文件，只包含「尚未生成」的 prompts  ==== 断点续跑相关 ====
    TMP_BASE_PROMPTS_FILE=$(mktemp)
    # tail -n +K 表示从第 K 行开始，这里是从已有图片数 + 1 行开始
    tail -n +"$((EXISTING_BASE_IMAGES + 1))" "${PROMPTS_FILE}" > "${TMP_BASE_PROMPTS_FILE}"

    export RESULT_ROOT="${BASE_RESULT_ROOT}"
    # 让内部脚本从 EXISTING_BASE_IMAGES 这个 run_id 开始编号  ==== 断点续跑相关 ====
    export RUN_ID="${EXISTING_BASE_IMAGES}"

    python -m torch.distributed.run --nproc_per_node=1 "./experiments/${SCRIPT}" \
        --model "${MODEL_ID}" \
        ${TASK_ARGS} \
        --num_inference_steps "${INFERENCE_STEP}" \
        --warmup_steps 1 \
        --prompts_file "${TMP_BASE_PROMPTS_FILE}"

    rm -f "${TMP_BASE_PROMPTS_FILE}"
fi

######################
#   并行 / 方法配置  #
######################

# 使用的 GPU 数
N_GPUS=2

# 并行配置
PARALLEL_ARGS="--pipefusion_parallel_degree 2 --ulysses_degree 1 --ring_degree 1"
# 方法名（用于结果目录分类；你可以换成 sp / dp / ring 等）
METHOD="pipefusion"

# 根结果目录：results/<method>/<ngpus>
RESULT_ROOT="./results/${METHOD}/${N_GPUS}"
mkdir -p "${RESULT_ROOT}"

######################
#       运行并行      #
######################

# 本次实验的 trace 目录：results/<method>/<ngpus>/traces
TRACE_DIR="${RESULT_ROOT}/traces"
mkdir -p "${TRACE_DIR}"

# 并行方法的 imgs 目录（由 Python 脚本创建）  ==== 断点续跑相关 ====
METHOD_IMG_DIR="${RESULT_ROOT}/imgs"

# 统计并行方法已有的图片数  ==== 断点续跑相关 ====
EXISTING_METHOD_IMAGES=$(count_images "${METHOD_IMG_DIR}")
echo "[INFO] Method(${METHOD}) existing images: ${EXISTING_METHOD_IMAGES}"

if (( EXISTING_METHOD_IMAGES >= TOTAL_PROMPTS )); then
    echo "[INFO] Method ${METHOD} images already complete (${EXISTING_METHOD_IMAGES}/${TOTAL_PROMPTS}), skip parallel run."
else
    if (( EXISTING_METHOD_IMAGES > 0 )); then
        echo "[INFO] Resume method ${METHOD}: found ${EXISTING_METHOD_IMAGES} images, continue from prompt index $((EXISTING_METHOD_IMAGES+1))."
    else
        echo "[INFO] No method ${METHOD} images found, start from scratch."
    fi

    # 生成一个临时 prompts 文件，只包含尚未生成的 prompts  ==== 断点续跑相关 ====
    TMP_METHOD_PROMPTS_FILE=$(mktemp)
    tail -n +"$((EXISTING_METHOD_IMAGES + 1))" "${PROMPTS_FILE}" > "${TMP_METHOD_PROMPTS_FILE}"

    # 这一次就不在外层循环 prompt 了，所有 prompt 交给脚本内部处理
    export RESULT_ROOT="${RESULT_ROOT}"
    # 并行实验内部的 run_id 起始 offset，从已有图片数继续  ==== 断点续跑相关 ====
    export RUN_ID="${EXISTING_METHOD_IMAGES}"

    python -m torch.distributed.run --nproc_per_node="${N_GPUS}" --master_port $(( 12345 + RANDOM % 20000 )) \
        "./experiments/${SCRIPT}" \
        --model "${MODEL_ID}" \
        ${PARALLEL_ARGS} \
        ${TASK_ARGS} \
        --num_inference_steps "${INFERENCE_STEP}" \
        --warmup_steps 1 \
        --prompts_file "${TMP_METHOD_PROMPTS_FILE}" \
        --torch_profiler \
        --torch_profiler_step "${PROF_STEP}" \
        --torch_profiler_dir "${TRACE_DIR}" \
        --method_tag ${METHOD}

    rm -f "${TMP_METHOD_PROMPTS_FILE}"
fi

######################
#   2) 解析所有 trace #
######################

COMM_DIR="${RESULT_ROOT}/comm"
mkdir -p "${COMM_DIR}"

TRACE_PATTERN="${TRACE_DIR}/trace_dp*_run*_rank0_of${N_GPUS}.json"

shopt -s nullglob
TRACE_FILES=(${TRACE_PATTERN})

if (( ${#TRACE_FILES[@]} == 0 )); then
    echo "[WARN] No trace files found in ${TRACE_DIR} for pattern=${TRACE_PATTERN}"
else
    for TRACE_FILE in "${TRACE_FILES[@]}"; do
        base=$(basename "${TRACE_FILE}")      # trace_xxx.json
        base_noext=${base%.json}             # trace_xxx
        tmp=${base_noext#trace_}             # 去掉前缀 trace_

        # 例如：
        #   tmp = dp1_cfg1_ulysses2_ring1_tp1_pp1_patchNone_run0003_rank0_of2
        #   method_tag = dp1_cfg1_ulysses2_ring1_tp1_pp1_patchNone_run0003
        method_tag=${tmp%_rank0*}

        # 从 method_tag 里把 run 索引抠出来：
        #   method_tag=..._run0003  => run_str=0003 => img_idx=3
        run_str=${method_tag##*_run}
        img_idx=$((10#${run_str}))

        COMM_FILE="${COMM_DIR}/comm_${method_tag}_${img_idx}.txt"

        echo "[INFO] analyze trace: ${TRACE_FILE} -> ${COMM_FILE}"

        python "./experiments/utils/analyze_tracing.py" \
            "${TRACE_FILE}" \
            --device 0 \
            --step "${PROF_STEP}" \
            > "${COMM_FILE}"
    done
fi
shopt -u nullglob

######################
#   3) 计算质量指标   #
######################

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
