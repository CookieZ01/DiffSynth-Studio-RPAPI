import os
import time
import uuid
import threading
import logging
import sys

import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffsynth.models.downloader import download_from_huggingface
from diffsynth import ModelManager, WanVideoPipeline, save_video

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO, # 设置为你需要的级别，INFO 通常足够
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout) # 输出到控制台
        # logging.FileHandler("api.log", encoding='utf-8') # 可选：输出到文件
    ]
)
logger = logging.getLogger("视频生成API")
# 如果想看到 diffsynth 库自身的 DEBUG 日志 (如果它有的话)，可以不单独获取 logger
# 或者获取 diffsynth 的 logger 并设置级别
# logging.getLogger("diffsynth").setLevel(logging.DEBUG) # 尝试获取并设置 diffsynth 库的日志级别

# --- 日志配置结束 ---

# 全局任务状态和 GPU 锁
task_status = {}
gpu_lock = threading.Lock()

# 创建 FastAPI 应用
app = FastAPI()

# 请求体模型
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    height: int = 320
    width: int = 480
    num_inference_steps: int = 20
    tiled: bool = True
    tea_cache_l1_thresh: float | None = 0.05
    tea_cache_model_id: str = "Wan2.1-T2V-1.3B"
    fps: int = 15
    quality: int = 5

# 启动时加载模型和管线
@app.on_event("startup")
def init_pipeline():
    # 确定目标计算设备
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"目标计算设备: {target_device}")
    if target_device == "cpu":
        logger.warning("未检测到 CUDA 设备，将在 CPU 上运行，速度会非常慢！")

    # 定义模型ID和本地路径
    model_hf_id = "Wan-AI/Wan2.1-T2V-1.3B" # 确保与请求中的 tea_cache_model_id 匹配
    local_model_dir = "models/Wan-AI/Wan2.1-T2V-1.3B"
    required_files = [
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ]
    expected_dtype = torch.bfloat16 # 定义期望的数据类型

    # 下载模型文件
    logger.info(f"确保模型文件在 {local_model_dir}...")
    os.makedirs(local_model_dir, exist_ok=True)
    for filename in required_files:
        local_path = os.path.join(local_model_dir, filename)
        if not os.path.exists(local_path):
            logger.info(f"    正在下载 {filename}...")
            try:
                download_from_huggingface(
                    model_id=model_hf_id,
                    origin_file_path=filename,
                    local_dir=local_model_dir
                )
                logger.info(f"    {filename} 下载完成.")
            except Exception as e:
                logger.error(f"从 Hugging Face 下载 {filename} 失败: {e}", exc_info=True)
                # 在启动时失败通常是致命的，可以选择退出或引发异常
                raise RuntimeError(f"模型文件 {filename} 下载失败，服务无法启动") from e
        else:
             logger.info(f"    {filename} 已存在.")

    # --- 模型加载策略调整 ---
    # 1. 先在 CPU 上初始化 ModelManager
    logger.info("在 CPU 上初始化 ModelManager...")
    model_manager = ModelManager(device="cpu") # 强制在 CPU 加载

    # 2. 加载模型到 ModelManager (仍在 CPU)
    logger.info(f"加载模型文件到 CPU (使用 dtype: {expected_dtype})...")
    try:
        model_manager.load_models(
            [os.path.join(local_model_dir, f) for f in required_files],
            torch_dtype=expected_dtype # 指定期望的数据类型
        )
    except Exception as e:
        logger.error(f"加载模型文件失败: {e}", exc_info=True)
        raise RuntimeError("模型文件加载失败，服务无法启动") from e

    # 3. 创建管线，并指定移动到目标设备 (如 "cuda")
    logger.info(f"创建 WanVideoPipeline 并将其移动到 {target_device}...")
    try:
        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=expected_dtype, # 保持数据类型一致
            device=target_device # 在这里指定最终运行的设备
        )
    except Exception as e:
        logger.error(f"创建管线失败: {e}", exc_info=True)
        raise RuntimeError("创建管线失败，服务无法启动") from e

    # 4. 启用 VRAM 管理 (与示例一致)
    logger.info("启用 VRAM 管理 (num_persistent_param_in_dit=None)...")
    # 只有在 CUDA 设备上运行时才启用 VRAM 管理才有意义
    if target_device == "cuda":
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
    else:
        logger.info("在 CPU 上运行，跳过 VRAM 管理配置。")

    # 保存到应用状态
    app.state.pipe = pipe
    app.state.target_device = target_device # 保存设备信息，以备后用
    logger.info("管线初始化完成.")

# 后台任务执行函数
def run_task(task_id: str, params: dict):
    task_start_time = time.time()
    logger.info(f"任务 {task_id}: 开始处理，参数: {params}")
    task_status[task_id]["status"] = "processing"
    pipe: WanVideoPipeline = app.state.pipe # 获取管线实例
    target_device = app.state.target_device # 获取目标设备

    try:
        # 将请求参数转换为 pipe.__call__ 需要的参数
        pipe_params = {
            "prompt": params["prompt"],
            "negative_prompt": params["negative_prompt"],
            "height": params["height"],
            "width": params["width"],
            "num_inference_steps": params["num_inference_steps"],
            "seed": params["seed"],
            "tiled": params["tiled"],
            "tea_cache_l1_thresh": params["tea_cache_l1_thresh"],
            "tea_cache_model_id": params["tea_cache_model_id"],
            # 可以根据 GenerateRequest 添加其他可选参数
            # "cfg_scale": params.get("cfg_scale", 5.0), # 使用 .get 获取可选参数
            # "motion_bucket_id": params.get("motion_bucket_id"),
            # "denoising_strength": params.get("denoising_strength", 1.0),
            "rand_device": "cpu" # 噪声生成设备，保持为 CPU
        }

        acquire_lock_start_time = time.time()
        logger.info(f"任务 {task_id}: 正在等待 GPU 锁...")
        with gpu_lock:
            lock_acquired_time = time.time()
            logger.info(f"任务 {task_id}: 已获取 GPU 锁 (等待 {lock_acquired_time - acquire_lock_start_time:.2f} 秒). 设置随机种子 {params['seed']}.")
            # 在目标设备上设置种子，如果使用 GPU
            if target_device == "cuda":
                torch.cuda.manual_seed_all(params["seed"])
            else:
                torch.manual_seed(params["seed"])

            logger.info(f"任务 {task_id}: 即将开始核心视频生成计算 (调用 pipe)...")
            pipe_start_time = time.time()
            video = pipe(**pipe_params) # 使用解包传递参数
            pipe_end_time = time.time()
            logger.info(f"任务 {task_id}: 核心视频生成计算 (pipe 调用) 完成 (耗时: {pipe_end_time - pipe_start_time:.2f} 秒).")

        # 确保输出目录存在
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        # 使用 UUID 或更可预测的任务 ID 命名文件
        # filename = f"output_{task_id}.mp4"
        filename = f"output_{int(time.time() * 1000)}.mp4" # 保持旧方式，或切换到 task_id
        filepath = os.path.join(output_dir, filename)

        logger.info(f"任务 {task_id}: 保存视频到 {filepath} (fps={params['fps']}, quality={params['quality']}).")
        save_start_time = time.time()
        save_video(video, filepath, fps=params["fps"], quality=params["quality"])
        save_end_time = time.time()
        logger.info(f"任务 {task_id}: 视频保存成功 (耗时: {save_end_time - save_start_time:.2f} 秒).")

        task_end_time = time.time()
        task_status[task_id].update({
            "status": "completed",
            "filepath": filepath,
            "duration_seconds": round(task_end_time - task_start_time, 2)
        })

    except Exception as e:
        logger.error(f"任务 {task_id}: 处理失败!", exc_info=True) # exc_info=True 会记录完整的 traceback
        task_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "duration_seconds": round(time.time() - task_start_time, 2)
        })
    finally:
         logger.info(f"任务 {task_id}: 处理函数执行完毕 (最终状态: {task_status[task_id].get('status')}).")

# 提交异步生成任务
@app.post("/generate_video_async")
async def generate_video_async(
    req: GenerateRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    logger.info(f"收到新任务请求: {task_id}, prompt: '{req.prompt[:50]}...'") # 记录部分 prompt
    task_status[task_id] = {
        "status": "queued",
        "start_time_epoch": time.time(), # 记录任务加入队列的时间戳
        "request_params": req.dict() # 可以选择性记录请求参数用于调试
    }
    background_tasks.add_task(run_task, task_id, req.dict())
    logger.info(f"任务 {task_id}: 已添加到后台队列.")
    return {"message": "任务已加入队列等待处理", "task_id": task_id}

# 查询任务状态
@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    logger.debug(f"查询任务状态: {task_id}")
    info = task_status.get(task_id)
    if not info:
        logger.warning(f"尝试查询不存在的任务: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    # 可以选择性地清理返回信息，比如不返回完整的请求参数
    response_info = info.copy()
    response_info.pop("request_params", None) # 示例：移除请求参数
    return response_info

# 下载结果视频
@app.get("/download_result/{task_id}")
async def download_result(task_id: str):
    logger.debug(f"请求下载结果: {task_id}")
    info = task_status.get(task_id)
    if not info:
        logger.warning(f"尝试下载不存在任务的结果: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    if info.get("status") != "completed":
        logger.warning(f"尝试下载未完成任务的结果: {task_id}, 状态: {info.get('status')}")
        raise HTTPException(status_code=404, detail=f"任务状态为 {info.get('status')}, 结果不可用")
    if not info.get("filepath") or not os.path.exists(info["filepath"]):
         logger.error(f"任务 {task_id} 状态为 completed 但找不到结果文件: {info.get('filepath')}")
         raise HTTPException(status_code=500, detail="结果文件丢失")

    logger.info(f"提供下载文件: {info['filepath']} (任务 {task_id})")
    return FileResponse(
        path=info["filepath"],
        media_type="video/mp4",
        filename=os.path.basename(info["filepath"])
    )

# 健康检查
@app.get("/health")
async def health():
    # 可以添加更复杂的健康检查，例如检查模型是否加载成功
    pipe_loaded = hasattr(app.state, 'pipe') and app.state.pipe is not None
    return {"status": "ok" if pipe_loaded else "error", "pipe_loaded": pipe_loaded}
