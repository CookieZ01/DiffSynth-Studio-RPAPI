import requests
import time
import json
import os

# --- 配置 ---
# 替换成你 FastAPI 服务器的实际 IP 地址和端口
BASE_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = "client_output" # 保存下载视频的目录
POLL_INTERVAL_SECONDS = 5 # 查询任务状态的间隔时间
# --- 配置结束 ---

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 定义生成请求的参数
generate_params = {
    "prompt": "一只可爱的猫咪在草地上玩耍", # 你想要的提示词
    "negative_prompt": "模糊, 低质量, 多余肢体",
    "seed": 123,
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "tiled": True,
    "tea_cache_l1_thresh": 0.05,
    "tea_cache_model_id": "Wan2.1-T2V-1.3B",
    "fps": 15,
    "quality": 5
}

print(f"[*] 向 {BASE_URL}/generate_video_async 发送生成请求...")
try:
    response = requests.post(f"{BASE_URL}/generate_video_async", json=generate_params)
    response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
    result = response.json()
    task_id = result.get("task_id")
    if not task_id:
        print("[!] 错误：未能从响应中获取 task_id。")
        print(result)
        exit()
    print(f"[*] 任务已提交，Task ID: {task_id}")

except requests.exceptions.RequestException as e:
    print(f"[!] 请求失败: {e}")
    exit()
except json.JSONDecodeError:
    print("[!] 错误：无法解析服务器响应 (不是有效的 JSON)。")
    print(response.text)
    exit()

# 2. 轮询任务状态
status_url = f"{BASE_URL}/task_status/{task_id}"
start_time = time.time()
print(f"[*] 开始轮询任务状态 (每 {POLL_INTERVAL_SECONDS} 秒)...")

while True:
    try:
        status_response = requests.get(status_url)
        status_response.raise_for_status()
        status_data = status_response.json()
        current_status = status_data.get("status")

        elapsed_time = round(time.time() - start_time, 1)
        print(f"    [{elapsed_time}s] 任务状态: {current_status}")

        if current_status == "completed":
            print("[+] 任务完成！")
            filepath = status_data.get("filepath") # 服务器上的文件路径
            duration = status_data.get("duration")
            print(f"    服务器端文件: {filepath}")
            print(f"    生成耗时: {duration} 秒")
            break
        elif current_status == "failed":
            error_msg = status_data.get("error", "未知错误")
            print(f"[!] 任务失败: {error_msg}")
            exit()
        elif current_status not in ["queued", "processing"]:
            print(f"[!] 未知任务状态: {current_status}")
            exit()

    except requests.exceptions.RequestException as e:
        print(f"[!] 查询状态失败: {e}")
    except json.JSONDecodeError:
        print("[!] 错误：无法解析状态响应。")
        print(status_response.text)

    time.sleep(POLL_INTERVAL_SECONDS)

# 3. 下载结果视频
download_url = f"{BASE_URL}/download_result/{task_id}"
output_filename = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")
print(f"[*] 开始下载结果视频到 {output_filename} ...")

try:
    download_response = requests.get(download_url, stream=True)
    download_response.raise_for_status()

    with open(output_filename, "wb") as f:
        for chunk in download_response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[+] 视频下载完成: {output_filename}")

except requests.exceptions.RequestException as e:
    print(f"[!] 下载失败: {e}")
except Exception as e:
     print(f"[!] 保存文件时出错: {e}")

print("[*] 客户端执行完毕。")
