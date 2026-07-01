import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from config import conf
from bert_classifer_model import BertClassifier
from utils import load_raw_data, build_dataloader, evaluate


def get_model_size(model_path):
    """获取模型文件大小（MB）"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0


def measure_inference_time(model, data_loader, device, num_samples=100):
    """测量推理时间"""
    model.eval()
    times = []

    sample_count = 0
    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            start_time = time.time()
            _ = model(input_ids, attention_mask)
            end_time = time.time()

            times.append((end_time - start_time) / input_ids.size(0) * 1000)  # ms per sample
            sample_count += input_ids.size(0)

    avg_time = sum(times) / len(times)
    return avg_time


def quantize_model():
    """动态量化模型"""
    print("=" * 60)
    print("FinBERT 模型量化")
    print("=" * 60)

    # 检查原始模型是否存在
    if not os.path.exists(conf.model_save_path):
        print(f"❌ 错误: 训练好的模型不存在: {conf.model_save_path}")
        print("请先运行 train.py 训练模型")
        return

    # 加载原始模型
    print("\n第一步：加载原始模型...")
    model = BertClassifier()
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    model.eval()

    # 统计原始模型信息
    original_size = get_model_size(conf.model_save_path)
    print(f"原始模型大小: {original_size:.2f} MB")

    # ========== 动态量化 ==========
    print("\n第二步：执行动态量化...")
    print("量化目标: nn.Linear 层")
    print("量化精度: INT8 (qint8)")

    # 动态量化：只在CPU上可用
    # 量化指定层：所有nn.Linear层
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # 要量化的层类型
        dtype=torch.qint8  # 量化精度
    )

    print("量化完成！")

    # 保存量化模型
    print("\n第三步：保存量化模型...")
    os.makedirs(os.path.dirname(conf.quantized_model_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), conf.quantized_model_path)

    quantized_size = get_model_size(conf.quantized_model_path)
    print(f"量化模型大小: {quantized_size:.2f} MB")

    # ========== 对比测试 ==========
    print("\n第四步：对比测试...")
    print("-" * 50)

    # 加载测试数据
    _, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, batch_size=8
    )

    # 测试原始模型
    print("\n原始模型 (FP32):")
    # 原始模型可在GPU上运行
    device_fp32 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fp32 = BertClassifier()
    model_fp32.load_state_dict(torch.load(conf.model_save_path, map_location=device_fp32))
    model_fp32.to(device_fp32)
    model_fp32.eval()

    acc_fp32, f1_fp32, _, _ = evaluate(model_fp32, test_loader, device_fp32)
    time_fp32 = measure_inference_time(model_fp32, test_loader, device_fp32, num_samples=100)

    print(f"  准确率: {acc_fp32:.4f}")
    print(f"  Macro-F1: {f1_fp32:.4f}")
    print(f"  单条推理时间: {time_fp32:.2f} ms")

    # 测试量化模型（只能在CPU上运行）
    print("\n量化模型 (INT8):")
    device_int8 = torch.device("cpu")
    model_int8 = torch.quantization.quantize_dynamic(
        BertClassifier(),
        {nn.Linear},
        dtype=torch.qint8
    )
    model_int8.load_state_dict(torch.load(conf.quantized_model_path, map_location=device_int8))
    model_int8.to(device_int8)
    model_int8.eval()

    acc_int8, f1_int8, _, _ = evaluate(model_int8, test_loader, device_int8)
    time_int8 = measure_inference_time(model_int8, test_loader, device_int8, num_samples=100)

    print(f"  准确率: {acc_int8:.4f}")
    print(f"  Macro-F1: {f1_int8:.4f}")
    print(f"  单条推理时间: {time_int8:.2f} ms")

    # ========== 对比报告 ==========
    print("\n" + "=" * 60)
    print("量化对比报告")
    print("=" * 60)
    print(f"{'指标':<20} {'原始模型(FP32)':<20} {'量化模型(INT8)':<20} {'变化':<15}")
    print("-" * 75)
    print(f"{'模型大小(MB)':<20} {original_size:<20.2f} {quantized_size:<20.2f} {quantized_size/original_size*100-100:+.1f}%")
    print(f"{'准确率':<20} {acc_fp32:<20.4f} {acc_int8:<20.4f} {(acc_int8-acc_fp32)*100:+.2f}%")
    print(f"{'Macro-F1':<20} {f1_fp32:<20.4f} {f1_int8:<20.4f} {(f1_int8-f1_fp32)*100:+.2f}%")
    print(f"{'推理时间(ms)':<20} {time_fp32:<20.2f} {time_int8:<20.2f} {(time_int8/time_fp32-1)*100:+.1f}%")

    size_reduction = (1 - quantized_size / original_size) * 100
    speed_up = time_fp32 / time_int8
    print(f"\n模型压缩率: {size_reduction:.1f}%")
    print(f"推理加速比: {speed_up:.2f}x")

    print("\n量化完成！")
    return quantized_model


if __name__ == "__main__":
    quantize_model()
