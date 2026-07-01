import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import conf
from bert_classifer_model import BertClassifier, get_num_params
from utils import load_raw_data, build_dataloader, evaluate


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_layerwise_optimizer_params(model, base_lr, weight_decay, decay_rate=0.85):
    """
    构建层衰减学习率的优化器参数组
    顶层学习率 = base_lr，逐层往下乘 decay_rate
    """
    # 收集所有层名
    no_decay = ['bias', 'LayerNorm.weight']

    # 定义层的深度顺序（从底到顶）
    layer_depth = {}
    layer_depth['embeddings'] = 0
    for i in range(12):
        layer_depth[f'encoder.layer.{i}'] = i + 1
    layer_depth['pooler'] = 13
    layer_depth['classifier'] = 14

    # 分组
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 确定该参数所在层的深度
        depth = 14  # 默认顶层
        if 'embeddings' in name:
            depth = 0
        elif 'encoder.layer.' in name:
            for i in range(12):
                if f'encoder.layer.{i}.' in name:
                    depth = i + 1
                    break
        elif 'pooler' in name:
            depth = 13

        lr = base_lr * (decay_rate ** (14 - depth))
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        # 查找是否已有相同配置的组
        found = False
        for group in param_groups:
            if abs(group['lr'] - lr) < 1e-12 and abs(group['weight_decay'] - wd) < 1e-12:
                group['params'].append(param)
                found = True
                break

        if not found:
            param_groups.append({
                'params': [param],
                'lr': lr,
                'weight_decay': wd
            })

    return param_groups


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, grad_accum_steps):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = len(train_loader)

    optimizer.zero_grad()
    train_bar = tqdm(train_loader, desc="  [Train]")

    for step, batch in enumerate(train_bar):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels) / grad_accum_steps

        # 反向传播
        loss.backward()

        # 梯度累积更新
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 记录
        total_loss += loss.item() * grad_accum_steps
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        train_bar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

    avg_loss = total_loss / n_batches
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, train_acc, train_f1


def train_model():
    """训练主函数"""
    set_seed(conf.seed)

    print("=" * 70)
    print("FinBERT 模型训练 (优化版 - 20万数据)")
    print("=" * 70)

    # ========== 初始化1：数据 ==========
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    # ========== 初始化2：模型、优化器、损失函数 ==========
    model = BertClassifier()
    model.to(conf.device)

    total_params, trainable_params = get_num_params(model)
    print(f"\n模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
    print(f"可训练占比: {trainable_params / total_params * 100:.1f}%")

    # 层衰减学习率优化器
    optimizer_grouped_params = get_layerwise_optimizer_params(
        model,
        base_lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
        decay_rate=conf.layerwise_lr_decay
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_params)

    # 学习率调度器
    total_steps = len(train_loader) * conf.num_epochs // conf.gradient_accumulation_steps
    warmup_steps = int(total_steps * conf.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    print(f"\n总训练步数: {total_steps}")
    print(f"Warmup步数: {warmup_steps}")
    print(f"梯度累积步数: {conf.gradient_accumulation_steps} (等效 batch={conf.batch_size * conf.gradient_accumulation_steps})")

    os.makedirs(os.path.dirname(conf.model_save_path), exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # ========== 训练循环 ==========
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(conf.num_epochs):
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch + 1}/{conf.num_epochs}")
        print(f"{'─' * 70}")

        # 训练
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            conf.device, conf.gradient_accumulation_steps
        )

        # 验证
        dev_acc, dev_f1, dev_preds, dev_labels = evaluate(model, dev_loader, conf.device)

        print(f"\n  训练集 - Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  Macro-F1: {train_f1:.4f}")
        print(f"  验证集 - Acc:  {dev_acc:.4f}  Macro-F1: {dev_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'dev_acc': dev_acc,
            'dev_f1': dev_f1
        })

        # 保存最优模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), conf.model_save_path)
            print(f"  ★ 最优模型已保存 (Dev F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  早停计数: {patience_counter}/{conf.early_stop_patience}")
            if patience_counter >= conf.early_stop_patience:
                print(f"\n早停触发！连续 {conf.early_stop_patience} 个epoch无提升")
                break

    # ========== 测试集评估 ==========
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, conf.device)

    print(f"\n最佳Epoch: {best_epoch}")
    print(f"测试集 Acc:  {test_acc:.4f}")
    print(f"测试集 Macro-F1: {test_f1:.4f}")

    print("\n分类报告:")
    report = classification_report(test_labels, test_preds, target_names=conf.class_list)
    print(report)

    # 保存测试报告
    with open("../logs/test_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FinBERT 测试报告 (20万数据优化版)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"训练数据量: ~20万条 (8类均衡)\n")
        f.write(f"Max Seq Length: {conf.max_seq_length}\n")
        f.write(f"学习率: {conf.learning_rate} (层衰减 {conf.layerwise_lr_decay})\n")
        f.write(f"等效Batch Size: {conf.batch_size * conf.gradient_accumulation_steps}\n")
        f.write(f"最佳Epoch: {best_epoch}\n\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Macro-F1: {test_f1:.4f}\n\n")
        f.write(report)
        f.write("\n\n训练历史:\n")
        for h in history:
            f.write(f"Epoch {h['epoch']:2d}: "
                    f"train_loss={h['train_loss']:.4f}, "
                    f"train_acc={h['train_acc']:.4f}, "
                    f"dev_acc={h['dev_acc']:.4f}, "
                    f"dev_f1={h['dev_f1']:.4f}\n")

    print(f"\n测试报告已保存: ../logs/test_report.txt")
    print(f"最优模型已保存: {conf.model_save_path}")

    print("\n训练完成！")
    return model


if __name__ == "__main__":
    train_model()
