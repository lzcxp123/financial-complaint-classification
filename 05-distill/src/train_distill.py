import os
import numpy as np
import torch
from sklearn.metrics import classification_report

from config import conf
from distill_model import (
    TeacherFinBERT, StudentBiLSTM,
    load_teacher_model, load_student_model, get_num_params
)
from utils import build_dataloader, evaluate_student, init_student_vocab


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_distill():
    """蒸馏训练主函数 - 教师: FinBERT, 学生: BiLSTM+Attention"""
    set_seed(conf.seed)

    print("=" * 70)
    print("知识蒸馏训练")
    print("  教师模型: FinBERT (来自 04-bert)")
    print("  学生模型: BiLSTM+Attention (来自 03-bilstm)")
    print("=" * 70)

    # ========== 初始化1：数据 ==========
    print("\n[1/4] 加载数据...")
    train_loader, dev_loader, test_loader = build_dataloader(
        conf.train_path, conf.dev_path, conf.test_path, conf.batch_size
    )

    vocab_size = len(conf.student_vocab)
    print(f"词表大小: {vocab_size}")

    # ========== 初始化2：教师模型 + 学生模型 + 蒸馏训练器 ==========
    print("\n[2/4] 加载教师模型 (FinBERT)...")
    teacher = load_teacher_model()

    print("\n[3/4] 加载学生模型 (BiLSTM+Attention)...")
    student = load_student_model(vocab_size, load_pretrained=True)
    student.to(conf.device)

    teacher_params = get_num_params(teacher)
    student_params = get_num_params(student)
    print(f"\n教师参数量: {teacher_params:,}")
    print(f"学生参数量: {student_params:,}")
    print(f"压缩比: {student_params / teacher_params * 100:.1f}%")

    # 创建蒸馏训练器
    print("\n[4/4] 初始化蒸馏训练器...")
    from distill_trainer import DistillTrainer
    trainer = DistillTrainer(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=dev_loader,
        config=conf
    )

    # 创建保存目录
    os.makedirs(os.path.dirname(conf.model_save_path), exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # ========== 训练循环 ==========
    best_f1 = 0
    best_epoch = 0
    history = []

    print("\n" + "=" * 70)
    print("开始蒸馏训练")
    print(f"  alpha={conf.alpha} (硬标签权重), temperature={conf.temperature}")
    print(f"  学习率: {conf.learning_rate}, batch_size: {conf.batch_size}")
    print("=" * 70)

    for epoch in range(conf.num_epochs):
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch + 1}/{conf.num_epochs}")
        print(f"{'─' * 70}")

        # 训练
        avg_loss, avg_hard, avg_soft, train_acc, train_f1 = trainer.train_epoch()

        # 学习率调度
        trainer.step_scheduler()

        # 验证
        dev_acc, dev_f1, _, _ = trainer.evaluate()

        print(f"\n  训练集 - Loss: {avg_loss:.4f} (hard: {avg_hard:.4f}, soft: {avg_soft:.4f})")
        print(f"         Acc: {train_acc:.4f}  Macro-F1: {train_f1:.4f}")
        print(f"  验证集 - Acc:  {dev_acc:.4f}  Macro-F1: {dev_f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'dev_acc': dev_acc,
            'dev_f1': dev_f1
        })

        # 保存最优模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            torch.save(student.state_dict(), conf.model_save_path)
            print(f"  ★ 最优模型已保存 (Dev F1={best_f1:.4f})")

    print(f"\n最佳模型: Epoch {best_epoch}, Dev F1={best_f1:.4f}")

    # ========== 测试集评估 ==========
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    # 加载最佳模型
    best_student = StudentBiLSTM(vocab_size)
    best_student.load_state_dict(
        torch.load(conf.model_save_path, map_location=conf.device)
    )
    best_student.to(conf.device)

    test_acc, test_f1, test_preds, test_labels = evaluate_student(
        best_student, test_loader, conf.device
    )

    print(f"\n测试集 Acc:  {test_acc:.4f}")
    print(f"测试集 Macro-F1: {test_f1:.4f}")

    print("\n分类报告:")
    report = classification_report(test_labels, test_preds, target_names=conf.class_list)
    print(report)

    # 保存测试报告
    with open("../logs/test_report_distill.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("知识蒸馏测试报告 (FinBERT -> BiLSTM+Attention)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"教师模型: FinBERT (04-bert)\n")
        f.write(f"学生模型: BiLSTM+Attention (03-bilstm)\n")
        f.write(f"教师参数量: {teacher_params:,}\n")
        f.write(f"学生参数量: {student_params:,}\n")
        f.write(f"压缩比: {student_params/teacher_params*100:.1f}%\n\n")
        f.write(f"蒸馏配置: alpha={conf.alpha}, temperature={conf.temperature}\n")
        f.write(f"学习率: {conf.learning_rate}\n")
        f.write(f"Batch Size: {conf.batch_size}\n")
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

    print(f"\n测试报告已保存: ../logs/test_report_distill.txt")
    print(f"蒸馏后模型已保存: {conf.model_save_path}")

    print("\n蒸馏训练完成！")
    return best_student


if __name__ == "__main__":
    train_distill()
