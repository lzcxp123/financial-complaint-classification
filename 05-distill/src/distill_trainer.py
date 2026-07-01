import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class DistillTrainer:
    """知识蒸馏训练器

    蒸馏架构：
    - 教师模型（Teacher）：FinBERT（04-bert训练），已训练好，全程冻结
    - 学生模型（Student）：BiLSTM+Attention（03-bilstm），需要蒸馏训练
    - 损失函数 = alpha * CE(学生输出, 硬标签) + (1-alpha) * KL(学生输出/T, 教师输出/T)
    """

    def __init__(self, teacher, student, train_loader, val_loader, config):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # 第一步：冻结教师模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # 第二步：初始化损失函数
        self.hard_criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

        # 第三步：初始化优化器
        self.optimizer = torch.optim.Adam(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.5
        )

    def train_epoch(self):
        """训练一个epoch"""
        self.student.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        all_preds = []
        all_labels = []

        train_bar = tqdm(self.train_loader, desc="  [Distill Train]")

        for batch in train_bar:
            teacher_input_ids, teacher_attention_mask, student_input_ids, labels = batch
            teacher_input_ids = teacher_input_ids.to(self.device)
            teacher_attention_mask = teacher_attention_mask.to(self.device)
            student_input_ids = student_input_ids.to(self.device)
            labels = labels.to(self.device)

            # ========== 教师模型前向（生成软标签）==========
            with torch.no_grad():
                teacher_logits = self.teacher(teacher_input_ids, teacher_attention_mask)
                # 使用温度T进行平滑
                teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=1)

            # ========== 学生模型前向 ==========
            student_logits = self.student(student_input_ids)

            # ========== 计算损失 ==========
            # 硬标签损失（标准CrossEntropy）
            hard_loss = self.hard_criterion(student_logits, labels)

            # 软标签损失（KL散度）
            student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=1)
            # KL散度需要乘以 T^2 进行缩放
            soft_loss = self.kl_criterion(student_log_probs, teacher_probs) * (self.config.temperature ** 2)

            # 总损失
            loss = self.config.alpha * hard_loss + (1 - self.config.alpha) * soft_loss

            # ========== 反向传播 ==========
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()

            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}',
                'soft': f'{soft_loss.item():.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_hard = total_hard_loss / len(self.train_loader)
        avg_soft = total_soft_loss / len(self.train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="macro")

        return avg_loss, avg_hard, avg_soft, train_acc, train_f1

    def evaluate(self):
        """在验证集上评估学生模型"""
        self.student.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                _, _, student_input_ids, labels = batch
                student_input_ids = student_input_ids.to(self.device)
                labels = labels.to(self.device)

                outputs = self.student(student_input_ids)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        return acc, f1, all_preds, all_labels

    def step_scheduler(self):
        """更新学习率"""
        self.scheduler.step()

    def get_student(self):
        """获取学生模型"""
        return self.student


if __name__ == "__main__":
    print("蒸馏训练器测试...")
    print("蒸馏损失公式:")
    print("  L = alpha * CE(student_logits, labels)")
    print("      + (1-alpha) * KL(log_softmax(student/T), softmax(teacher/T)) * T^2")
    print(f"\n当前配置: alpha=0.3, temperature=4")
    print("  alpha=0.3 表示 30%硬标签 + 70%软标签")
    print("  教师模型: FinBERT (04-bert)")
    print("  学生模型: BiLSTM+Attention (03-bilstm)")
