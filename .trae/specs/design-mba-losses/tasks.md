# Tasks

## 阶段一：理论与设计（须先于实现完成，作为实现的依据）

- [x] Task 1: 在 `documents/LACE改进方案深度理论分析.md` 末尾追加"第六章 MBA 损失函数族：批判性分析与新设计"
  - [ ] SubTask 1.1: 撰写 §6.1 对 LACE-Multi 的批判性分析（证明 $h(P_t)$ 在 $P_t=e^{-2}$ 处取极大、最难点放大回退；证明乘以无界 $L_{CE}$ 在 $P_t\to0$ 梯度爆炸 → 噪声敏感）
  - [ ] SubTask 1.2: 撰写 §6.2 对 f-Multi 的批判性分析（指出原推导隐含共线假设；用 f-softargmax Jacobian 给出严格梯度，说明 $g=1+\sigma h_\alpha$ 仅在 $\alpha\to0$ 成立）
  - [ ] SubTask 1.3: 撰写 §6.3 对训练状态耦合的批判性分析（证明简化版 $\lambda=\sigma(a\bar P_t+b)$ 在正常训练下等价单调调度；指出退化解风险）
  - [ ] SubTask 1.4: 撰写 §6.4 MBA 族统一框架：理性门 $\phi_\gamma$、温度化 $\tau_\delta$、统一梯度放大因子 $g=1+\sigma\psi$、单调性定理、有界性定理、Bayes 一致性草图
  - [ ] SubTask 1.5: 撰写 §6.5 MBA-CE 特化推导（$\psi(P_t)$ 显式形式、严格单调证明、退化到 LACE-Multi 的论证）
  - [ ] SubTask 1.6: 撰写 §6.6 MBA-f 特化推导（基于 f-softargmax Jacobian 的严格梯度、D3 充要条件、$\alpha\to0$ 退化）
  - [ ] SubTask 1.7: 撰写 §6.7 MBA-PS 特化推导（主动+反应双信号、非退化解保证、退化到 MBA-CE）
  - [ ] SubTask 1.8: 撰写 §6.8 D1–D6 解决情况对照表与三方案退化/推广关系总结

## 阶段二：代码骨架与基线

- [x] Task 2: 创建清晰的代码目录结构（`src/methods`、`src/models/cnn`、`src/models/vit`、`src/datasets`、`src/configs`、`src/utils`、`src/train.py`、`src/evaluate.py`、`requirements.txt`、`README.md`）
  - [x] SubTask 2.1: 创建目录与空 `__init__.py`、`README.md` 说明结构
  - [x] SubTask 2.2: 编写 `requirements.txt`（torch、torchvision、numpy、tqdm、pyyaml 等）
- [x] Task 3: 实现数据集加载器 `src/datasets/cifar.py`（CIFAR-10/100，标准增强，可配置长尾/噪声扩展占位）
  - [x] SubTask 3.1: CIFAR-10/100 训练/测试 DataLoader，含 Normalize 与 Cutout/RandAugment
- [x] Task 4: 实现网络模型
  - [x] SubTask 4.1: `src/models/cnn/resnet.py` 实现 ResNet-56（适配 32×32 CIFAR 输入，参考 He et al. CIFAR 版本）
  - [x] SubTask 4.2: `src/models/vit/vit.py` 实现小 ViT（patch=4，depth=6/7，适配 CIFAR 分辨率）
- [x] Task 5: 实现基线损失 `src/methods/baselines.py`（CE、Focal Loss、PolyLoss）
- [x] Task 6: 实现原方案损失 `src/methods/lace_variants.py`（LACE-Multi、f-Multi，含 α-散度实现）
- [x] Task 7: 实现三个新 MBA 损失 `src/methods/mba.py`（MBA-CE、MBA-f、MBA-PS）与统一损失注册表 `src/methods/__init__.py`
  - [x] SubTask 7.1: 实现 MBACE 损失（理性门 + 温度化 CE，含 gamma_learnable、eps_init 参数）
  - [x] SubTask 7.2: 实现 MBAF 损失（理性门 + alpha-divergence NLL 形式，含 alpha 可配置）
  - [x] SubTask 7.3: 实现 MBAPS 损失（主动余弦调度 + 反应式 batch 方差 + 类别感知 a/b/c 参数）
  - [x] SubTask 7.4: 更新 `src/methods/__init__.py` 注册全部 8 种损失（ce, focal, poly, lace_multi, f_multi, mba_ce, mba_f, mba_ps）
  - [x] SubTask 7.5: 运行退化关系验证脚本（MBACE vs LACE-Multi, MBAF alpha=0 vs MBACE, MBAPS 结构验证）
  - [x] SubTask 7.6: 运行注册表验证脚本（build_loss / get_loss_names / 大小写不敏感 / 未知错误路径）

## 阶段三：训练与评估流水线

- [x] Task 8: 实现配置驱动训练流水线 `src/train.py`（YAML 配置，支持损失×模型×数据集组合，记录 Top-1/ECE/参数轨迹）
  - [x] SubTask 8.1: 配置加载、模型/损失/数据集构建、SGD/Adam 优化、学习率调度
  - [x] SubTask 8.2: 训练循环 + 验证（Top-1、ECE）、可学习参数轨迹记录、checkpoint
- [x] Task 9: 实现评估脚本 `src/evaluate.py`（加载 checkpoint，输出准确率/ECE/混淆矩阵）
- [x] Task 10: 编写 `src/configs/` 下的实验配置（CIFAR-10/100 × {ResNet-56, ViT-S} × 8 损失 × 3 种子），含 `defaults.yaml` 与各组合配置

## 阶段四：实验执行与结果汇总

- [ ] Task 11: 在 CIFAR-10 + ResNet-56 上运行 8 损失 × 3 种子，汇总 Top-1/ECE 表
- [ ] Task 12: 在 CIFAR-10 + ViT-S 上运行 8 损失 × 3 种子，汇总表
- [ ] Task 13: 在 CIFAR-100 + ResNet-56 上运行 8 损失 × 3 种子，汇总表
- [ ] Task 14: 在 CIFAR-100 + ViT-S 上运行 8 损失 × 3 种子，汇总表
- [ ] Task 15: 汇总四张表为最终结果表，绘制可学习参数训练轨迹图（`src/utils/plot.py`）

## 阶段五：论文草稿

- [ ] Task 16: 在 `documents/paper_draft/` 撰写顶会论文草稿（NeurIPS/ICML 格式），含动机、批判性分析、MBA 方法、理论、实验、消融、结论，并整合用户综述中的相关工作

## 阶段六：验证

- [ ] Task 17: 按 `checklist.md` 逐项验证，修复不通过项

# Task Dependencies

- Task 1 须先完成（理论是实现的依据），其内 SubTasks 按序。
- Task 2 无依赖，可与 Task 1 并行。
- Task 3、4 依赖 Task 2。
- Task 5、6、7 依赖 Task 2（损失实现互不依赖，可并行）。
- Task 8 依赖 Task 3、4、5、6、7。
- Task 9 依赖 Task 8。
- Task 10 依赖 Task 8。
- Task 11–14 依赖 Task 8、10；彼此可并行（若资源允许）。
- Task 15 依赖 Task 11–14。
- Task 16 依赖 Task 15 与 Task 1（理论）。
- Task 17 依赖全部前序任务。
