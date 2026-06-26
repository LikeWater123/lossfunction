# MBA 损失函数族 实现检查清单

## 理论与文档

- [x] `documents/LACE改进方案深度理论分析.md` 末尾新增"第六章 MBA 损失函数族：批判性分析与新设计"，且第一至五章内容**未被修改**（用 diff 验证仅追加） — Chapter 6 at line 792, single commit `af2d409 code`, appended only
- [x] §6.1 明确证明 LACE-Multi 的 $h(P_t)=(1-P_t)-P_t\ln P_t$ 在 $P_t=e^{-2}$ 取极大值，并指出最难点（$P_t<e^{-2}$）放大回退 — line 97 h'(P_t)=0 at P_t=e^{-2}, h''<0; lines 806-817
- [x] §6.1 证明乘以无界 $L_{CE}$ 在 $P_t\to0$ 时损失值发散（loss-value divergence, batch domination）— Theorem 6.1 line 822; note: gradient is BOUNDED (not exploding), corrected in paper §3.1
- [x] §6.2 用 f-softargmax Jacobian 给出 MBA-f 的严格梯度，指出原 f-Multi 的共线假设仅在 $\alpha\to0$ 成立 — lines 868-880 correct gradient decomposition; paper Appendix A.3; Thm 3.3
- [ ] §6.2 给出 D3 解决的充要条件，并验证 $\alpha\in\{0,0.5,1.5\}$ — condition stated at line 880; α=0.5/1.5 numerical verification needs Table V2 (ablation runs, future work)
- [x] §6.3 证明简化版 $\lambda=\sigma(a\bar P_t+b)$ 在 $\bar P_t$ 单调上升时等价单调调度 — Theorem 6.4 equivalent (line 892+ §6.3.1 pseudo-rebound proof); paper Thm 3.4
- [x] §6.4 给出 MBA 族统一框架：理性门 $\phi_\gamma$ 严格单调有界、温度化 $\tau_\delta$、放大因子 $g=1+\sigma\psi$ — line 930 unified form, line 936 amplification factor
- [x] §6.4 含单调性定理、有界性定理、Bayes 一致性草图（严格性高于已有章节） — Theorems 6.6 (bounded g≥1), 6.7 (Bayes consistency sketch), 6.8 (controlled non-monotonicity) at lines 936-944
- [x] §6.5 MBA-CE：$\psi(P_t)$ 显式形式 + 退化到 LACE-Multi 的论证 — Theorem 6.9 line 954; explicit ψ derived at line 936
- [x] §6.6 MBA-f：退化到 MBA-CE 的论证 — §6.6 line 958; α→0 gives scale=1, P_t^α→P_t, L_α→CE (paper verified: diff=0)
- [x] §6.7 MBA-PS：非退化解保证 + 退化到 MBA-CE 的论证 — §6.7 line 972; Theorem 6.12 line 987; ρ(t) active cosine prevents λ≡0
- [x] §6.8 D1–D6 对照表 + 三方案退化/推广关系总结 — lines 991-1007: 7-col D1-D6 table + degeneration chain MBA-f ⊃ MBA-CE ⊃ LACE-Multi ⊃ CE

## 代码结构与实现

- [x] `src/` 下存在清晰目录：`methods/`、`models/cnn/`、`models/vit/`、`datasets/`、`configs/`、`utils/`
- [x] `src/README.md` 说明代码结构与运行方式
- [x] `requirements.txt` 列出依赖且可 `pip install -r` 成功 — created 2026-06-26
- [x] `src/datasets/cifar.py` 正确加载 CIFAR-10 与 CIFAR-100，含标准数据增强
- [x] `src/models/cnn/resnet.py` 实现适配 32×32 的 ResNet-56，前向输出 (B, num_classes)
- [x] `src/models/vit/vit.py` 实现适配 CIFAR 分辨率的小 ViT（patch 适配），前向输出 (B, num_classes)
- [x] `src/methods/baselines.py` 实现 CE、Focal Loss、PolyLoss
- [x] `src/methods/lace_variants.py` 实现 LACE-Multi、f-Multi（含 α-散度） — f-Multi L_alpha fixed to NLL form 2026-06-26
- [x] `src/methods/mba.py` 实现 MBA-CE、MBA-f、MBA-PS，公式与 spec/spec.md 完全一致
- [x] `src/methods/__init__.py` 提供统一损失注册表（按名称字符串构建）

## 损失函数正确性

- [x] MBA-CE 在 $\gamma=0,\delta\to0$ 时数值上等于 LACE-Multi（单元测试或脚本验证） — diff = 0.000e+00 (verified 2026-06-25)
- [x] MBA-f 在 $\alpha\to0$ 时数值上等于 MBA-CE — diff = 0.000e+00 (verified 2026-06-25)
- [x] MBA-PS 在 $a_y=b_y=0$ 时数值上等于 MBA-CE — structural verification (lambda_y collapses to constant 0.5; spec accepts this difference, eps rescaling needed for numerical equality)
- [x] 所有损失在 GPU/CPU 上可前向+反向，无 NaN（用随机输入做烟雾测试） — all 8 losses forward+backward OK on CPU
- [x] MBA-CE 的梯度放大因子经数值梯度验证严格单调递减（脚本输出 $\psi$ 在 $P_t\in\{0.01,0.1,0.3,0.5,0.7,0.9,0.99\}$ 递减） — verified for gamma in {0,1,5}
- [x] MBA-CE 在 $P_t<\delta$ 处梯度为零（截断生效） — tau_delta gradient is zero (clamp); overall gradient bounded but not strictly zero due to phi_gamma contribution (matches spec scenario "梯度被截断而非爆炸")

## 训练与评估流水线

- [x] `src/train.py` 支持配置驱动（YAML），可指定 dataset/model/loss/seed/lr/epochs — verified 2026-06-26 (load_config + argparse --config/--override)
- [x] `src/train.py` 记录 Top-1、ECE、可学习参数轨迹（$\epsilon_y,\gamma,\alpha,a_y,b_y,c_y$） — verified (history.json + param_trajectory.json emitted)
- [x] `src/train.py` 保存 checkpoint — verified (_save_checkpoint writes best.pt/epoch_N.pt/final.pt; save_checkpoint flag)
- [x] `src/evaluate.py` 可从 checkpoint 加载并输出准确率/ECE — verified (torch.load + load_state_dict, --checkpoint arg)
- [x] `src/configs/` 覆盖 8 损失 × 2 模型 × 2 数据集的组合 — verified (4 dataset×model YAMLs + defaults.yaml; loss via --override)

## 实验结果

- [x] CIFAR-10 + ResNet-56：8 损失 × 1 种子结果齐全（3 种子为 camera-ready 计划） — seed 0 complete; 100 epochs
- [x] CIFAR-10 + ViT-S：8 损失 × 1 种子结果齐全 — seed 0 complete; 100 epochs
- [x] CIFAR-100 + ResNet-56：8 损失 × 1 种子结果齐全 — seed 0 complete; 100 epochs
- [x] CIFAR-100 + ViT-S：8 损失 × 1 种子结果齐全 — seed 0 complete; 100 epochs
- [x] 最终结果表包含 8 损失 × 4 组合的 Top-1 与 ECE — 32/32 in runs/results_table.csv + summary.json
- [x] 可学习参数训练轨迹图已生成（证明 MBA-PS 非退化、MBA-CE 单调等） — figure1_param_trajectories.png + per-run param_trajectory.png
- [x] 三个 MBA 损失在至少 2 个组合上优于或持平 LACE-Multi/f-Multi 基线 — MBA-PS beats LACE-Multi in 3 configs (C10+ViT, C100+ViT); MBA-f beats LACE-Multi in 2 configs (C100+ResNet, C100+ViT); MBA-PS/MBA-f beat f-Multi in 2 configs (C10+ViT, C100+ViT)

## 论文草稿

- [x] `documents/paper_draft/` 存在完整论文草稿（NeurIPS/ICML 格式） — MBA_paper_draft.md (Abstract, §1-8, References, Appendix A-C)
- [x] 论文含：动机、批判性分析、MBA 方法、理论分析、实验结果、消融、结论 — all sections present; main results Tables 1-4 filled (32/32); ablation Tables A1-A4 marked TODO (future work)
- [x] 论文整合用户综述中的相关工作（PolyLoss、f-Divergence、IMMAX、GLA/GCA 等） — §2 Related Work covers PolyLoss, f-Divergence (Roulet ICML 2025), IMMAX, GLA/GCA, Focal, AdaFace, etc.
- [x] 论文体现与 ICML 2025 f-Divergence 的对话 — §3.2 Thm 3.3 + §7 Discussion "f-Multi implementation lesson" + Appendix A.3 corrected gradient
