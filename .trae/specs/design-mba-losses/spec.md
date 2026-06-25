# MBA 损失函数族设计与实现 Spec

## Why

用户在 `LACE改进方案深度理论分析.md` 中提出了三个改进损失函数（LACE-Multi、f-Multi、训练状态耦合），但在第五章的理论分析中存在三个**未被察觉的理论缺口**，会直接影响发顶会的可信度与实验有效性：

1. **LACE-Multi 的放大因子 $h(P_t)=(1-P_t)-P_t\ln P_t$ 非单调**：在 $P_t=e^{-2}\approx0.135$ 处取极大值，意味着**最难的样本（$P_t<0.135$）反而比中等难样本获得更小的梯度放大**（例如 $h(0.01)=1.056 < h(0.1)=1.330$）。文档表格中已暴露这一点但未予修正。同时，乘以无界的 $L_{CE}$ 在 $P_t\to0$（尤其是噪声标签）处会导致梯度爆炸，噪声鲁棒性仅靠 $\epsilon_y$ 单调衰减来缓解，过于粗糙。
2. **f-Multi 的梯度推导不严谨**：声称 $g(P_t^\alpha)=1+\sigma(\epsilon_y)h_\alpha(P_t^\alpha)$ 且 "$h_\alpha$ 性质与 $h$ 相同，故 f-Multi 同样解决 D3"。但该推导隐含假设 $\nabla_\theta L_\alpha$ 与 $\nabla_\theta P_t^\alpha$ 共线（如同 CE 中 $\nabla_\theta P_t = -P_t\nabla_\theta L_{CE}$），这对一般的 f-softargmax 输出概率**不成立**。因此"D3 已解决"的结论缺乏严格依据，是审稿人容易击穿的理论漏洞。
3. **训练状态耦合的"回弹"是被动反应式的**：简化版 $\lambda=\sigma(a\bar P_t+b)$ 在正常训练（$\bar P_t$ 单调上升）下**仍等价于单调调度**，所谓"解决 D1"只有在性能退化时才触发，且存在退化解（$\lambda\to0$ 退化为 CE）风险，且基础形式缺乏类别感知。

本变更在补全上述批判性分析的同时，提出**MBA（Monotone Bounded Amplification，单调有界放大）损失函数族**——三个新损失函数，各自精确对应修复上述一个缺口，并在 CIFAR-10/100 上用经典 CNN 与 ViT 验证，最终产出顶会论文草稿。

## What Changes

- **新增理论章节**：在 `documents/LACE改进方案深度理论分析.md` 末尾追加"第六章 MBA 损失函数族：批判性分析与新设计"，包含：(1) 对三个原损失函数的批判性理论分析（指出非单调性、梯度推导漏洞、被动回弹）；(2) MBA 族统一框架的严格推导（理性门 $\phi_\gamma(P_t)=\frac{1-P_t}{1+\gamma P_t}$、温度化内损失 $\tau_\delta$、梯度放大因子的单调性定理与有界性定理、Bayes 一致性草图）；(3) 三个成员的特化推导与 D1–D6 解决情况对照表。
- **新增代码目录结构**：`src/methods/`（损失函数）、`src/models/cnn/` 与 `src/models/vit/`（网络）、`src/datasets/`（CIFAR-10/100）、`src/train.py`、`src/evaluate.py`、`src/configs/`、`src/utils/`。
- **实现基线损失**：CE、Focal Loss、PolyLoss、LACE-Multi、f-Multi（用于公平对比）。
- **实现三个新损失函数**：MBA-CE、MBA-f、MBA-PS（详见 ADDED Requirements）。
- **实现网络**：经典 CNN（ResNet-56 适配 CIFAR）与 ViT（ViT-S/16 适配 32×32）。
- **实现数据集加载器**：CIFAR-10、CIFAR-100（含标准增强与长尾/噪声可选扩展）。
- **训练与评估流水线**：统一配置驱动，支持损失×模型×数据集组合，记录 Top-1、ECE、参数轨迹。
- **实验验证**：在 CIFAR-10 与 CIFAR-100 上对全部损失 × {ResNet-56, ViT-S/16} 完成 3 次种子复现，输出准确率/校准/轨迹对比表。
- **新增顶会论文草稿**：`documents/paper_draft/` 下产出符合 NeurIPS/ICML 格式的论文草稿（动机→批判→方法→理论→实验→分析），结合用户综述与本设计。

## Impact

- **Affected specs**: 无既有 spec（首次创建）。
- **Affected documents**: `documents/LACE改进方案深度理论分析.md`（追加第六章，**不修改**已有第一至五章内容）、新增 `documents/paper_draft/`。
- **Affected code**: 新增 `src/` 整个代码树；不触碰 `documents/` 中已有分析文档（仅追加）。
- **风险**：训练 ViT 在 CIFAR 上需调超参（patch size、数据增强），需预留调参时间；f-Multi 的严格梯度推导需谨慎，避免再次出错。

## ADDED Requirements

### Requirement: MBA-CE 损失函数（修复 LACE-Multi 的非单调性与梯度爆炸）

系统 SHALL 提供名为 `MBA-CE` 的损失函数，定义为：

$$L_{\text{MBA-CE}} = \left[1 + \sigma(\epsilon_y)\,\phi_\gamma(P_t)\right]\cdot \tau_\delta(P_t)$$

其中：
- $\phi_\gamma(P_t) = \dfrac{1-P_t}{1+\gamma P_t}$，$\gamma\geq0$ 为可学习（或可配置）参数，**严格单调递减**且有界于 $[0,1]$；$\gamma=0$ 时退化为 $(1-P_t)$（LACE-Multi 门控）。
- $\tau_\delta(P_t)=-\ln(\max(P_t,\delta))$ 为温度化（截断）交叉熵，$\delta\in(0,1)$ 小常数，截断 $P_t\to0$ 的发散以提供噪声鲁棒性。
- $\epsilon_y$ 为类别感知可学习参数，sigmoid 约束于 $(0,1)$（继承 D2/D4 解决）。

**理论保证**（须在第六章证明）：
- 梯度放大因子 $g(P_t)=1+\sigma(\epsilon_y)\,\psi(P_t)$，其中 $\psi(P_t)$ **在 $(0,1)$ 上严格单调递减且非负**，从而 D3 **严格解决**（无 $e^{-2}$ 处的非单调回退）。
- 放大因子有界：$g(P_t)\le 1+\sigma(\epsilon_y)$，且因 $\tau_\delta$ 截断，难样本梯度不爆炸。
- $\gamma=0,\delta\to0$ 时严格退化为 LACE-Multi（含其非单调性），证明 MBA-CE 是 LACE-Multi 的严格推广。

#### Scenario: 严格单调放大
- **WHEN** 训练样本真实类别概率 $P_t$ 在 $(0,1)$ 内变化
- **THEN** MBA-CE 的梯度放大因子 $g(P_t)$ 随 $P_t$ 增大严格递减，即 $P_t$ 越小放大越大，不存在 $P_t<e^{-2}$ 区间的回退

#### Scenario: 噪声鲁棒
- **WHEN** 标签被污染使 $P_t\to0$
- **THEN** 因 $\tau_\delta$ 在 $P_t<\delta$ 处梯度为零，噪声样本梯度被截断而非爆炸

#### Scenario: 退化关系
- **WHEN** 令 $\gamma=0$ 且 $\delta\to0$
- **THEN** MBA-CE 退化为 LACE-Multi，放大因子恢复为 $1+\sigma(\epsilon_y)h(P_t)$（含其 $e^{-2}$ 处非单调性）

### Requirement: MBA-f 损失函数（修复 f-Multi 的梯度推导漏洞）

系统 SHALL 提供名为 `MBA-f` 的损失函数，定义为：

$$L_{\text{MBA-f}} = \left[1 + \sigma(\epsilon_y)\,\phi_\gamma(P_t^\alpha)\right]\cdot L_\alpha(\mathbf{z},y)$$

其中 $L_\alpha$ 为 Roulet et al. (ICML 2025) 的 $\alpha$-散度损失，$P_t^\alpha$ 为 f-softargmax 输出的正确类概率。

**理论保证**（须在第六章给出**正确**推导）：
- 利用 f-softargmax 的 Jacobian 结构 $J_{p^*}(\mathbf{z})$，给出 $\nabla_\theta L_{\text{MBA-f}}$ 的**严格**分解，明确指出 $\nabla_\theta P_t^\alpha$ 与 $\nabla_\theta L_\alpha$ 一般不共线，原 f-Multi 的 $g=1+\sigma(\epsilon_y)h_\alpha$ 形式仅在 $\alpha\to0$（退化为 CE）时成立。
- 给出 D3 解决的**充要条件**：当 $\phi_\gamma$ 严格递减且 $L_\alpha$ 的自梯度结构满足对齐条件时，放大因子单调递减；并验证 $\alpha\in\{0,0.5,1.5\}$ 下条件是否成立。
- $\alpha\to0$ 时严格退化为 MBA-CE。

#### Scenario: 梯度分析正确性
- **WHEN** 使用 $\alpha\neq0$ 的 f-散度损失
- **THEN** MBA-f 的梯度推导显式包含 f-softargmax Jacobian 项，不再使用 CE 风格的共线假设

#### Scenario: 退化关系
- **WHEN** 令 $\alpha\to0$
- **THEN** MBA-f 退化为 MBA-CE

### Requirement: MBA-PS 损失函数（修复训练状态耦合的被动回弹与退化解）

系统 SHALL 提供名为 `MBA-PS` 的损失函数，定义为：

$$L_{\text{MBA-PS}} = \left[1 + \sigma(\epsilon_y)\,\lambda_y(s(t))\,\phi_\gamma(P_t)\right]\cdot \tau_\delta(P_t)$$

其中：
- $\lambda_y(s(t))=\sigma\!\big(a_y\,\rho(t)+b_y\,s_{\text{react}}(t)+c_y\big)$，$(a_y,b_y,c_y)$ 为类别感知可学习参数（每类 3 个，远少于完整 MLP）。
- $\rho(t)=\tfrac12(1+\cos(\pi t/T))$ 为**主动式**余弦调度，按构造随训练进度变化，不依赖性能退化——保证 D1 被真正（而非被动）解决。
- $s_{\text{react}}(t)$ 为**反应式**信号：归一化 batch 内置信度方差 $\mathrm{Var}(P_t)$（捕捉样本分散度），使 $\lambda$ 能对训练异常做出反应。
- 主门控 $\phi_\gamma$ 与温度化 $\tau_\delta$ 沿用 MBA-CE（D3 严格、有界、噪声鲁棒）。

**理论保证**（须在第六章给出）：
- $\lambda_y$ 在训练中按 $\rho(t)+s_{\text{react}}$ 双信号驱动，**不**仅依赖单调上升的 $\bar P_t$，从而避免"伪单调"。
- 给出非退化解保证：因 $\rho(t)$ 周期性变化，$\lambda_y$ 不会坍缩到恒定 0，避免退化为 CE。
- $\lambda_y$ 退化为常数时退化为 MBA-CE。

#### Scenario: 主动调度
- **WHEN** 训练正常进行且 $\bar P_t$ 单调上升
- **THEN** 因 $\rho(t)$ 余弦变化，$\lambda_y$ 仍随训练阶段改变，不呈现单调失效

#### Scenario: 反应能力
- **WHEN** 训练中出现 batch 置信度方差突变（如学习率调整）
- **THEN** $s_{\text{react}}$ 反馈使 $\lambda_y$ 做出调整

#### Scenario: 退化关系
- **WHEN** 令 $a_y=b_y=0$（$\lambda_y$ 恒为常数）
- **THEN** MBA-PS 退化为 MBA-CE

### Requirement: 实验验证（CIFAR-10/100 × CNN × ViT）

系统 SHALL 提供可复现的实验流水线，满足：
- 在 CIFAR-10 与 CIFAR-100 上验证全部损失函数。
- 至少包含一个经典 CNN（ResNet-56，适配 32×32 输入）与一个 ViT（ViT-S/16 或等效小 ViT，patch 适配 CIFAR 分辨率）。
- 每个损失×模型×数据集组合至少 3 个随机种子。
- 评估指标：Top-1 准确率、ECE（期望校准误差）、可学习参数训练轨迹。
- 基线对比：CE、Focal Loss、PolyLoss、LACE-Multi、f-Multi，以及三个新 MBA 损失。

#### Scenario: 完整对比
- **WHEN** 运行 CIFAR-10 + ResNet-56 实验
- **THEN** 产出包含 8 种损失（5 基线+3 新）的 Top-1/ECE 对比表，含均值与标准差

#### Scenario: 架构泛化
- **WHEN** 同一损失在 ResNet-56 与 ViT-S/16 上分别训练
- **THEN** 结果可对比，验证损失函数对架构的泛化性

### Requirement: 顶会论文草稿

系统 SHALL 在 `documents/paper_draft/` 下产出符合 NeurIPS/ICML 格式的论文草稿（LaTeX 或 Markdown），包含：动机、对现有 LACE 系列与 f-Multi 的批判性分析、MBA 族方法、理论分析（单调性/有界性/一致性/退化关系）、实验结果、消融与分析。须体现发顶会的潜力（理论创新点叠加 + 与 ICML 2025 f-Divergence 对话）。

## MODIFIED Requirements

### Requirement: 理论文档扩展

`documents/LACE改进方案深度理论分析.md` SHALL 在末尾追加"第六章 MBA 损失函数族：批判性分析与新设计"，**不修改**已有第一至五章内容。第六章须包含：(1) 三段批判性分析；(2) MBA 族统一框架；(3) 三成员特化推导；(4) D1–D6 对照表；(5) 与原三方案的退化/推广关系。理论性须高于已有章节（含严格定理与证明）。

## REMOVED Requirements

无。
