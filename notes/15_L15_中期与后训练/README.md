<div class="titlepage">

<img src="assets/cover.jpg" alt="image" />

<div class="tcolorbox">

**频道**：Stanford Online**发布日期**：2026 年 5 月 27 日

**时长**：1:19:54**链接**：[`https://youtu.be/2oH6PWPrYFo`](https://youtu.be/2oH6PWPrYFo)

</div>

</div>

# 动机：预训练能力为何还不等于可控助手

自回归预训练优化“给定前缀，下一 token 是什么”，却没有显式告诉模型要遵守指令、用何种风格回答、何时拒绝。后训练把潜在能力整理成可由自然语言调用的行为接口。

<figure data-latex-placement="H">
<img src="assets/instruction_following.jpg" />
<figcaption>复杂绘图指令被分解并执行，说明指令跟随本身是一种强控制机制。</figcaption>
</figure>

<div class="importantbox">

本讲主线先用监督微调（SFT）模仿理想响应，再从偏好数据学习“哪个回答更好”；中期训练则在预训练末段改变数据配方，为后训练准备能力与分布。

</div>

## 本章小结

后训练不是给模型凭空注入全部知识，而是选择、组合并校准预训练已形成的行为。

# 证据边界：现代后训练配方并不透明

公开论文常只给出框架；真实数据来源、标注规范、过滤器与配比通常缺失。旧论文反而可能公开更完整的标注说明。因此复现时必须区分“论文明确披露”“开源实现推断”和“行业传闻”。

<figure data-latex-placement="H">
<img src="assets/posttraining_sparse.jpg" />
<figcaption>课程提醒：现代模型的后训练细节与数据经常不公开。</figcaption>
</figure>

<div class="warningbox">

不要把模型卡当完整配方只知道数据集名称无法复现结果；还需 chat template、损失掩码、采样权重、去重、过滤、优化器与参考模型版本。

</div>

## 本章小结

严谨的课程笔记应记录证据等级，避免把猜测写成确定事实。

# SFT 数据的演进与共同结构

开放 SFT 从 FLAN、Self-Instruct 发展到 Alpaca、ShareGPT/Vicuna、OpenAssistant、WizardLM、Tulu 与 Nemotron。名称不同，训练记录通常都能规约为“上下文/指令 $`x`$ 与目标响应 $`y`$”。

<figure data-latex-placement="H">
<img src="assets/sft_progression.jpg" />
<figcaption>开放世界中 SFT 数据从任务模板、自动生成到真实对话和复杂后训练配方的演进。</figcaption>
</figure>

<div class="knowledgebox">

三类来源人工编写的任务与答案质量高但昂贵；真实对话自然但许可与隐私复杂；教师模型生成可扩展，但会继承教师风格与错误。

</div>

## 本章小结

数据集的关键差异不只在规模，更在任务覆盖、响应来源、风格、许可和验证方式。

# FLAN：任务模板与混合的力量

FLAN 把大量 NLP 数据集转成自然语言指令，并用多个模板表述同一任务。它证明了任务混合与措辞多样性可以显著增强零样本迁移，但随机样例也暴露出输入长短和输出格式差异巨大。

<figure data-latex-placement="H">
<img src="assets/flan_examples.jpg" />
<figcaption>FLAN 随机样例：分类、摘要与结构化信息抽取可共享统一的文本接口。</figcaption>
</figure>

<div class="warningbox">

模板泄漏同一基准的训练模板与测试模板过近，会把“遵循新指令”混同于“识别熟悉模板”；划分应按原任务族和来源去重。

</div>

## 本章小结

模板把异构任务统一起来，但模板多样性与跨划分去重决定泛化是否真实。

# Alpaca：蒸馏式指令数据的简洁配方

Alpaca 用强模型根据种子任务扩写指令—响应对，使小团队也能构造大规模 SFT 数据。其优势是便宜、格式统一，代价是响应同质化、事实错误和教师偏差。

<figure data-latex-placement="H">
<img src="assets/alpaca_examples.jpg" />
<figcaption>Alpaca 随机样例：健康建议、概念解释与代码任务被整理为统一指令—回答对。</figcaption>
</figure>

```
record = {
    "messages": [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ],
    "source": "dataset/version",
    "teacher": teacher_id,
    "license": license_id,
}
```

## 本章小结

合成指令数据应携带教师、提示词、采样参数和验证结果，不能只留下最终文本。

# SFT 目标：只在目标响应上计算交叉熵

令输入/上下文为 $`x`$，目标响应 token 为 $`y_1,\ldots,y_T`$，监督微调损失为
```math
\mathcal L_{\mathrm{SFT}}(\theta)=-\sum_{t=1}^{T}m_t\log \pi_\theta(y_t\mid x,y_{<t}).
```
$`\theta`$ 是模型参数；$`\pi_\theta`$ 是条件 token 分布；$`y_{<t}`$ 是此前目标 token；$`m_t\in\{0,1\}`$ 是损失掩码。通常用户和 system token 的 $`m_t=0`$，assistant 目标 token 的 $`m_t=1`$。

```
logits = model(input_ids).logits[:, :-1]
targets = input_ids[:, 1:]
loss = cross_entropy(logits.transpose(1, 2), targets,
                     reduction="none")
loss = (loss * assistant_mask[:, 1:]).sum()
loss /= assistant_mask[:, 1:].sum().clamp_min(1)
```

<div class="warningbox">

常见错误错误的 chat template 或边界 token 会让掩码错位；训练前应反解若干样本，逐 token 检查角色、特殊符号和 mask。

</div>

## 本章小结

SFT 在数学上仍是最大似然，但“对哪些 token 求似然”决定模型学习谁的行为。

# 数据风格会进入模型，而基准未必能测出

不同 SFT 数据在礼貌程度、拒答、列表偏好、长度和语气上差异明显。传统准确率基准可能几乎不变，但用户偏好会显著变化。

<figure data-latex-placement="H">
<img src="assets/benchmark_sensitivity.jpg" />
<figcaption>多种指令微调数据在不同基准上的表现不一致，平均分掩盖了任务间差异。</figcaption>
</figure>

<div class="importantbox">

双轨评估同时报告能力基准与行为评估：事实/推理是否正确，以及风格、帮助性、安全性和长度是否符合目标。

</div>

## 本章小结

单一平均分不能回答“数据是否更好”；评价维度必须覆盖产品真正关心的行为。

# 知识抽取、知识注入与幻觉

SFT 擅长让模型调用已经学会的知识，却不一定适合可靠注入新事实。若目标事实超出模型已有能力，行为克隆可能先提高训练集拟合，随后损害泛化并增加幻觉。

<figure data-latex-placement="H">
<img src="assets/knowledge_alignment.jpg" />
<figcaption>知识抽取与对齐：训练已知与未知事实会呈现不同学习轨迹。</figcaption>
</figure>

<div class="knowledgebox">

实践判断若目标是“教会回答格式与检索策略”，SFT 很合适；若目标是大量新知识，应优先考虑预训练/中期训练、检索增强，或可更新的外部工具。

</div>

## 本章小结

把行为对齐与事实记忆分开设计，能减少把流畅模仿误当成可靠知识获得。

# 安全 SFT：从真实交互提取场景

安全数据的难点不是写几句拒答，而是覆盖真实用户如何表达恶意、误导、诈骗、危险操作和边界请求。WildChat 等真实对话可帮助发现长尾场景，再由规则与人工审查转成训练任务。

<figure data-latex-placement="H">
<img src="assets/safety_scenarios.jpg" />
<figcaption>从真实用户对话中抽取安全场景、类别和对抗性提示。</figcaption>
</figure>

<div class="warningbox">

隐私与二次伤害真实对话必须做个人信息去除、许可检查和危险内容访问控制；审查流程也要保护标注者。

</div>

## 本章小结

安全覆盖来自系统化场景发现，不来自单一“拒绝模板”。

# 少量高针对性安全数据也能显著改变行为

课程给出的实验显示，约数百条 Alpaca 风格安全样例即可明显改善多类安全指标。这说明模型往往已经具备相关能力，只需少量数据指定决策边界和表达方式。

<figure data-latex-placement="H">
<img src="assets/safety_small_data.jpg" />
<figcaption>约 500 条安全样例即可在多个数据集上显著降低不安全得分。</figcaption>
</figure>

<div class="importantbox">

不是越多越好安全样例比例过高可能造成过度拒答；应联合测量攻击成功率、正常请求帮助率与拒答校准。

</div>

## 本章小结

高杠杆数据的价值来自边界覆盖与标签一致，而不只是样本数量。

# Tulu 3：把多来源 SFT 组织为可审计管线

Tulu 3 展示了开放后训练中较完整的配方：通用指令、数学/代码、工具、对话与安全数据分别构建，再经筛选、格式化与配比进入训练。

<figure data-latex-placement="H">
<img src="assets/tulu_pipeline.jpg" />
<figcaption>Tulu 3 的公开数据表给出安全与非服从数据的来源及规模。</figcaption>
</figure>

```
for step in range(num_steps):
    source = rng.choice(sources, p=mixture_weights)
    batch = loaders[source].next()
    loss = sft_step(batch)
    audit.log(step=step, source=source,
              tokens=batch.assistant_tokens, loss=loss)
```

## 本章小结

开放配方的贡献是让数据来源、规模和阶段可追踪，而不是只发布一个合并后的 JSON。

# 把 SFT 数据配方放在一起

课程归纳出三点：SFT 最擅长抽取预训练已有行为；即使事实正确的数据也可能伤害模型；安全、指令跟随和风格等正确行为的小样本能带来大变化，但总体仍受长期数据收益影响。

<figure data-latex-placement="H">
<img src="assets/sft_takeaways.jpg" />
<figcaption>SFT 数据部分的三条结论：能力抽取、事实数据风险与小量行为数据的高杠杆效应。</figcaption>
</figure>

<div class="knowledgebox">

配方单位建议按“能力簇”管理数据：基础对话、知识问答、推理、代码、工具、安全与风格；每簇记录独立规模、采样率、epoch 和评估集。

</div>

## 本章小结

SFT 配方是一组目标行为的预算分配，不是把所有公开指令集简单拼接。

# 中期训练与两阶段训练

“midtraining”或“两阶段训练”在预训练后段更换数据分布：稳定阶段使用宽泛混合，衰减阶段提高高质量、代码、数学或指令型数据占比，并配合学习率衰减。

<figure data-latex-placement="H">
<img src="assets/two_phase_training.jpg" />
<figcaption>稳定阶段与衰减阶段的数据混合饼图展示了两阶段训练的配方变化。</figcaption>
</figure>

若第 $`k`$ 阶段数据分布为 $`p_k(d)`$、学习率为 $`\eta_k`$，更新可写为
```math
\theta_{t+1}=\theta_t-\eta_k\,\mathbb E_{d\sim p_k}[\nabla_\theta\ell(d;\theta_t)].
```
$`d`$ 是训练样本，$`p_k`$ 是阶段混合，$`\ell`$ 是 token 损失。数据分布与学习率同时改变，消融时必须分别控制。

## 本章小结

中期训练连接“广泛世界建模”与“目标能力塑形”，它仍是 token 级预训练而非偏好优化。

# RLHF 的三步标准流程

典型流程是：先收集示范做 SFT；再对同一提示的多个回答做成对排序并训练奖励模型；最后用强化学习优化策略，同时约束其不要偏离参考策略。

<figure data-latex-placement="H">
<img src="assets/rlhf_three_steps.jpg" />
<figcaption>InstructGPT 风格流程：示范、比较、奖励建模与策略优化。</figcaption>
</figure>

<div class="importantbox">

数据与算法不可分偏好数据决定奖励信号，优化算法决定模型能多激进地追逐它；高质量比较标签也不能抵消失控的过优化。

</div>

## 本章小结

RLHF 的核心资产不只是算法，还包括提示分布、候选生成策略、标注规范与人群。

# 从模仿到优化：生成—验证鸿沟

SFT 拟合参考答案分布 $`p^*(y\mid x)`$；RLHF 则寻找使可测奖励最大的策略。人们未必能直接写出最佳答案，却常能在候选中判断哪个更好，这就是生成—验证鸿沟。

<figure data-latex-placement="H">
<img src="assets/imitation_optimization.jpg" />
<figcaption>模仿拟合参考分布，优化则直接提高可测奖励。</figcaption>
</figure>

<figure data-latex-placement="H">
<img src="assets/generation_verification_gap.jpg" />
<figcaption>不同标注者对摘要风格偏好差异明显，说明验证也并非绝对客观。</figcaption>
</figure>

## 本章小结

偏好比较通常比从零写答案容易，但“容易比较”不等于“没有主观性”。

# RLHF 数据：标注者、规范与报酬都是模型的一部分

偏好标签由具体人群在具体指南与工资条件下产生。专家标注成本高，但复杂领域的通才标签可能不可靠；低报酬又会诱导匆忙判断。

<figure data-latex-placement="H">
<img src="assets/expert_compensation.jpg" />
<figcaption>专家标注与通用标注报酬差异巨大，直接影响可扩展性和标签质量。</figcaption>
</figure>

<div class="knowledgebox">

最小标注元数据匿名标注者组别、资格、指南版本、任务耗时、候选顺序随机化、平局/跳过、复核结果与报酬区间。

</div>

## 本章小结

标注平台不是中性传感器；人员选择与激励塑造了奖励函数。

# 人口统计与风格偏好：平均标签不等于普遍价值

不同宗教、文化、年龄和教育背景可能对同一回答给出不同判断。把标签平均成一个奖励分数，会把群体差异压缩成单一“标准用户”。

<figure data-latex-placement="H">
<img src="assets/annotator_demographics.jpg" />
<figcaption>标注者人口统计分布变化会显著改变不同模型行为的评分。</figcaption>
</figure>

<div class="warningbox">

多数偏好并非安全真理应分组报告一致率，保留分歧与平局；对安全和公平等高风险决策，需由政策约束而非简单多数投票决定。

</div>

## 本章小结

RLHF 对齐的是被采样、被表达和被聚合的偏好，而不是抽象的“人类价值”。

# 模型生成反馈与自训练

AI feedback 可生成候选、批注偏好或充当裁判，显著降低成本；UltraFeedback、Zephyr、Tulu 等配方体现了这一趋势。但同一家族模型生成、筛选并评估会形成闭环偏差。

<figure data-latex-placement="H">
<img src="assets/lm_generated_feedback.jpg" />
<figcaption>模型生成偏好反馈：候选生成、偏好标注与训练形成自动化管线。</figcaption>
</figure>

```
candidates = policy.sample(prompt, n=K)
ranking = judge.rank(prompt, candidates)
if verifier.agrees(ranking) and not leakage(prompt):
    preference_store.add(prompt, ranking, judge.version)
else:
    human_queue.add(prompt, candidates)
```

## 本章小结

自训练的瓶颈从“生成数据”转为“获得独立、校准且不循环自证的验证信号”。

# KL 正则化的偏好优化目标

给定提示分布 $`x\sim\mathcal D`$，策略 $`\pi_\theta`$ 的经典目标是
```math
\max_\theta\;\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot|x)}[r_\phi(x,y)]
-\beta\,\mathbb E_{x\sim\mathcal D}\!\left[\mathrm{KL}\!\left(\pi_\theta(\cdot|x)\Vert\pi_{\rm ref}(\cdot|x)\right)\right].
```
$`r_\phi`$ 是奖励模型；$`\pi_{\rm ref}`$ 是通常由 SFT 得到的参考策略；$`\beta>0`$ 控制保守程度；KL 项惩罚策略偏离参考分布。

<div class="importantbox">

解释第一项鼓励高奖励，第二项防止模型利用奖励模型漏洞并保持语言质量。$`\beta`$ 太大几乎不学习，太小则易奖励黑客和模式坍塌。

</div>

## 本章小结

RLHF 是带信赖区域的奖励优化，而不是无限最大化一个不完美评分器。

# PPO 在语言模型中的机制

PPO 使用旧策略采样 on-policy 序列，并限制新旧策略概率比变化。令
```math
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\rm old}}(a_t|s_t)},
```
其中 $`s_t=(x,y_{<t})`$，$`a_t=y_t`$。裁剪代理目标为
```math
L^{\rm clip}=\mathbb E_t\!\left[\min\bigl(\rho_tA_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\bigr)\right].
```
$`A_t`$ 是优势估计，$`\epsilon`$ 是裁剪半径；$`\min`$ 防止通过过大概率更新获取虚假收益。

<figure data-latex-placement="H">
<img src="assets/ppo_objective.jpg" />
<figcaption>InstructGPT 中 PPO 目标同时包含奖励、KL 约束与预训练梯度混合。</figcaption>
</figure>

```
responses = old_policy.generate(prompts)
rewards = reward_model(prompts, responses)
advantages = estimate_advantage(rewards, value_model)
ratio = exp(policy.logp(responses) - old_policy.logp(responses))
objective = minimum(ratio * advantages,
                    clip(ratio, 1-eps, 1+eps) * advantages)
loss = -objective.mean() + beta * kl_to_reference(policy)
```

## 本章小结

PPO 功能强但系统复杂：策略、旧策略、参考策略、奖励模型与价值模型必须协调。

# 为什么尝试摆脱 PPO

PPO 的 on-policy 采样成本高、实现细节多且训练不稳定。课程回顾了控制 token、只训练胜者、奖励模型筛选 best-of-$`N`$ 等简化尝试；它们部分有效，却未充分利用输家与参考策略信息。

<figure data-latex-placement="H">
<img src="assets/avoid_ppo.jpg" />
<figcaption>摆脱 PPO 的早期思路：控制 token、只学偏好回答、奖励筛选与 best-of-<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mi>N</mi><annotation encoding="application/x-tex">N</annotation></semantics></math>。</figcaption>
</figure>

<div class="warningbox">

best-of-$`N`$ 的隐性代价$`N`$ 越大越可能找到奖励模型的漏洞，同时推理成本线性增加；必须用独立评估检查真实质量。

</div>

## 本章小结

简化算法的目标不是少写代码，而是减少在线组件并保留对偏好差异的有效学习信号。

# DPO：从 KL 正则化目标消去显式奖励模型

在非参数策略假设下，KL 正则化目标的最优策略满足
```math
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\rm ref}(y|x)\exp\!\left(\frac{r(x,y)}{\beta}\right),
```
其中 $`Z(x)`$ 是对所有回答归一化的配分函数。反解奖励：
```math
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\rm ref}(y|x)}+\beta\log Z(x).
```
在同一提示的胜者与败者差分中，$`\log Z(x)`$ 抵消，于是无需显式拟合奖励模型。

<figure data-latex-placement="H">
<img src="assets/dpo_derivation.jpg" />
<figcaption>DPO 从 KL 正则化 RLHF 的闭式最优策略反解隐式奖励。</figcaption>
</figure>

## 本章小结

DPO 的关键是用策略相对参考策略的对数概率比表示奖励差。

# DPO 损失、梯度直觉与代码

对偏好对 $`(x,y_w,y_l)`$，$`y_w`$ 是胜者，$`y_l`$ 是败者，DPO 损失为
```math
\mathcal L_{\rm DPO}=-\log\sigma\!\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}-
\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}
\right]\right).
```
$`\sigma`$ 是 sigmoid；$`\beta`$ 控制相对参考策略的变化尺度。梯度提高胜者相对概率、降低败者相对概率；模型已强烈区分时，更新自动变小。

```
win_margin = policy_logp_win - ref_logp_win
lose_margin = policy_logp_lose - ref_logp_lose
logits = beta * (win_margin - lose_margin)
loss = -logsigmoid(logits).mean()
```

<div class="knowledgebox">

与朴素正/负 SFT 的区别参考策略比值校准了模型原先的偏好；sigmoid 权重又让“已经分对”的样本少更新，从而避免所有偏好对使用相同步长。

</div>

## 本章小结

DPO 像分类式 SFT，但其目标来自偏好概率模型与 KL 正则化推导。

# DPO 的迭代配方与 PPO 对比

DPO 可以离线训练，也可循环“采样候选—收集偏好—更新策略”。开放模型常组合奖励筛选、专家偏好数据、SFT 与多轮 DPO。

<figure data-latex-placement="H">
<img src="assets/dpo_iteration.jpg" />
<figcaption>开放模型中的 DPO 与专家迭代：候选生成、偏好标注、SFT 和 DPO 交替进行。</figcaption>
</figure>

<div class="center">

|          | PPO                            | DPO                        |
|:---------|:-------------------------------|:---------------------------|
| **数据** | 训练中在线采样                 | 固定或周期刷新偏好对       |
| **组件** | 策略、价值、奖励、参考、旧策略 | 策略与冻结参考策略         |
| **优点** | 可直接优化序列奖励             | 简单稳定，像监督训练       |
| **风险** | 不稳定、昂贵、奖励黑客         | 离线分布受限、偏好数据偏差 |

</div>

## 本章小结

DPO 降低系统复杂度，但没有消除偏好采集、分布漂移和过优化问题。

# 失败模式：过优化、模式坍塌与长度偏差

优化不完美奖励时，早期真实质量上升，随后模型开始利用评分器漏洞；同时输出分布可能坍塌，熵下降，表达越来越单一。长度也常成为奖励代理，导致冗长回答获得不成比例的优势。

<figure data-latex-placement="H">
<img src="assets/rlhf_failure_modes.jpg" />
<figcaption>RLHF 风险：奖励过拟合与模式坍塌可在继续优化时恶化。</figcaption>
</figure>

<div class="warningbox">

监控三条曲线同时跟踪训练奖励、独立人工/强验证器质量和到参考策略的 KL；若奖励继续上升而独立质量下降，应立即早停并审计奖励漏洞。

</div>

## 本章小结

训练目标的持续改善并不保证真实目标改善，独立验证与保守更新是必要条件。

# 可复现的中期/后训练协议

1.  冻结预训练 checkpoint、tokenizer 与 chat template；

2.  按能力簇记录 SFT 来源、许可、过滤、配比与实际 token；

3.  对用户/system/assistant token 逐项验证损失掩码；

4.  中期训练单独记录阶段边界、混合与学习率；

5.  偏好数据记录候选策略、采样参数、标注者与指南；

6.  PPO/DPO 都保留参考策略哈希与 $`\beta`$；

7.  以固定预算比较 SFT、DPO、PPO，并报告能力、安全、风格、KL 与长度；

8.  失败样例进入新的数据/评估集，但防止训练—测试污染。

```
run = dict(
  base_checkpoint="sha256:...", tokenizer="sha256:...",
  chat_template="v3", sft_manifest="sft-v7.jsonl",
  preference_manifest="pref-v4.jsonl", algorithm="dpo",
  beta=0.1, seed=2026, evaluator="independent-v2")
assert no_overlap(run["sft_manifest"], heldout_eval)
```

## 本章小结

可复现性来自 checkpoint、数据、格式、标签与算法状态的联合版本化。

# 总结与延伸

<figure data-latex-placement="H">
<img src="assets/lecture_recap.jpg" />
<figcaption>课程回顾：RLHF 数据收集困难，算法比 SFT 复杂，并需警惕奖励过优化。</figcaption>
</figure>

<div class="importantbox">

最终结论SFT 决定模型“像什么样的助手”，偏好优化决定它在候选行为中“更偏向哪一个”，中期训练则提前塑造可被后训练调用的能力。三者都首先是数据分布设计问题，其次才是算法问题。

</div>

## 延伸问题

研究可继续追问：如何用多群体偏好而非单一平均奖励；如何发现奖励黑客而非事后修补；怎样动态选择 $`\beta`$；何时 DPO 的离线偏好分布不足，需要在线策略改进；中期数据变化与后训练收益是否具有可预测的 scaling law。

## 建议练习

1.  在小模型上逐 token 验证 SFT mask；

2.  用合成偏好实现 DPO，并画胜负 margin；

3.  扫描 $`\beta`$，比较偏好胜率与 KL；

4.  构造长度偏置奖励，观察过优化；

5.  对同一偏好集按标注者群体重算结果。
