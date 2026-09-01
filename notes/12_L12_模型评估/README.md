<div class="titlepage">

**Stanford CS336 第 12 讲：模型评估\
标准中文图文课程笔记**

五道口纳什 & Codex

2026 年 8 月 30 日

<img src="assets/cover.jpg" style="width:86.0%;height:46.0%" alt="image" />

<div class="tcolorbox">

**视频作者/频道**：Stanford Online

**发布日期**：2026 年 5 月 19 日

**视频时长**：1:18:34

**视频链接**：[`https://youtu.be/JpAxdTWQJxM`](https://youtu.be/JpAxdTWQJxM)

**PDF制作Skill**：[`https://github.com/wdkns/wdkns-skills`](https://github.com/wdkns/wdkns-skills)

</div>

</div>

# 评估为何是建模的“北极星”

## 动机：训练完成不等于知道模型是否“好”

前面的课程回答了架构、优化、系统和扩展规律，但训练数据会塑造模型行为：代码、自然语言、多语种或生物序列数据，会产生不同能力。要讨论“用什么数据训练”，必须先把希望模型呈现的行为变成可以测量的目标。评估因此不是收尾环节，而是决定训练方向的北极星。

<figure data-latex-placement="H">
<img src="assets/intro_scope.jpg" style="width:88.0%;height:64.0%" />
<figcaption>本讲承接训练主线：数据塑造行为，而评估先定义我们希望得到何种行为。</figcaption>
</figure>

<div class="importantbox">

构念、操作化与指标 “智能”“有用”“安全”是抽象构念。评估的工作是把构念操作化为：输入分布、交互协议、评分规则和聚合统计量。指标值从来不是构念本身；它只是特定规则下的观测。

</div>

## 同一个“好模型”有多种定义

一个模型可以因为生成概率高、考试正确率高、人类更偏好、运行成本低、能完成工具任务或更少产生有害输出而被称为“好”。这些标准并不等价。课程用质量与运行成本的散点图说明：即使只比较“能力”，是否把推理费用纳入目标，也会改变最优模型。

<figure data-latex-placement="H">
<img src="assets/quality_cost_tradeoff.jpg" style="width:88.0%;height:64.0%" />
<figcaption>模型质量与运行成本形成二维权衡；“最好”取决于使用者选择的目标。</figcaption>
</figure>

<div class="warningbox">

Goodhart 风险 一旦单个指标成为优化目标，系统会学会利用该指标的盲点。高分可能来自真实能力，也可能来自模板、冗长回答、测试污染或评测器偏差。评估必须同时说明“测什么”与“如何可能被钻空子”。

</div>

## 一张评估设计表

建议在运行任何 benchmark 前写清楚五项：

1.  目标构念：知识、推理、偏好、工具能力、安全或真实工作产出；

2.  样本来源：静态题库、真实用户、专家任务或私有数据；

3.  协议：zero-shot/few-shot、采样温度、工具权限、token 预算；

4.  评分器：精确匹配、单元测试、人类或 LLM judge；

5.  不确定性：样本量、置信区间、评测器一致性与泄漏风险。

## 本章小结

评估不是把 prompt 喂给模型再算均值，而是把抽象目标转化为可重复实验。任何分数都必须连同数据、协议和评分器一起解释。

# 困惑度：最接近训练目标的评估

## 核心思想与公式

自回归语言模型对序列 $`x_{1:N}`$ 的平均负对数似然为
```math
\mathcal{L}_{\mathrm{NLL}}
=-\frac{1}{N}\sum_{t=1}^{N}\log p_{\theta}(x_t\mid x_{<t}),
```
困惑度定义为
```math
\operatorname{PPL}(x_{1:N})
=\exp\!\left(\mathcal{L}_{\mathrm{NLL}}\right).
```

$`N`$
被计分的 token 数量；padding 等无效位置必须排除。

$`x_t`$
第 $`t`$ 个真实 token。

$`x_{<t}`$
该 token 之前的上下文。

$`p_{\theta}`$
参数为 $`\theta`$ 的模型条件分布。

$`\log`$ 与 $`\exp`$
自然对数与指数；二者使困惑度与平均交叉熵单调等价。

直觉上，PPL 可以理解为模型在每一步面对的“有效分支数”；越低表示越能集中概率到真实续写。它连续、便宜、与训练损失一致，因此非常适合监控预训练和 scaling law。

```
import torch

def perplexity(logits, targets, mask):
    # logits: [batch, time, vocab]
    logp = logits.log_softmax(dim=-1)
    token_logp = logp.gather(-1, targets[..., None]).squeeze(-1)
    nll = -(token_logp * mask).sum() / mask.sum()
    return torch.exp(nll)
```

## 为什么它有理论底座

设真实数据分布为 $`q`$，模型分布为 $`p`$。交叉熵满足
```math
H(q,p)=H(q)+D_{\mathrm{KL}}(q\Vert p).
```

$`H(q,p)`$
用模型 $`p`$ 编码真实数据 $`q`$ 的平均代价。

$`H(q)`$
数据本身不可约的熵。

$`D_{\mathrm{KL}}(q\Vert p)`$
模型与真实分布之间的 KL 散度，恒非负。

因此在分布与模型类理想、概率归一且测试集独立时，最低困惑度在 $`p=q`$ 处取得。这解释了为什么“持续降低困惑度”是有原则的目标，而非任意排行榜。

## 从分布内到分布外

历史语言模型基准通常在固定语料的留出集上测 PPL，属于分布内评估。GPT-2 展示了另一条路线：在 WebText 上训练，再 zero-shot 测多个标准数据集。图中不同规模模型在 LAMBADA、WikiText、PTB、1BW 等数据集表现并不同步，提醒我们迁移收益依赖目标数据分布。

<figure data-latex-placement="H">
<img src="assets/gpt2_perplexity_table.jpg" style="width:88.0%;height:64.0%" />
<figcaption>GPT-2 的分布外评估：小数据集可能受益明显，较大数据集不一定同步改善。</figcaption>
</figure>

## “困惑度伪装”的任务

完形填空和多项选择常把生成问题限制到候选集合，本质仍是比较条件概率。LAMBADA 要求根据长上下文预测目标词；HellaSwag 则在多个句子续写中选概率最高者。

<figure data-latex-placement="H">
<img src="assets/lambada_cloze.jpg" style="width:88.0%;height:64.0%" />
<figcaption>LAMBADA：只对上下文之后的目标词评分，是条件困惑度的典型例子。</figcaption>
</figure>

<figure data-latex-placement="H">
<img src="assets/hellaswag_completion.jpg" style="width:88.0%;height:64.0%" />
<figcaption>HellaSwag：比较多个候选续写的条件概率。</figcaption>
</figure>

若候选答案为 $`a_i=(a_{i,1},\ldots,a_{i,L_i})`$，常用打分为
```math
s_i=\frac{1}{L_i}\sum_{j=1}^{L_i}\log p_{\theta}(a_{i,j}\mid q,a_{i,<j}),
\qquad \hat{i}=\arg\max_i s_i.
```

$`q`$
问题与提示上下文。

$`L_i`$
第 $`i`$ 个候选答案的 token 长度。

$`s_i`$
候选的平均 token 对数概率。

$`\hat{i}`$
模型最终选择的候选编号。

<div class="warningbox">

困惑度排行榜的三个陷阱 第一，模型必须给出归一化概率，不能只返回真实 token 的任意高分；第二，下游输出通常应按候选答案而非整段 prompt 计分；第三，不同 tokenizer 的 token 粒度不同，裸 PPL 不一定可直接横向比较。

</div>

<figure data-latex-placement="H">
<img src="assets/perplexity_warning_exam.jpg" style="width:88.0%;height:64.0%" />
<figcaption>课堂总结：困惑度仍很重要，但现实行为需要任务型 benchmark 补充。</figcaption>
</figure>

## 本章小结

困惑度是平滑、可扩展、与训练目标一致的信号，却不会自动区分“事实正确”“有帮助”或“安全”。它适合做底层健康指标，不能独自承担全部产品目标。

# 考试型基准：可控、清晰，也会过期

## MMLU 的设计价值

考试题让研究者控制学科与难度，通常有无歧义答案且易于自动评分。MMLU 覆盖 57 个学科，用 few-shot 提示测试 GPT-3。课程特别指出：尽管名称含 Language Understanding，它更接近知识测验。

<figure data-latex-placement="H">
<img src="assets/mmlu_prompt_results.jpg" style="width:88.0%;height:64.0%" />
<figcaption>MMLU 的 few-shot 提示、示例题和随模型规模提升的知识成绩。</figcaption>
</figure>

## 从 MMLU-Pro 到 GPQA、HLE

模型进步后，原有题库会饱和。研究者通过更难的问题、更精细的专家筛选、减少模糊题和提高抗检索性延长 benchmark 寿命。DIAMOND 的流程图展示了题目写作、两轮专家验证、非专家验证与回修；难题不是“看起来复杂”就够了，必须证明答案稳定且专业人士能达成一致。

<figure data-latex-placement="H">
<img src="assets/diamond_validation.jpg" style="width:88.0%;height:64.0%" />
<figcaption>DIAMOND 的多阶段专家验证：错误或解释不足会触发反馈与修订。</figcaption>
</figure>

Humanity’s Last Exam 汇集跨学科困难题，但它依然保留结构化评分。图中的横向对比说明，不同 benchmark 的绝对准确率不能直接当成单一能力刻度：题目分布和难度不同。

<figure data-latex-placement="H">
<img src="assets/hle_accuracy.jpg" style="width:88.0%;height:64.0%" />
<figcaption>HLE 与 GPQA、MATH、MMLU 的成绩对比；更难题库重新拉开模型差距。</figcaption>
</figure>

<div class="knowledgebox">

多项选择并不天然“简单” 选项限制的是输出空间，不是推理深度。一个四选一题仍可要求长链条推导；真正风险是随机猜测底线、选项模式、题面歧义与静态题库泄漏。

</div>

## 从准确率到置信区间

若 $`n`$ 道独立题中答对 $`k`$ 道，准确率为 $`\hat p=k/n`$。近似标准误为
```math
\operatorname{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}.
```

$`n`$
有效题目数。

$`k`$
正确题数。

$`\hat p`$
样本准确率。

$`\operatorname{SE}`$
由有限样本引起的随机不确定性近似。

接近的模型必须报告区间或 bootstrap，而不能只按小数点后两位排序。

## 本章小结

考试型基准的优势是标准化和易评分；弱点是寿命有限、容易污染，并可能把知识记忆误当成通用能力。题目生产和验证流程与模型跑分同等重要。

# 聊天评估：从人类偏好到 LLM Judge

## Pairwise 比较与 Elo

开放式回答没有唯一字符串答案。Arena 的关键设计是让真实用户对匿名模型做成对比较，再根据胜负更新排名。若模型 A、B 的 Elo 分别为 $`R_A,R_B`$，常见期望胜率为
```math
P(A>B)=\frac{1}{1+10^{(R_B-R_A)/400}}.
```

$`R_A,R_B`$
两个模型的当前 Elo 分数。

$`P(A>B)`$
在当前配对分布下 A 胜过 B 的预测概率。

$`400`$
Elo 标度常数，决定分差与赔率的映射。

<figure data-latex-placement="H">
<img src="assets/elo_formula.jpg" style="width:88.0%;height:64.0%" />
<figcaption>Arena 用匿名成对选择构造 Elo；界面允许 A 胜、B 胜、都好或都差。</figcaption>
</figure>

真实用户带来生态有效性，也带来人口偏差、刷票、语言分布变化和评判能力差异。动态加入新模型与新 prompt 是优点，但意味着排行榜不是固定试卷。

## LLM-as-a-Judge 的效率与偏差

AlpacaEval 用强模型判断候选回答对基线的胜率，速度远高于真人评审。课堂提醒：judge 可能偏好更长回答，也可能偏好与自己风格相似的回答。回归校正、多个 judge 集成、交换答案顺序和明确 rubric 都是降低偏差的方法。

<figure data-latex-placement="H">
<img src="assets/judge_bias_alpacaeval.jpg" style="width:88.0%;height:64.0%" />
<figcaption>AlpacaEval 的核心是“相对基线的胜率”，但 judge 与基线选择会进入指标。</figcaption>
</figure>

WildBench 从真实对话中采样问题，并为每个样本生成具体 checklist。相比“整体感觉是否好”，rubric 将正确性、完整性、格式和事实问题拆开，提高 judge 的任务边界清晰度。

<figure data-latex-placement="H">
<img src="assets/wildbench_rubric.jpg" style="width:88.0%;height:64.0%" />
<figcaption>WildBench 将历史、query、回答与 checklist 交给 judge，支持 pairwise 与单回答打分。</figcaption>
</figure>

```
def paired_judge(prompt, answer_a, answer_b, judges):
    votes = []
    for judge in judges:
        v1 = judge(prompt, answer_a, answer_b)  # A/B/tie
        v2 = judge(prompt, answer_b, answer_a)  # swap order
        votes.append(debias_order(v1, v2))
    return aggregate_with_uncertainty(votes)
```

<div class="importantbox">

先写 rubric，再选择 judge Judge 不是“自动真相机器”。先把要判断的维度变成可核查条件，再用人类小样本校准 judge，最后才扩大自动评分规模。

</div>

## 本章小结

聊天评估提高了现实性，却把误差从“答案键”转移到“评判者”。有效报告应公开 prompt 来源、配对策略、judge 模型、rubric、位置校正和人类一致性。

# 智能体基准：评的是模型、脚手架与环境

## 从回答问题到改变环境状态

智能体需要读代码库、调用工具、修改文件并通过测试。SWE-bench 用真实 GitHub issue 与代码库衡量补丁能否通过测试；TerminalBench 则把任务放进容器，运行 agent 后再由独立测试阶段验收。

<figure data-latex-placement="H">
<img src="assets/terminalbench.jpg" style="width:88.0%;height:64.0%" />
<figcaption>TerminalBench 把任务描述、Docker 环境、agent 执行和测试验收分离。</figcaption>
</figure>

```
def evaluate_agent(task, image, agent, budget):
    env = start_fresh_container(image)
    observation = env.reset(task)
    trace = agent.run(observation, env.tools, budget=budget)
    result = run_hidden_tests(env)
    return {
        "passed": result.passed,
        "cost": trace.token_cost,
        "steps": trace.num_steps,
        "trace": trace,
    }
```

这段机制刻意把测试隐藏在 agent 外，避免直接读答案；同时必须固定镜像、时间、token、网络与工具权限，否则不同系统的分数不可比。

## 脚手架会改变结论

Kaggle 型 ML agent 能读数据、写训练代码并提交预测。排行榜显示，即便底层 LLM 相同，不同 agent scaffold 的总成绩也可能明显不同。长轨迹中，层级委派可以让主 agent 只接收子任务结果，减少上下文污染。

<figure data-latex-placement="H">
<img src="assets/kaggle_agent_leaderboard.jpg" style="width:88.0%;height:64.0%" />
<figcaption>ML agent 排行榜同时体现基础模型与脚手架能力。</figcaption>
</figure>

<div class="warningbox">

不要把 agent 分数直接写成“模型能力” Agent 结果是基础模型、prompt、记忆、工具、并行/委派策略、预算与环境共同产生的。比较模型时必须尽量锁定其他变量；比较产品系统时则应明确是在比较完整 agent。

</div>

## 成功率与 pass@k

若单次成功概率为 $`p`$，允许独立尝试 $`k`$ 次，则至少一次成功的概率为
```math
\operatorname{pass@}k=1-(1-p)^k.
```

$`p`$
单次固定预算下的成功概率。

$`k`$
可用独立尝试次数。

$`1-(1-p)^k`$
至少一次成功的概率。

因此必须同时报告 $`k`$、每次预算和总成本，不能把大量采样后的 pass@k 与单次成功率混为一谈。

## 本章小结

智能体评估的单位是“闭环任务执行”。除了最终通过率，还应保留轨迹、成本、步数和失败类型，才能区分模型不会、脚手架迷路与环境故障。

# 纯推理基准：试图把知识记忆剥离出去

## ARC-AGI 的出发点

考试题仍依赖语言和世界知识。ARC-AGI 使用新颖的网格变换任务，目标是让人类可以解决、模型却不能靠背题。每个样例通过输入/输出对揭示潜在规则，测试时要把规则迁移到新网格。

<figure data-latex-placement="H">
<img src="assets/arc_agi.jpg" style="width:88.0%;height:64.0%" />
<figcaption>ARC-AGI-1：从少量彩色网格示例归纳变换规则。</figcaption>
</figure>

难点不只是视觉识别，而是提出假设、用多个例子排除错误规则并执行离散程序。ARC-AGI-2 进一步增加多步组合。

## “人类可解”仍是上限假设

游戏环境与 Game Boy 风格任务尝试把推理放入交互系统，但课程指出，这类基准通常仍以“人类可完成”为边界，因此无法直接衡量超人推理。图中的低分说明当前模型仍有空间，也说明环境接口、探索预算和得分设计会决定结果。

<figure data-latex-placement="H">
<img src="assets/gameboy_reasoning_limit.jpg" style="width:88.0%;height:64.0%" />
<figcaption>交互推理基准的模型得分仍很低；课堂强调它受限于人类可解任务。</figcaption>
</figure>

<div class="knowledgebox">

知识与推理很难完全分离 任何任务都需要表示、先验与接口知识。更现实的目标是减少可直接记忆的答案，并通过新生成实例、程序化验证和行为轨迹观察推理过程。

</div>

## 本章小结

纯推理基准通过独特实例和规则归纳降低记忆收益，却不能自动摆脱环境设计与人类能力边界。它是能力拼图的一块，不是“通用智能”的单一刻度。

# 安全、现实性与生态有效性

## 安全不是一个标量

HarmBench 与 AIR-Bench 把风险拆为违法活动、欺骗、网络攻击、隐私、偏见等类别。分类体系帮助构造覆盖面，但同一句请求在不同政治、法律与社会语境中可能有不同风险。

<figure data-latex-placement="H">
<img src="assets/safety_taxonomy.jpg" style="width:88.0%;height:64.0%" />
<figcaption>安全基准把有害行为拆成多个风险类别；下方开始讨论越狱攻击。</figcaption>
</figure>

拒答率本身不是充分指标：模型可能对无害请求过度拒绝，也可能在表面拒绝后泄漏操作步骤。越狱算法还能自动搜索后缀绕过安全策略。因此安全评测应同时覆盖攻击成功率、正常任务效用和情境判断。

<figure data-latex-placement="H">
<img src="assets/contextual_safety.jpg" style="width:88.0%;height:64.0%" />
<figcaption>不同模型对极端请求的回答示例，随后引出“安全高度依赖情境”。</figcaption>
</figure>

<div class="warningbox">

能力具有双重用途 网络安全 agent 既可帮助防御，也可用于攻击。评估应区分能力、意图、授权环境与现实可执行性，而不是把“会不会”直接等同于“是否安全”。

</div>

## 生态有效性：任务像真实世界吗

课程强调 realism/ecological validity：真实用户聊天比人工模板更贴近日常分布；GDPVal 采集经济职业任务；MedHELM 从临床医生收集问题，避免只用标准化考试替代临床工作。

<figure data-latex-placement="H">
<img src="assets/ecological_validity.jpg" style="width:88.0%;height:64.0%" />
<figcaption>真实职业任务与 MedHELM 临床任务：把评估从考试题推进到工作产出。</figcaption>
</figure>

<div class="importantbox">

现实性与可控性需要组合 实验室题库便于归因，真实任务更能预测部署表现。稳健评估套件应同时包含可控诊断题与生态任务，并让两者失败模式互相解释。

</div>

## 本章小结

安全与现实性都要求回到使用情境。分类覆盖、对抗测试、正常效用、真实工作流程和授权边界必须共同进入报告。

# 有效性、污染、饱和与基准失灵

## 训练污染不只是逐字重复

若测试题出现在训练语料中，成绩可能反映记忆。更隐蔽的污染包括题目改写、答案解析、同源网页或衍生数据。课程给出四条路线：使用概率异常等方法检测；鼓励报告 train-test overlap；持续生成新评测；使用私有评测。

<figure data-latex-placement="H">
<img src="assets/contamination_mitigations.jpg" style="width:88.0%;height:64.0%" />
<figcaption>污染治理路线：检测、报告规范、动态新题与私有评测。</figcaption>
</figure>

```
def audit_contamination(train_docs, eval_items, normalize, similarity):
    hits = []
    index = build_index(normalize(d) for d in train_docs)
    for item in eval_items:
        exact = index.exact(normalize(item.text))
        near = index.search(normalize(item.text), similarity)
        hits.append({"id": item.id, "exact": exact, "near": near})
    return hits
```

检测结果本身也不完美：低概率顺序可能提示记忆，但常识题也可能天然高概率。最可靠策略仍是控制数据来源、保留时间戳、使用新题与私有题交叉验证。

## 基准饱和与错误题

当模型接近满分，少量错误题、模糊题和评分 bug 会主导差异。课程展示一项审计：多个著名 benchmark 的大量“模型错误”可由题目本身的问题解释；智能体基准还可能被空输出或简单策略钻空子。

<figure data-latex-placement="H">
<img src="assets/benchmark_errors.jpg" style="width:88.0%;height:64.0%" />
<figcaption>不同 benchmark 中由题目错误引起的模型错误比例；静态分数需要题目级审计。</figcaption>
</figure>

<div class="warningbox">

排行榜差距小于题库噪声时，不要强行排序 应报告题目级复核、置信区间和多个随机种子。对 agent benchmark，还应让模型检查轨迹或由人工抽样确认“通过”是否真的完成任务。

</div>

## 指标的效度如何评估

与另一个排行榜相关并不自动证明有效，除非目标本来就是模拟那个排行榜。应分别检查：

- 构念效度：题目是否真的测目标能力；

- 内容效度：任务覆盖是否充分；

- 生态效度：是否预测真实使用；

- 评分效度：judge 或测试是否能区分好坏；

- 稳健性：提示、顺序、采样与版本变化是否改变结论。

## 本章小结

污染让模型提前见题，饱和让噪声占主导，漏洞让系统学会“过测试”。基准必须像软件一样持续维护、版本化和做回归审计。

# 把课程变成可执行评估协议

## 标准流程

1.  写目标构念与部署决策：分数将支持什么选择？

2.  设计互补套件：PPL、可控任务、真实偏好、agent、推理与安全。

3.  冻结协议：模型版本、prompt、采样、工具、预算与环境镜像。

4.  选择评分器并校准：答案键、测试、人类、LLM judge 各自做误差审计。

5.  运行统计分析：区间、分层结果、失败类型与成本。

6.  做污染与漏洞检查：近重复、时间切分、私有题、轨迹复核。

7.  记录版本：数据、代码、judge 和模型任何变化都要生成新评估版本。

<div class="importantbox">

课程最终结论 不存在唯一“真正的评估”。应按要测的构念选择评估，并明确比赛规则：比较的是方法、基础模型，还是完整 agent。困难度、现实性和效度缺一不可。

</div>

<figure data-latex-placement="H">
<img src="assets/evaluation_takeaways.jpg" style="width:88.0%;height:64.0%" />
<figcaption>本讲结论：没有一个评估统治全部目标，规则与评估对象必须写清楚。</figcaption>
</figure>

## 本章小结

成熟评估不是单个数字，而是一份可复现的证据包：任务、协议、原始输出、评分器、统计量、成本、失败案例和版本信息。

# 总结与延伸

## 整讲内容压缩

本讲从最贴近训练目标的困惑度出发，依次扩展到考试、聊天、智能体、纯推理与安全评估。每扩展一步，任务更接近真实行为，但评分也更难控制：精确答案变成人类偏好，静态生成变成长程交互，单一能力变成带情境的风险判断。

核心关系可以压缩为：
```math
\text{评估结论}
=f(\text{构念},\text{样本},\text{协议},\text{评分器},\text{统计},\text{版本}).
```

构念
想测量的能力、偏好或风险。

样本
题目或真实交互来自何种分布。

协议
提示、工具、预算和采样规则。

评分器
答案键、单元测试、人类或 LLM judge。

统计
聚合方式、不确定性和分层分析。

版本
模型、数据与评估代码的时间状态。

## 进一步推论

第一，训练与评估共同定义优化方向：数据决定模型能学什么，评估决定团队会继续强化什么。第二，越接近部署，越要接受异质性与上下文；越追求因果诊断，越需要小而可控的测试。第三，动态题、私有题和真实任务能降低污染，却牺牲完全公开复现；实践中需要公开可复现层与保密审计层并存。

## 建议延伸练习

1.  对同一模型同时测 PPL、MMLU 子集和一个真实任务，解释三者不一致的原因。

2.  为一组开放回答设计 rubric，并比较人类、单一 judge 和多 judge 集成的一致性。

3.  给 agent benchmark 加入成本与轨迹指标，分析“更高成功率”是否只是花费更多预算。

4.  对静态题库做 exact/near-duplicate 污染审计，并人工复核最高相似样本。
