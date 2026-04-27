# 学术写作工具 (Scientific Writer)

## 1. 项目简介

**学术写作工具** 是一个基于 AI 的自动化科学写作助手，定位为"深度研究与格式化写作工具"。它结合大语言模型的推理能力、结构化的学术写作流程以及外部文献检索，能够生成包含真实引用的、可直接投稿的学术论文。

项目支持两种运行模式：
- **标准模式（Standard）**：交互式对话，由 AI Agent 自主规划并执行论文撰写，生成初稿。
- **工作流模式（Workflow）**：结构化流水线，对已有初稿进行多维度量化评审与闭环修改，直至达标。

---

## 2. 使用流程总览

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  阶段1：数据准备  │ --> │  阶段2：标准模式      │ --> │  阶段3：工作流模式    │
│  放入data/文件夹  │     │  生成论文初稿         │     │  迭代评审与完善       │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

完整的学术写作流程分为三个阶段：

1. **数据准备**：将参考文献、实验数据、图片、大纲等素材放入 `data/` 目录。
2. **标准模式**：通过与 Agent 交互，完成文献检索、逻辑规划与逐节撰写，产出完整初稿。
3. **工作流模式**：对初稿进行自动化评审打分，未达标则进入自我修改闭环，迭代完善直至通过阈值。

---

## 3. 阶段一：数据准备

**目的**：为论文撰写提供基础素材与上下文。

在项目根目录创建 `data/` 文件夹，将以下文件放入其中：

| 文件类型 | 示例 | 作用 |
|----------|------|------|
| 参考文献 PDF | `related_work_1.pdf` | 作为知识库或引用来源，Agent 可提取关键信息 |
| 实验数据 | `experiment.csv`, `results.json` | 作为论文的数据基础，直接用于 Results 章节 |
| 模型/示意图 | `model_arch.png`, `flowchart.svg` | 自动归入论文 `figures/` 目录 |
| 论文大纲 | `outline.md`, `draft.tex` | `.tex` 文件会触发**编辑模式**，Agent 在现有手稿上修改而非从零撰写 |
| 其他参考资料 | `notes.md`, `supplement.docx` | 作为上下文参考材料归入 `sources/` |

**文件处理规则**：
- `.tex` → `drafts/`（触发编辑模式，保留版本历史）
- 图片（`.png`/`.jpg`/`.svg` 等） → `figures/`
- 数据文件（`.csv`/`.json`/`.txt` 等） → `data/`
- 其他文件（`.md`/`.docx`/`.pdf` 等） → `sources/`

> **提示**：`data/` 中的文件会在标准模式启动时被自动分类复制到论文输出目录，原始文件将被清理。

---

## 4. 阶段二：标准模式（生成初稿）

**目的**：通过与 AI Agent 的交互，完成文献调研、逻辑规划与逐节撰写，生成可直接评审的论文初稿。

### 4.1 启动方式

```bash
# 进入交互式 CLI
writer

# 或显式指定标准模式
writer standard
```

### 4.2 交互流程

#### （1）用户输入描述信息

在提示符下输入论文需求，例如：

```
> 撰写一篇关于 Transformer 在蛋白质结构预测中应用的论文，要求 NeurIPS 格式，包含真实引用
> Continue the paper          （继续编辑最近的论文）
> Update my draft about LLM   （按主题匹配已有论文并继续编辑）
```

#### （2）智能意图识别

Agent 自动判断用户意图：
- **继续编辑**：输入包含 `continue`、`edit`、`update`、`the paper` 等关键词时，Agent 自动定位最近修改的论文目录，读取已有内容并在其基础上修改。
- **创建新论文**：无匹配关键词时，创建新的时间戳目录（`writing_outputs/YYYYMMDD_HHMMSS_<topic>/`）。

#### （3）文献检索与资料收集

Agent 根据论文主题，动态调用外部工具检索真实文献：
- **`research_lookup`**：通过 Perplexity Sonar Pro 检索学术文献，获取真实引用信息。
- **`parallel_web`** / **`WebSearch`**：进行网络搜索，补充最新研究进展与背景资料。

检索结果自动整理并保存至当前论文目录的 `sources/` 中，作为后续撰写的参考上下文。

#### （4）Venue 模板自动下载（投稿目标为会议/期刊时）

若用户在描述中提及具体投稿目标（如 `ACL/EMNLP`、`ICLR 2025`、`NeurIPS`、`CVPR`、`IEEE`、`ACM` 等），Agent 在撰写 `.tex` 前会**强制**调用 `skills/venue-templates/scripts/download_venue_template.py` 拉取该会议/期刊的官方 LaTeX 模板（`.sty` / `.cls` / `.bst`）。

```bash
# 列出当前支持的所有 venue
python skills/venue-templates/scripts/download_venue_template.py --list

# 拉取 ACL/EMNLP/NAACL 模板（同一份 acl.sty）
python skills/venue-templates/scripts/download_venue_template.py \
    --venue acl \
    --output-dir writing_outputs/<paper_dir>/final \
    > writing_outputs/<paper_dir>/final/venue_manifest.json

# 拉取年度版本模板（如 ICLR 2025）
python skills/venue-templates/scripts/download_venue_template.py \
    --venue iclr --year 2025 \
    --output-dir writing_outputs/<paper_dir>/final \
    > writing_outputs/<paper_dir>/final/venue_manifest.json
```

| Venue 别名 | 覆盖会议/出版社 | 是否需 `--year` |
|-----------|----------------|----------------|
| `acl` | ACL / EMNLP / NAACL（共用 `acl.sty`） | 否 |
| `iclr` | ICLR | **是** |
| `neurips` | NeurIPS | **是** |
| `icml` | ICML | **是** |
| `aaai` | AAAI | **是** |
| `ijcai` | IJCAI | **是** |
| `cvpr` | CVPR / ICCV / ECCV | **是** |
| `ieee` | IEEE 会议/期刊（IEEEtran） | 否 |
| `acm` | ACM SIG 系列（acmart） | 否 |
| `lncs` | Springer LNCS / MICCAI | 否 |

下载产物会缓存于 `~/.cache/venue-templates/<venue>_<year>/`，并解压到论文目录的 `final/`。`venue_manifest.json` 由 Agent 直接消费，其中的 `documentclass` / `usepackages` / `bibstyle` 字段会被原样写入 `.tex` 头部，确保编译产出与会议官方模板视觉完全一致。

> **失败兜底**：若官方模板 URL 失效，可手动将 `.sty` / `.bst` 放入缓存目录的 `manual/` 子目录后重跑下载脚本，脚本会自动覆盖优先使用。

#### （5）逐节撰写与逻辑连贯

Agent 按照学术论文结构（IMRaD 或其他指定格式）逐节撰写：

1. **Introduction**：研究背景、问题陈述、贡献点
2. **Methods**：实验设计、模型架构、评估指标
3. **Results**：数据展示、对比实验、消融实验
4. **Discussion**：结果分析、局限性、未来工作
5. **Conclusion**：总结与展望

**撰写特点**：
- **动态调用工具**：撰写过程中可随时调用文献检索工具验证引用、调用数据文件生成图表。
- **前后连贯**：后文自动继承前文的术语定义、符号系统与逻辑链条，确保全文一致性。
- **格式合规**：默认输出标准 LaTeX，支持数学公式、表格、图表引用与 BibTeX 引用。
- **防中断机制**：通过 `Stop Hook` 强制 Agent 在长文档生成过程中持续工作，不会中途询问"是否继续"。

#### （6）输出初稿

撰写完成后，所有产物保存在 `writing_outputs/<timestamp>_<topic>/` 下：

```
writing_outputs/20260426_143000_transformer_protein/
├── drafts/                 # 各章节草稿与编辑历史
├── final/                  # 最终初稿（.tex / .pdf）
├── references/             # BibTeX 引用文件
├── figures/                # 论文图表
├── data/                   # 实验数据副本
├── sources/                # 检索到的参考文献与资料
├── progress.md             # 撰写过程日志
└── SUMMARY.md              # 论文摘要与说明
```

---

## 5. 阶段三：工作流模式（迭代评审完善）

**目的**：对标准模式生成的初稿进行系统化、量化的同行评审，并根据评审意见进行闭环修改，直至论文质量达到预设标准。

### 5.1 启动方式

```bash
# 基础用法（对默认最新初稿进行评审完善）
writer workflow

# 指定主题并配置评审参数
writer workflow "Transformer在蛋白质结构预测中的应用" \
    --threshold 8.0 \
    --iterations 5

# 复用已有论文目录（在已有初稿基础上迭代）
writer workflow "继续评审" \
    --output writing_outputs/20260426_200126_task_adaptive_rag_kg \
    --threshold 7.0 \
    --iterations 3
```

参数说明：
- `--threshold, -t`：评审通过阈值（默认 `7.0`，满分 10 分）
- `--iterations, -i`：最大迭代次数（默认 `3`）
- `--output, -o`：指定已有论文目录，跳过新建目录步骤，直接在现有目录中追加评审结果。适用于对标准模式产出的初稿进行迭代完善。

### 5.2 评审与迭代流程

#### （1）设置评审参数

系统根据用户输入或默认值确定：
- **阈值分数 `a`**：论文通过评审的最低平均分（如 `8.0`）。
- **最大迭代次数 `b`**：最多允许的修改轮次（如 `5`）。

#### （2）多维度量化评审

将初稿发送给 **评审 Agent**，从以下四个维度进行 1-10 分量化评分：

| 维度 | 说明 | 评审要点 |
|------|------|----------|
| **创新性 (Innovation)** | 研究的原创程度 | 问题定义是否新颖、方法是否有创新、与已有工作的区别 |
| **严谨性 (Rigor)** | 研究方法的严密性 | 实验设计是否严谨、数据分析是否充分、控制变量是否合理 |
| **清晰度 (Clarity)** | 论文表达的易读性 | 逻辑结构是否清晰、表达是否准确、图表是否易懂 |
| **方法论 (Methodology)** | 研究方法的先进性 | 方法是否适合问题、技术细节是否充分、复现性如何 |

评审 Agent 同时生成：
- **对抗性问题**：3 个刁钻但具有建设性的学术问题，揭示论文潜在弱点。
- **修改建议**：针对低分维度的具体、可操作的改进方案。

#### （3）条件判断与闭环修改

```
初稿 -> 评审打分 -> 条件判断
              |
      +-------+-------+
      |               |
  达标(>=a)       未达标(<a)
      |               |
   输出终稿      迭代次数 < b?
                      |
              +-------+-------+
              |               |
            是             否
              |               |
        进入修改环节      输出当前最佳版本
              |
        吸收评审意见
        回答对抗性问题
        靶向重构论文
              |
        返回 -> 重新评审
```

**自我修改环节**：
- Agent 根据评审报告中的低分维度、对抗性问题和修改建议，对论文进行**靶向重构**。
- 必须回答所有对抗性问题，并将答案融入论文相关部分。
- 保持 LaTeX 格式、术语一致性与全文结构稳定。
- 修改后的版本进入下一轮评审。

**熔断机制**：
- 若连续两轮评审平均分变化小于 `0.1`，判定为"修改进入瓶颈"，提前终止迭代，避免无效循环。

#### （4）终稿输出

迭代结束条件（满足其一即可）：
1. 平均分达到或超过阈值 `a`；
2. 已达到最大迭代次数 `b`；
3. 触发熔断机制（连续改进幅度过小）。

输出产物：

```
writing_outputs/20260426_143000_transformer_protein/
├── final/
│   ├── manuscript.tex      # 终稿 LaTeX 源文件
│   └── manuscript.pdf      # 编译后的 PDF
├── reviews/
│   └── review_iteration_1.json   # 每轮评审的详细评分与意见
├── PEER_REVIEW.md          # 最终评审报告汇总
└── ...（其他同标准模式输出）
```

---

## 6. 项目目录结构

```
academic-writing/
├── pyproject.toml                  # 包配置 (Python>=3.10)
├── requirements.txt                # 依赖清单
├── readme.md                       # 项目说明
├── .env                            # API Key 配置
├── .gitignore
│
├── writer/                        # 主包源码
│   ├── __init__.py                 # 公开 API
│   ├── main.py                     # CLI 入口（标准模式 + 工作流模式）
│   ├── runner.py                   # 异步 API（generate_paper / generate_paper_workflow）
│   ├── helpers.py                  # 通用辅助（API Key、数据文件、目录扫描、文本统计）
│   ├── data_types.py               # 数据模型（ProgressUpdate、PaperResult、WorkflowConfig 等）
│   │
│   └── pipeline/                   # 学术写作流水线
│       ├── stages.py               # 所有阶段节点（基类 + 大纲/撰写/审计/评审/条件/修改/工具）
│       └── controller.py           # 工作流控制（上下文 + 迭代器 + 流水线构建 + 引擎）
│
├── skills/                         # 技能脚本
│   ├── citation_management/        # 引用管理（BibTeX、DOI、PubMed/Google Scholar）
│   ├── diagram_generation/         # 科学示意图生成
│   ├── image_generation/           # AI 图片生成
│   ├── research_lookup/            # 学术研究检索（Perplexity Sonar Pro）
│   ├── venue-templates/            # 会议/期刊官方 LaTeX 模板下载（ACL/ICLR/NeurIPS/ICML/AAAI/IJCAI/CVPR/IEEE/ACM/LNCS）
│   │   └── scripts/
│   │       ├── download_venue_template.py  # 自动拉取官方 .sty/.cls/.bst（带缓存与 manual/ 兜底）
│   │       ├── customize_template.py
│   │       ├── query_template.py
│   │       └── validate_format.py
│   └── web_search/                 # 网络搜索
│
├── .claude/                        # Claude Agent 配置（项目级覆盖用户级）
│   ├── settings.local.json         # 本地 env 块 + 工具白名单
│   └── WRITER.md                   # Writer Agent 系统提示（含 venue 模板强制流程）
│
└── writing_outputs/                # 论文输出目录（运行时生成）
    └── YYYYMMDD_HHMMSS_<topic>/
        ├── drafts/                 # 草稿版本
        ├── final/                  # 最终稿（PDF / TEX）
        ├── references/             # BibTeX 引用文件
        ├── figures/                # 图表
        ├── data/                   # 实验数据
        ├── sources/                # 检索到的参考文献与资料
        ├── reviews/                # 评审报告（工作流模式）
        ├── progress.md             # 进度日志
        └── SUMMARY.md              # 项目摘要
```

---

## 7. 环境配置与运行

### 7.1 安装依赖

提供两种安装方式，按需选择：

#### 方式一：极简安装（仅装依赖）

适合临时试用或偏好直接运行模块的场景。

```bash
pip install -r requirements.txt
```

启动时使用 `python -m writer.main` 替代 `writer` 命令：

```bash
python -m writer.main                # 进入交互式 CLI
python -m writer.main standard       # 标准模式
python -m writer.main workflow ...   # 工作流模式
```

#### 方式二：可编辑安装（推荐，注册 `writer` 命令）

将本项目以"可编辑模式"安装到环境中，自动注册 `writer` 命令到 PATH。

```bash
pip install -r requirements.txt   # 先装依赖
pip install -e .                  # 再安装项目本身（注册 writer 命令）
```

启动时直接使用 `writer` 命令（本文后续示例统一以此为准）：

```bash
writer                            # 进入交互式 CLI
writer standard                   # 标准模式
writer workflow ...               # 工作流模式
```

> **提示**：若使用 `uv`，将上述 `pip` 替换为 `uv pip` 即可，效果一致。

### 7.2 配置 API Key

在项目根目录创建 `.env` 文件：

```env
# OpenAI 兼容代理（OpenAI SDK 路径，BASE_URL 必须带 /v1）
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_MODEL="claude-sonnet-4-6"

# Anthropic / Claude CLI（Claude Agent SDK 路径，BASE_URL 不能带 /v1）
# Claude CLI 会自动在 BASE_URL 后追加 /v1/messages，因此此处仅给根 URL
export ANTHROPIC_API_KEY="your-key"
export ANTHROPIC_BASE_URL="https://api.example.com"

# 可选：学术研究检索（Perplexity Sonar Pro，通过 OpenRouter）
export OPENROUTER_API_KEY="your-key"

# 可选：联网搜索（Parallel Web Systems）
export PARALLEL_API_KEY="your-key"
```

> **⚠️ BASE_URL 关键差异**：
> - `OPENAI_BASE_URL` **必须以 `/v1` 结尾**（OpenAI SDK 要求带版本前缀）
> - `ANTHROPIC_BASE_URL` **绝不能带 `/v1`**（Claude CLI 会自行拼接 `/v1/messages`）
> - 若误把 `ANTHROPIC_BASE_URL` 设为带 `/v1` 后缀，请求路径会变成 `/v1/v1/messages`，CLI 收到 404 后会以 *"该模型不存在"* 形式报错——这是最常见的踩坑点。

> **⚠️ 用户级与项目级 settings 的覆盖关系**：Claude CLI 会按 用户级 `~/.claude/settings.json` → 项目级 `.claude/settings.json` → 本地 `.claude/settings.local.json` 的顺序合并 `env` 块，**后者覆盖前者**。若用户级 settings 中预置了其它模型（如 kimi、deepseek），项目根目录的 `.claude/settings.local.json` 已显式写入 `ANTHROPIC_*` 锁定为 `claude-sonnet-4-6`，确保不会被全局配置抢走。如需改用其它模型，编辑 `.claude/settings.local.json` 的 `env` 块即可。

### 7.3 安装 LaTeX（PDF 编译必需）

```bash
# macOS
brew install --cask mactex-no-gui

# Ubuntu/Debian
sudo apt-get install texlive-full
```

---

## 8. 注意事项

1. **工作目录**：所有输出强制写入当前工作目录下的 `writing_outputs/`，不会写入 `/tmp/` 等临时目录。
2. **模型配置**：通过 `OPENAI_MODEL` 环境变量统一指定模型，默认 `claude-sonnet-4-6`。`writer/main.py`、`writer/runner.py` 与 `writer/pipeline/stages.py` 中的 `ReviewNode`、`AuditNode` 等节点均会以 `setting_sources=["project"]` + 显式 `env` 透传的方式调用 Claude Agent SDK，主动注入 `ANTHROPIC_MODEL` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`，避免被用户级 `~/.claude/settings.json` 中的全局 env 抢占。
3. **`ANTHROPIC_BASE_URL` 格式**：必须是根 URL（如 `https://api.example.com`），**不能带 `/v1`** —— CLI 会自行追加 `/v1/messages`。误带 `/v1` 会导致 404，CLI 错误信息会伪装成"模型不存在"。
4. **`setting_sources` 锁定**：Claude Agent SDK 调用以 `setting_sources=["project"]` 启动，仅加载项目级 `.claude/settings.json` 与 `.claude/settings.local.json`，主动忽略用户级全局配置，确保多机器/多用户协作时行为一致。
5. **Venue 模板强制流程**：当撰写目标为正式会议/期刊投稿时，`.claude/WRITER.md` 强制 Agent 在写 `.tex` 之前先调用 `download_venue_template.py`，禁止使用手写的 `\documentclass[11pt]{article}` + 自制版心，确保产出 PDF 与会议官方模板视觉一致（避免 desk reject）。
6. **数据文件处理**：`data/` 中的文件在标准模式启动时会被自动分类复制到论文目录，原始文件将被删除。如需保留原件，请提前备份。
7. **编辑模式**：若 `data/` 中包含 `.tex` 文件，Agent 会将其识别为已有手稿，进入编辑模式而非从零撰写。
8. **知识库检索**：`HybridRetriever` 的关键词检索当前为占位实现，实际生产环境建议接入 Elasticsearch 或 Meilisearch。
9. **LaTeX 依赖**：若系统未安装 `pdflatex`，编译步骤会静默失败，但 `.tex` 源文件仍会完整生成。
10. **评审评分鲁棒性**：工作流模式下，ReviewNode 对 LLM 评分响应采用三级容错机制——(1) 标准 JSON 解析（含 Markdown 围栏提取）；(2) 正则兜底，即使 JSON 字符串内出现未转义的半角双引号或 Markdown 混排，也能提取 `score` 数值；(3) 若仍失败，自动重试一次极简 Prompt（仅要求返回一个 1-10 数字）。重试失败才会降级到 5.0，并打印完整原始响应到 stderr 供诊断。以此消除"所有维度恒为 5.0"的伪故障。
