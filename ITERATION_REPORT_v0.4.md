# ITERATION REPORT v0.3.1 → v0.4.0

所有数字均为本机实跑（Windows 11，Python 3.13，PyMuPDF 1.28.2，无 tesseract），基线为 commit `dea5dc8`（v0.3.1）。
没有运行官方 Harness，也没有模型请求数的实测；下文的“请求数”是按工作流步骤估算的，不是实测值。

## 一、审计发现（改动前，v0.3.1）

| 问题 | 证据 | 影响的评分点 |
|---|---|---|
| 一行标题被 MuPDF 拆成 3–5 条“行”，`len(lines) <= 2` 判定失败，真实标题被漏掉 | BN 漏 §2、§3、§3.1–3.4；GloVe 漏 §4.4、§4.7；Chinchilla 漏 D.1–D.4、H.1–H.4 | 拆得清（章节层级） |
| 编号 + 单个符号（“1 N”）被当成标题 | BN 出现两个标题 “1 N” | 拆得清 |
| 分母中的根号先被当成根号处理，分式失去分母，公式被线性化却仍标 `auto` | Attention 公式 (1) 输出 `softmax(Q\sqrt{d_{k}} K^{T})V`，状态 `auto` | 排得对 / 诚实性 |
| LaTeX 中保留 Unicode 数学斜体（𝐿、𝛼）；`\cdotmin` 粘连成未定义命令 | Chinchilla 公式 (2)、Attention 公式 (3) | 公式保真 |
| 有文字层的页面里，嵌入为图片的表格完全不可见，`table_query` 只返回 `table_not_found` | 合成样例；GPT-3 有 22 个此类图片区域（均为图） | 提得全 / 核得回 |
| “Table 2 (continued)” 不与原表关联，续表的行查不到 | 合成样例 | 拆得清（跨页续表） |
| 契约字段名与元数据同名（`evidence`、`case_id`）时，骨架把 `evidence` 置为 null，验证器判 `no_evidence` 失败 | 构造契约复现 | 运行稳定性 |
| Agent 需手改 `/app/output/result.json`：易写坏 JSON、漏字段、覆盖 `case_id` | SKILL.md v0.3.1 第 5 步 | 调用与易用性 / 异常处理 |
| 仓库缺少 v0.3 控制样本（`tests/v03/fixtures/controls/*`、`legacy_wrong_cell`），干净克隆无法完整跑 v03 套件；`run_regression.py` 在 GBK 控制台打印数学字符时崩溃 | 本次复现 | 增益可验证性 |

## 二、改动

| 文件 | 改动 |
|---|---|
| `submission/scripts/parse_pdf.py`（0.4.0） | 按基线计“印刷行”判断短标题；拒绝“编号 + 符号”标题；无编号样式标题（`kind: styled`，相对字号定层级）；`Appendix A: …`、`D.2. Title`；“Lemma 1” 类结尾不当目录行；行尾 `Batch-` + 大写保留连字符；图片区域检测（`raster_regions`）、题注关联（`tables_as_images`）、带来源记录的裁图与可选 OCR；“(continued)” 题注关联续表 |
| `submission/scripts/mathlayout.py` | 横线按最外层优先处理（根号/分式），根号按基线归入分子/分母；无法解释的横线 → `partial`；`latex_plain()`：数学斜体 → 普通字母，希腊 → `\alpha` 等，粗体 → `\mathbf`/`\boldsymbol`，粘连命令拆分 |
| `submission/scripts/table_query.py` | 续表合并检索（`part`、`table_parts`），`table_is_image`，`did_you_mean`，`--markdown`，`--list` 含全部续表行键与列路径；`Table 1` 不再匹配 `Table 12` |
| `submission/scripts/fill_result.py`（新） | `put` / `unreported` / `note` / `show` / `finalize` / `repair`；预检、原子写入、`.bak`、骨架恢复 `case_id` 与契约键、受保护键 |
| `submission/scripts/run_case.py` | 元数据优先构建骨架；骨架副本；`outline.md`；`structure_summary.json` 增强；打印 `next_steps`（绝对路径命令） |
| `submission/scripts/verify_result.py` | 契约中的元数据同名字段判 `meta_field`；`status_reason` 计入元数据键 |
| `submission/SKILL.md` | 重写：技能目录/解释器自动探测、outline 优先、批量命令、`fill_result.py` 工作流、预算、失败处理 |
| `submission/README.md`、`examples/minimal_example.md` | v0.4 方法与限制；示例改为新工作流（示例输出取自实跑） |
| `tests/v04/`（新）、`tests/run_regression.py`、`tests/README.md` | v0.4 套件；控制台 UTF-8 |

## 三、实测结果

| 套件 | v0.3.1 | v0.4.0 |
|---|---|---|
| `tests/run_regression.py`（表格/版式/公式/扫描/异常/验证器/最终答案） | 34/34 | **34/34** |
| `tests/v03/run_v03.py`（表格 36、公式 10、案例 6、负控 19、效率 3） | 74/74 | **74/74** |
| `src/reproduce.py`（双栏顺序 1.0、断句 1.0、表格 1.0、加密 failed、扫描 partial、重复运行一致、示例 4/4、伪造引文负控 exit 1） | 全部通过 | **全部通过，数值不变** |
| `tests/v03/run_holdout.py`（Kaplan、BN、Chinchilla） | 26/55 | **34/55**（见第五节说明） |
| `tests/v04/run_v04.py` | 9/47（工作流组因无 `fill_result.py` 只记 1 项失败） | **67/67**（synthetic 20、real 25、workflow 21、package 1） |

v0.3.1 在 v0.4 套件上已经通过的 9 项：三栏顺序、栏缝、页眉删除、跨页段落拼接、“RELATED WORK” 标题、标题不当章节、续表第 1 部分取值、一个 LaTeX 单元用例、提交包限制。**三栏阅读顺序与跨页拼接在 v0.3.1 就已正确**，v0.4 只是补上了回归用例。

### 章节恢复（同一组 9 篇 PDF，v0.3.1 → v0.4.0）

| 论文 | 章节数 | 新增（摘要） | 删除 |
|---|---|---|---|
| BN 1502.03167 | 13 → 18 | §2、§3、§3.1–3.4、附录 “Variant of the Inception Model Used” | 两个假标题 “1 N” |
| GloVe | 15 → 17 | §4.4、§4.7 | — |
| Chinchilla 2203.15556 | 30 → 41 | D.1–D.4、H.1–H.4、J、模型卡小节 2 个 | — |
| ResNet | 16 → 20 | 附录内加粗小标题 “PASCAL VOC”、“MS COCO” 等 4 个 | — |
| Kaplan 2001.08361 | 46 → 48 | “List of Figures/Tables” | — |
| GPT-3 | 52 → 53 | “Contributions” | — |
| Attention / BERT 扫描 / Hubble 扫描 | 不变 | — | — |

新增条目经人工对照页面，均为真实标题；无编号标题的规则没有独立留出验证。

### 公式

| 公式 | v0.3.1 | v0.4.0 |
|---|---|---|
| Attention (1) | `softmax(Q\sqrt{d_{k}} K^{T})V`，**auto（错误）** | `softmax(\frac{QK^{T}}{\sqrt{d_{k}}} )V`，auto |
| Attention (3) | `…\cdotmin(…)` | `…\cdot min(…)` |
| Chinchilla (2) | `\hat{𝐿}(𝑁,𝐷)≜𝐸+\frac{𝐴}{𝑁^{𝛼}}…` | `\hat{L}(N,D)≜E+\frac{A}{N^{\alpha}} +\frac{B}{D^{\beta}}` |

### 效率

- 解析耗时：GPT-3（75 页）1.9 s → 2.5 s（图片区域检测；只为表格类/无题注图片生成裁图，最多 12 个）。其余论文变化 < 0.1 s。
- 估算的请求数（未实测）：发现 + 解析 1、读题与 outline 1–2、查询 2–4（批量）、回填 1–3（批量）、finalize 1、修正 ≤ 2，约 8–13 次。
  v0.3.1 需要 Agent 读取并重写整个 `result.json`，每次修改都要重发全部内容；v0.4 每个字段一条命令，失败时不写入。

## 四、诚实性与边界

- 图片形式的表格：只定位、裁图、可选 OCR（标注未核验），**不重建单元格**；目读结果仍是 `image_read_unverified`，状态为 `partial`。
- `put --force` 允许保留不在引文中的转述，但验证器会判 `value_not_in_evidence`，`finalize` 给出 `partial` 并列出该字段。
- `finalize` 的状态规则：存在验证失败、`not_determined`、`unreadable` 或目读字段 → `partial`；全部 PDF 解析失败或没有可回答字段 → `failed`；
  `string_match_only` 不降级（它是文本问答的常态），但会出现在 `needs_review`。
- 本机没有 tesseract，OCR 相关路径（扫描页 OCR、图片区域 OCR）均未实测。

## 五、留出集说明

留出集从 26/55 升到 34/55，提升来自 Chinchilla 公式 (2)/(3) 的符号检查（LaTeX 规范化）。这些改动是在 v0.3 已经看过留出结果之后做的，
**对这部分留出已不是盲测**，只能算回归。BN 第 8 页 “Figure 4” 无横线表格（0/7）与 Chinchilla 表 2/表 3 列名问题（标签拆列、“FLOPs FLOPs”）仍未解决。

## 六、仍存在的限制

1. 无任何横线、且题注写成 “Figure N” 的表格仍检测不到（BN 留出 0/7）。
2. 右对齐多词表头首词归入左列（Chinchilla 表 3）。
3. 公式：指数内嵌套下标层级（Kaplan 1.4 的 `α_B`）；多行/对齐公式只给 partial + 裁图；`min`/`argmin` 带下限时字母被拆散（Chinchilla (3)、(7) 为 partial）。
4. 图片表不做结构重建；扫描页仍依赖 Agent 目读。
5. 未测：官方 Harness 下的请求数与三次重复、真实低质量扫描、出版社三栏期刊实物（只测了合成三栏）。

## 七、复现

```bash
python tests/run_regression.py --skill submission/scripts --label new
python tests/v03/run_v03.py --skill submission/scripts --label v03
python tests/v03/run_holdout.py --skill submission/scripts --label v03
python tests/v04/run_v04.py --skill submission/scripts --label v04
python src/reproduce.py
# 前后对照：git archive dea5dc8 submission/scripts | tar -x -C /tmp/v031 && python tests/v04/run_v04.py --skill /tmp/v031/submission/scripts --label v031
```
测试 PDF 不随仓库分发（`tests/pdfs/`、`tests/holdout_pdfs/` 在 .gitignore 中）；本次从开发目录 `v0.3/tests/` 复制，
v03 控制样本（`tests/v03/fixtures/controls/*`，约 17 MB，含 PDF）同样来自开发目录，尚未提交到仓库。
