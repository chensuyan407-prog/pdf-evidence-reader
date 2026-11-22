# ITERATION_REPORT — pdf-evidence-reader v0.1.0 → v0.2.0

日期 2026-11-21。只列实际运行得到的数据；未测项写“未测”。不预估比赛总分。

## 范围与方法

- 开发副本：`v0.2/`（复制自 `bundle/`）。原提交包 `pdf-evidence-reader_arm_v0.1.0.zip`、原 `bundle/` 和 D 盘 `grading/` 评审证据只读，未改动。
- 案例：评审用的三份 PDF（ResNet 12 页双栏、GPT‑3 75 页、BERT 16 页 150 dpi 扫描）与加密 ResNet，原样复制到 `tests/pdfs/`；另生成遮挡版 BERT（`tests/make_masked_pdf.py`，首页摘要涂白）。
- 同一套检查跑两遍：`tests/run_regression.py --skill <scripts>`，旧版用 v0.1 脚本副本（`--parse-only`，跳过 v0.1 没有的工具），新版用 `submission/scripts`。检查按内容与坐标定位，不依赖区块 ID。
  - 旧版报告：`tests/out/old/report.json`；新版报告：`tests/out/new/report.json`
- 留出表：开发前按页码清单选定 GPT‑3 第 19 页表 3.8（`tests/HOLDOUT.md`），表格代码定稿后才首次打开。开发只用了第 63 页表 H.1 和第 15 页表 3.4。
- 实现中没有按论文名、页码、表号或已知答案写的分支。问题相关的检索词只出现在 `tests/agent_fill.py`（模拟 Agent 作答步骤，不属于 Skill）。
- 运行环境：Windows 11，Python 3.13.14，PyMuPDF 1.28.2，numpy 2.4.4，**未安装 tesseract**。

## 三类结果

| 类别 | v0.1 | v0.2 | 证据 |
|---|---|---|---|
| **最终答案正确**（6 题 + 遮挡对照 2 项） | 评审实测 B 组严格 3/6（GPT‑3 2/2 需手工数列；ResNet 公式字形失真；BERT 2 题空答）。本轮未重跑 v0.1 作答 | 8/8：GPT‑3 2/2、ResNet 2/2、BERT 2/2（页图目读）、遮挡 BERT 2/2 正确留空 | `tests/out/new/cases/*/result.json`；评审数据见 grading/GRADE_REPORT.md |
| **结构化表可直接取值**（按行列名称查询） | 1/6（表 H.1 五项全部 `row_not_found`：T35 只有表头；表 3.4 可取） | 6/6；留出表数值 36/36 正确，但按名称查询 0/3（`row_ambiguous`） | `tests/out/*/gpt3/doc.json`，`tests/out/holdout_result.json` |
| **验证器抓住错配**（5 个错配输入 + 1 个正确输入） | 抓住 0/5；正确输入退出码 0，但 v0.1 不识别表格证据，对每个输入都只报“无引文”并退出 0 | 抓住 5/5，正确输入通过；逐字段给出判定 | `tests/out/*/verifier/*/verification.json` |

说明：v0.2 的最终答案由 `tests/agent_fill.py` 按 SKILL.md 流程生成，不是独立 Agent 在官方 Harness 中作答。BERT 两项由开发者看 `crop_region.py` 的裁图读出，存在同人出题、同人读图的偏差。官方条件（DeepSeek Harness、64 次请求预算、三次重复）未测。

## 逐项对照

### 1. 跨列表头与数据体分离（GPT‑3 第 63 页，表 H.1）

- **原因**：表顶线 y=101、组下短线 y=113、表头底线 y=131、表底线 y=646。v0.1 只把间距小于 140pt 的横线串成一张表，因此表格止于表头。各模型列之间只有 2–5pt 空白，按“≥7pt 空白”分列会把八列并成一格。
- **改法**（`parse_pdf.py`）：
  - 只要两线之间没有正文行或题注，就串成同一张表；
  - 按词左/右/中线对齐跨行聚类出叶列；稀疏列由表头补出；
  - 无表头的纯文字列并入行标签列；
  - `\cmidrule` 下方的列构成组头跨列；
  - 输出列 `path`、行 `records`、`key_columns` 和单元格 `bbox`；新增 `table_query.py`。
- 现有输出接口（`rows`、`cells`、`header`、`caption`、md/html/csv）保留。

| 查询（`table_query.py`，按名称取值） | v0.1 | v0.2（与原页核对） |
|---|---|---|
| HellaSwag / acc / dev × Zero‑Shot / 13B | row_not_found | 70.9 ✓，bbox [263.3, 134.9, 273.7, 140.9] |
| LAMBADA / ppl / test × One‑Shot / 6.7B | row_not_found | 4.61 ✓ |
| 相邻列：同行 Zero‑Shot / 175B、/ 6.7B | row_not_found | 78.9 ✓、67.4 ✓ |
| 同名不同指标：LAMBADA / acc / test × One‑Shot / 6.7B | row_not_found | 65.4 ✓ |
| 只给 Name=LAMBADA（不给指标） | — | `row_ambiguous`（正确拒答） |
| 第二张开发表：表 3.4 GPT‑3 Few‑Shot × De→En | 40.6 ✓ | 40.6 ✓（行标签“GPT‑3 Few‑Shot”不再被拆成两列） |
| **留出表** 表 3.8：数值单元格 | — | 36/36 与原页一致 |
| **留出表** 表 3.8：按名称查询 3 次 | — | 0/3，均 `row_ambiguous` |

- **副作用核查**（GPT‑3 全文）：
  - 检出表从 35 张变为 26 张，其中带 “Table” 题注的 21 张全部检出；
  - 无题注候选从 15 处降到 5 处（第 29、51×2、56、58 页），是否误检未逐一核实；
  - 题注编号支持 “Table H.1”，附录标题 “H …” 成为章节，目录行不再被当作标题。

### 2. 真引文支持错答案（`verify_result.py`）

- 新证据类型 `table_cell`：`{field, table, page, row{列名: 值}, column "组头/叶列", value}`。验证器会按名称重新取值。
- 每个字段分别报告引文是否匹配、值是否受支持。
- 旧格式（`claim` + `quote`）仍可读，按文本证据处理。

| 输入（每次只改一个条件） | v0.1 退出码 / 报告 | v0.2 判定（退出码） |
|---|---|---|
| 正确：70.9，定位正确 | 0 / 只报“无引文” | `supported_by_table_cell`（0） |
| 答案改为 999，证据不变 | 0 / 未发现 | `value_mismatch`（1） |
| 答案改为同行 175B 的 78.9，仍称来自 13B | 0 / 未发现 | `column_mismatch`（1）：指出 78.9 在 “Zero‑Shot / 175B” 列 |
| 值正确，证据行换成 StoryCloze | 0 / 未发现 | `evidence_unsupported`（1）：证据值 70.9，该格实为 79.5 |
| 值正确，证据页换成 62 | 0 / 未发现 | `evidence_unsupported`（1） |
| 评审原负控 `controls/wrong-cell`（旧格式，999 + 真引文） | 评审实测 6/6 通过、零问题 | `value_not_in_evidence`（1）；同文件另一字段为 `string_match_only`，并注明引文含 26 个数字，无法证明取的是哪一个 |

- **普通文本问答**，验证器实际检查两件事：
  - 引文是否在所引区块、表或页中出现（忽略空白、大小写、连字符）；
  - 答案字符串或其中的数字是否出现在已核实的引文里。
- 判定为 `string_match_only` 时，输出里写明“字符串匹配，不做语义检查”。图像目读判定为 `image_read_unverified`。

### 3. 扫描件（BERT 扫描版，无 OCR）

| 项 | v0.1 | v0.2 |
|---|---|---|
| 页面状态 | partial，16 页 `ocr_needed_but_unavailable`，无其他产物 | 同样 partial，另输出每页页图（150 dpi）和 XY‑cut 区域；第 1 页 17 个区域 |
| 首页摘要区域裁图 `crop_region.py` | 无此工具 | 墨迹比 0.129，可读；读出 “Bidirectional Encoder Representations from Transformers”“SQuAD v2.0 Test F1 to 83.1” |
| 两题答案 | 空（评审） | 正确，`field_status=image_read`，验证判定 `image_read_unverified`（未标成文本层逐字匹配） |
| 遮挡版（摘要涂白） | 未测 | 裁图 `blank=true`，两字段留空，`unreported` 写明区域与墨迹比；验证判定 `unreported_ok` |

- 遮挡后右栏正文仍写有 BERT 全称，但题目问的是摘要，所以没有拿来填答，只在 `uncertainty` 中说明。
- 安装 OCR 后的识别质量未测。

### 4. 双栏与边缘文字（ResNet 第 1 页；GPT‑3 第 63 页）

| 检查（按文字与坐标定位） | v0.1 | v0.2 |
|---|---|---|
| 摘要两段在阅读顺序中相邻 | 否：中间插入 13 个块（图刻度、图例、右栏图注与正文） | 是（相邻） |
| 摘要内无右栏内容插入 | 否 | 是 |
| “training error (%)” 归图 | margin_note | figure_text |
| “test error (%)” 归图 | margin_note | figure_text |
| arXiv 侧标从正文剔除 | 是 | 是（margin_note） |
| 页码从正文剔除 | 是 | 是（footer） |
| GPT‑3 p63 表数据不挂在 Acknowledgements 下 | 否：p63‑b7 为 body，章节 Acknowledgements | 是：role=table，章节 “Results on All Tasks for All Model Sizes” |

- ResNet 有栏缝的页数：v0.1 为 8/12（第 1、5、8、11 页漏检），v0.2 为 12/12。
- GPT‑3（单栏）无栏缝的页数：v0.2 为 75/75。开发中曾在第 46 页出现一次误报，已修正。

### 5. 公式（ResNet 第 3 页 (1)、(2)）

| 项 | v0.1 | v0.2（与公式裁图逐符号核对） |
|---|---|---|
| (1) | `y = F(x, {W_{i}}) + x. (1)`：粗体与花体丢失 | `𝐲 = ℱ(𝐱, {W_{i}}) + 𝐱. (1)`，LaTeX `\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_{i}\}) + \mathbf{x}. (1)` ✓ |
| (2) | `y = F(x, {W_{i}}) + W_{s}x. (2)` | `𝐲 = ℱ(𝐱, {W_{i}}) + W_{s}𝐱. (2)` ✓ |
| 公式裁图 | 无 | `equations/eq_p3_b9.png`、`eq_p3_b15.png` |

- 依据是字体名：CMBX 为粗体，CMSY 中的大写字母为花体，CMMI7 字号加基线偏移判为下标。`text` 和 `marked` 字段不变。
- 字体名无法识别时，会在 `uncertainty` 中说明并给出裁图。
- 只核对了这两个公式；其他公式、分式、根号等未测。

### 附带修复

- `parse_pdf.py --pages 1,3` 这类不连续页码：v0.1 抛 IndexError，v0.2 正常完成。

## 作者自测（合成样本，`python src/reproduce.py`）

| 指标 | v0.1（评审复现） | v0.2（第一阶段后 / 第二阶段后） |
|---|---|---|
| 双栏顺序准确率 Skill / 原始流 / 几何排序 | 1.0000 / 0.5714 / 0.9643 | 不变 / 不变 |
| 跨栏、跨页断句拼回；表格单元格 | 1.0；1.0 | 不变 |
| 加密 → failed；扫描 → partial；重复运行一致 | 是 | 是 |
| 示例案例证据核验 | 3/3（引文） | 4/4：`table_value` 为 `supported_by_table_cell`，另两项为 `string_match_only` |
| 伪造引文负控 | 退出码 1 | 退出码 1 |

- 示例案例的旧证据用自由文字 `claim`，新验证器判为“无证据”。因此示例改为按字段名填写证据，属于预期变化。

## 耗时（本机，`tests/out/*/report.json`）

| 解析 | v0.1 | v0.2 |
|---|---|---|
| ResNet 12 页 | 1.60 s | 1.78 s |
| GPT‑3 75 页 | 3.66 s | 4.03 s |
| BERT 扫描 16 页 | 0.28 s | 1.20 s（多出的时间用于渲染页图和切分区域） |
| 加密 PDF | 0.21 s | 0.22 s |

- 作者自测全流程 1.23 s（脚本内计时）。
- 每案例“填写 + 验证”0.26–0.40 s，为脚本时间，不含模型思考。
- 官方条件下的端到端时间与请求数未测。

## 剩余失败与未测

1. **叠放表**（留出表 3.8）：中间第二组表头被当作数据行，行标签重复。按名称查询返回 `row_ambiguous`，不返回错值，但也取不到值。尚未修复；修复后需要另找一张未看过的表作为留出表。
2. GPT‑3 中 5 个无题注表候选未核实是否误检；无横线表格仍只依赖 PyMuPDF `find_tables`。
3. 公式只核对了 2 个；不重建分式、根号、上下限或矩阵。
4. 扫描页的目读由开发者完成，未由独立 Agent 验证；OCR 路径未测。
5. 文本问答的验证仍是字符串匹配（已在输出中写明）。
6. 未测：其他出版社版式、三栏、真实低质量扫描、官方 Harness 三次重复。
7. GitHub 公开链接仍未提供。

## 复现命令（在 `v0.2/` 下）

```bash
python tests/make_masked_pdf.py
python tests/make_verifier_fixtures.py <review folder>/controls/wrong-cell/result.json
python tests/run_regression.py --skill submission/scripts --label new
python tests/run_regression.py --skill <v0.1 scripts 的副本> --label old --parse-only
python src/reproduce.py
```
