# ITERATION_REPORT — pdf-evidence-reader v0.2.0 → v0.3.0

日期 2026-11-21。只列实际运行得到的数据，未测项写“未测”，不预估比赛分数。
本轮首要目标：解析残缺时，系统不把错误结果标成“已获原文支持”；读得清楚的表格和公式，调用者能直接查询到完整结果。

## 条件

- 开发副本 `v0.3/`（复制自 `v0.2/`）。原 v0.2.0 包、`v0.2/`、D 盘 `grading/`、`grading-v2/` 只读使用。
- 环境：Windows 11，Python 3.13.14，PyMuPDF 1.28.2，numpy 2.4.4，fontTools 4.62.1，无 tesseract。
- v0.2 基线：把 `v0.2/submission/scripts` 复制到临时目录，用**同一批测试脚本、同一输入、同一机器、同一计时口径**运行。
- 三组记录：
  - 作者原自测：`src/reproduce.py`。
  - 已知缺陷回归：v0.2 套件 `tests/run_regression.py`，加上 v0.3 套件 `tests/v03/run_v03.py`。
  - 新论文留出：`tests/v03/run_holdout.py`。
- 实现中没有论文名、固定页码、目标数值或答案特例。问题相关的检索词只出现在 `tests/v03/agent_fill_v03.py`，它模拟 Agent 的作答步骤，不属于 Skill。

## 四项指标

### 指标 1–2、4：评审 v2 的三个案例（6 个字段）

| 指标 | v0.2 | v0.3 | 证据 |
|---|---|---|---|
| 最终答案正确数 | 3/6 | **6/6** | `results_v0.3/v02_new_suite.json`、`v03_new_suite.json` → `cases` |
| 无需目读即可正确结构化提取的字段数 | 0/6 | **3/6** | 同上 |
| 必须人工复核的字段数 | 3/6 | 3/6 | 同上 |

- **v0.2 答对的 3 个字段**：Attention 两个字段都按完整模型名查不到行；GloVe 公式 (8) 按编号查不到。答对的是 GloVe α（字符串匹配）和 Hubble 两项（目读）。
- **v0.3 无需目读的 3 个字段**：
  - Attention 38.1 和 2.3 × 10¹⁹ FLOPs，判定为 `supported_by_table_cell`。
  - GloVe 公式 (8)，判定为 `supported_by_equation_record`（状态 auto）。
- **两个版本都需复核的 3 个字段**：GloVe α 是 `string_match_only`；Hubble 两项是 `image_read_unverified`，扫描件没有文字层。

### 指标 3：错误答案被误判为支持的次数（负控）

| 版本 | 被误判为支持 | 证据 |
|---|---|---|
| v0.2 | **2/15** | `results_v0.3/v02_new_suite.json` → `controls` |
| v0.3 | **0/15** | `results_v0.3/v03_new_suite.json` → `controls` |

- **v0.2 误判的两例**：评审 v2 的残缺 “2.3 ·”，以及 Chinchilla 拆分解析中只含 “Billion” 的单元格。
- **v0.3 的已知例外**：冻结 1 时，留出探针（Chinchilla “Billion”）曾 1 次被误判为支持。修复后该例转为回归（见下文）。

## 一、表格：科学计数法与跨列单元格（Attention 第 8 页表 2）

| 检查（`tests/v03/run_v03.py`，table 组） | v0.2 | v0.3 |
|---|---|---|
| 逻辑列数 | 11 列 | **5 列** |
| 10 个模型名整名保留（含 “[38]” 等引用） | 否，被拆成 2–3 列 | 是 |
| Transformer (base model) × BLEU/EN-FR | row_not_found | **38.1** |
| base 的训练开销，原样 | 拆成 “3.3 ·” 和 “1018” | `3.3 · 10^18` |
| base 的训练开销，规范化 | — | `3.3 × 10¹⁸`，numeric 3.3e18 |
| big 的训练开销，原样 / 规范化 | 拆分 | `2.3 · 10^19` / `2.3 × 10¹⁹` / 2.3e19 |
| 跨两个方向共享 | — | `shared_across` = EN-DE、EN-FR |
| bbox 覆盖尾数、乘号、10 与上标指数 | — | 是：[410.1, 229.1, 448.0, 240.3]，`value_parts` 逐字形带坐标 |
| 相邻列：base EN-DE 27.3；big EN-DE 28.4、EN-FR 41.8；GNMT 两个方向成本分别为 2.3·10^19 和 1.4·10^20 | 0/5 | 5/5 |
| GNMT 行只给组表头查询 | — | `col_ambiguous`（正确拒答） |
| 合成表（`tests/v03/make_numeric_table.py`）：16 个值，含整数、千分位、负数、×/· 科学计数法和单独的 10^20 | 0/16 | 16/16 |
| 合成表：两处上标脚注（b、*）记为脚注而非指数 | 0/2 | 2/2 |
| 合成表：引用标签 “Prior work [12]” 不拆 | 否 | 是 |

v0.3 合计 36/36，v0.2 合计 0/36。

改法（`parse_pdf.py`、`table_query.py`）：
- 表格改用字符级记号，按字符间距和上下标层级切词。行内合并 “尾数 · 10^指数”、“10^指数”、“负号 + 数字”。数字后紧贴的上标记为脚注；“[38]” 不算数字。
- 二元运算符（· × + − = / ±）把两侧紧邻记号绑成一个表达式。没有自身表头的纯文字列，按整格文本判断后并入行标签列。
- 冻结后又加了一条：两个相邻列在每一行都只隔一个词距、且边界两侧不全是数字时，合并为一列。
- 居于两列之间且与两列都重叠的值记为跨列。
- 副作用（v0.2 回归套件）：GPT‑3 表 H.1、表 3.4、ResNet、BERT 的 34 项全部通过。Attention 表 1 的复杂度列 “O(n² · d)” 在 v0.2 被拆成两列，v0.3 为 4 列、整格。

## 二、验证器：“正确查到了错误解析的单元格”

分两层独立测试。

**端到端**（新解析器从原 PDF 取完整值）：见上表。

**验证器隔离**（输入残缺解析）：

| 负控（`tests/v03/fixtures/controls/`，搬到新目录后验证） | v0.2 判定 | v0.3 判定 / needs_review |
|---|---|---|
| 正确：38.1 与 2.3 × 10¹⁹ FLOPs | 38.1 被支持；成本查不到（不支持组表头查询） | supported_by_table_cell / 否（两项） |
| 虚构的 999 | value_mismatch | value_mismatch / 否 |
| 同行相邻列的 27.3 | column_mismatch | column_mismatch / 否 |
| 错误行 | evidence_unsupported | evidence_unsupported / 否 |
| 错误页 | evidence_unsupported | evidence_unsupported / 否 |
| 评审 v2 的残缺 “2.3 ·”（v0.2 原始解析） | **supported_by_table_cell，无需复核** | **parsed_cell_incomplete / 是**（见下方原因） |
| 人为构造 “2.3 ·”（v0.3 解析中截断） | evidence_unsupported（查询失败） | parsed_cell_incomplete / 是 |
| 人为构造 “2.3”（无运算符，只剩尾数） | evidence_unsupported（查询失败） | parsed_cell_incomplete / 是（原页上表达式越过格边） |
| 人为构造 “2.3 · 10”（丢指数） | evidence_unsupported（查询失败） | parsed_cell_incomplete / 是 |
| Chinchilla 拆分解析中的 “Billion” | **supported_by_table_cell** | parsed_cell_incomplete / 是（左侧一个词距内有 “8.0”） |
| 真实但无关的 dropout 引文支持 BLEU = 0.1 | string_match_only / 是 | string_match_only / 是（写明“字符串匹配，不做语义检查”） |

- 评审 v2 残缺 “2.3 ·” 在 v0.3 被拒的原因有三条：格尾为运算符；原页在格边 2.2 pt 处接着是 “1019”；答案本身以运算符结尾。返回的复核位置为第 8 页、bbox [380.1, 224.2, 457.8, 246.3]。
- 三个人为构造的残缺在 v0.2 的“拒绝”，是因为它不支持组表头查询，属于查询失败，不是识别出残缺。

- **v0.3 的输出**：每个字段分别给出 `value_equals_parsed_cell`、`source_region_complete`、`needs_visual_check`、`reasons` 和 `review_location`。原 PDF 通过 SHA-256 定位。找不到 PDF 时降为 `supported_by_parsed_cell_region_unchecked`，并要求复核。
- **完整性检查的误报（扫描全部表体单元格）**：
  - GPT‑3：3019 格中标为可能残缺 36 格。其中第 26、38 页的 30 格是真拆分，例如 “3.9 (2e-4)”。第 63 页 4 格是误报：文字标签与下一列文字只隔 2.0 pt。
  - Attention：142 格中 0 格。
  - ResNet：366 格中 124 格，几乎都在第 5 页架构表（多行单元格，确实解析不全）和第 7、11 页的拆分标签。
  - 误报方向一律是“要求复核”。

## 三、跨多个文字块的编号公式（GloVe 第 4 页公式 (8)）

| 检查 | v0.2 | v0.3 |
|---|---|---|
| 按编号 8 返回页码、bbox、裁图、转写状态 | 无（编号从 7 跳到 9） | 第 4 页，bbox [78.85, 473.47, 290.27, 519.52]，`equations/eq_p4_n8.png`，状态 auto |
| 求和上下限 i,j=1 与 V | 否 | `\sum_{i,j=1}^{V}` |
| f(X_ij) | 否 | `f(X_{ij})` |
| w_i 的转置 | 否 | `w_{i}^{T}` |
| 带波浪号的 w_j 和 b_j | 否 | `\tilde{w}_{j}`、`\tilde{b}_{j}` |
| 对数前负号 | 否 | `-\log X_{ij}` |
| 右括号外平方 | 否 | `)^{2}` |
| 编号缺口告警 | 有（7→9） | 无，1–22 连续 |

做法：
- 以编号行为锚点，在同栏内聚合数学行，遇正文行停止。
- 无 Unicode 映射的数学字形，从嵌入字体程序取字形名：Type 1 解密后读 CharStrings 顺序，并按 FreeType 规则把 .notdef 换到第 0 位；CFF 读 charset；TrueType 读 glyph order。
- 以主基线重建上下限、重音、上下标、分式和根号。
- 降为 `partial` 并列出原因的情况：多条基线、主字号正文词、上下标无合理基字。GloVe 中公式 (4)、(10)、(11)、(15)、(20) 因此标为 partial。

## 四、图像证据与原始 PDF 绑定（Hubble 第 2 页）

| 负控（搬到新目录后验证） | v0.2 | v0.3 |
|---|---|---|
| 正确裁图 | image_read_unverified | image_read_unverified，来源 bound+rendered（重渲染像素差 0.0），需复核 |
| 证据页改为 5（评审 v2 的 wrong_image_page） | image_read_unverified（未发现） | image_provenance_mismatch：“evidence cites page 5 but the crop was cut from page 2” |
| 换成另一份 PDF 的裁图 | 未发现 | mismatch：源 PDF SHA-256 不属于本案例 |
| 裁图文件被替换（同记录、换第 3 页像素） | 未发现 | mismatch：裁图 SHA-256 与记录不符 |
| bbox 越界（x1 = 900，页宽 471.36） | 未发现 | mismatch：bbox 超出页面 |
| 旧格式、无来源记录 | image_read_unverified | image_read_unbound，降级并需复核 |
| 整个结果目录搬走、源 PDF 路径失效、PDF 随目录携带 | — | 按 SHA-256 找到 PDF，bound+rendered |
| 同上但不带 PDF | — | bound（记录自洽，未重渲染），仍需复核 |

- 来源记录存放在 `<crop>.prov.json` 和证据里的 `provenance` 字段，内容包括源 PDF 哈希、物理页、页面尺寸、bbox、dpi、裁图哈希和像素尺寸。
- 路径相对于 result.json。目读结果始终保持人工复核。

## 五、效率（同一问题：Attention 表 2 base 的 EN-FR BLEU）

| 条件 | 工具耗时 | 调用步骤 | 结果 | 需人工找框 |
|---|---:|---:|---|---|
| A：PyMuPDF 全文取文本 + 渲染第 8 页 | 0.18 s | 2 | 文本中有 38.1 | 是，需读图或数列 |
| A：只取第 8 页 | 0.07 s | 2 | 同上 | 是 |
| v0.2 B 冷启动（run_case + 查询） | 2.06 + 0.26 s | 2 | row_not_found | 是 |
| v0.3 B 冷启动（全文解析，无缓存） | 1.36 s | 2 | 38.1 | 否 |
| v0.3 同篇第二次查询（不重解析） | 0.10 s | 1 | 38.1 | 否 |
| v0.3 新输出目录、缓存命中 | 0.54 s | 2 | 38.1 | 否 |
| v0.3 只解析第 8 页（列出未读 14 页） | 0.55 s | 2 | 38.1 | 否 |

- v0.3 冷启动比 v0.2 快，主要因为：PyMuPDF find_tables 回退只在有 “Table” 题注的页上运行；公式重建只在有编号行的页上运行。GPT‑3 全文解析从约 4 s 降到 2.6–2.8 s。
- 缓存键由三部分组成：PDF 内容哈希、解析器版本、配置（页范围、OCR、语言、预览）。
- A 与 B 使用相同的页范围和同等的读图权限。
- 模型请求数：未测。

## 六、新论文留出

论文在改代码前选定并记录了哈希，见 `tests/HOLDOUT_v0.3.md`。冻结实现（Freeze 1，记录了脚本哈希）之后，才首次打开目标页。标准答案先从页图写入 `tests/holdout_truth_v0.3.json`，再运行解析器。

| 留出项（冻结 1） | v0.2 | v0.3 | 说明 |
|---|---|---|---|
| Kaplan 表 5、表 6（10 个科学计数法值与单位，加两项“检出并关联题注”） | 2/12 | **12/12** | |
| Kaplan 公式 (1.4) 符号检查（7 项，含检出与状态诚实） | 2/7 | 4/7 | 状态为 partial，未误标 auto。真错误 1 处：指数中 α 的下标 B 丢失层级（`\alphaB`）。另 2 项失败是因为标准答案写的是 ASCII “*”，原页印的是数学星号 “∗”。标准答案未事后修改。 |
| Kaplan 验证器：正确答案受支持、构造截断被拒 | 0/2 | **2/2** | |
| BN 第 8 页结果表（题注为 “Figure 4”，页面无任何横线） | 0/7 | 0/7 | 未检测到 |
| Chinchilla 表 2、表 3 | 2/13 | 2/13 | 标签和值被拆成多列 |
| Chinchilla 公式 (2)、(3) | 4/14 | 4/14 | (2) 标为 auto，结构正确但用 Unicode 数学斜体（𝐿、𝛼），预登记的 ASCII 检查不通过；(3) 为 partial |
| 合计 | 10/55 | 24/55 | 证据：`results_v0.3/holdout_v02.json`、`holdout_v03_freeze1.json` |

**留出后修改。** Chinchilla 探针暴露出一个验证器漏洞：只含 “Billion” 的单元格被判为支持。为此加了两条通用规则：
- 验证器：格边一个词距内有非数字相邻文字时，判为可能残缺。
- 解析器：每行都只隔一个词距、且边界两侧不全是数字的相邻列，合并为一列。

修改后 Chinchilla **转为回归**，探针加入负控。没有剩余未用论文，所以这两条规则**只在回归数据上验证**。Freeze 2 重跑：Kaplan 和 BN 与 Freeze 1 相同；Chinchilla 表格升到 4/13（Tokens 两项改对，FLOPs 列名与 Table 2 列名仍失败）；合计 26/55，见 `results_v0.3/holdout_v03_freeze2.json`。

## 七、作者原自测

`python src/reproduce.py`，指标与 v0.2 一致：
- 双栏顺序 1.0（基线 0.5714 / 0.9643）、断句拼回 1.0、表格单元格 1.0。
- 加密样例 → failed；扫描样例 → partial；重复运行一致。
- 示例案例证据 4/4；伪造引文负控退出码 1。

证据：`results_v0.3/selftest_metrics_v03.json`、`execution/run.log`。

## 八、已知缺陷回归

- **v0.2 套件**（`tests/run_regression.py`）在 v0.3 上 34/34：表格 6/6、版式 7/7、公式 4/4、扫描 1/1、异常 2/2、验证器 6/6、最终答案 8/8。证据：`results_v0.3/v03_known_defects_suite.json`。
  - 中间曾有一次回退：GPT‑3 正确负控被误判为残缺（表 H.1 相邻数字只隔 2.0 pt）。定位后改为“两侧都是数字时不判为连续”，已恢复。
- **v0.3 套件**（`tests/v03/run_v03.py`）：v0.3 为 74/74（表格 36、公式 10、案例 6、负控 19、效率 3）；v0.2 为 11/74。

## 九、仍存在的失败与限制

1. 没有横线、且题注写成 “Figure N” 的表格检测不到（BN 留出 0/7）。
2. 右对齐的多词表头会把首词归入左列（Chinchilla 表 3 的 “FLOPs FLOPs”），按列名查询返回 col_not_found。
3. 公式：
   - 指数内的嵌套下标会丢失层级。
   - Unicode 数学斜体字母未规范化。
   - 多行或对齐公式只给 partial 加裁图。
   - 只支持带编号的展示公式。
4. 叠放表（表中再次出现表头）返回 row_ambiguous。
5. 验证器的词距规则在文字列之间有少量误报（GPT‑3 3019 格中 4 格），误报方向为要求复核。
6. 冻结后新增的两条规则没有独立留出验证。
7. 未测：OCR 质量（无 tesseract）、真实低质量扫描、三栏版式、官方 Harness 下的模型请求数与三次重复、GitHub 公开链接。

## 复现（在 `v0.3/` 下；测试 PDF 不随包分发，来源与哈希见 `tests/HOLDOUT_v0.3.md` 和评审目录）

```bash
pip install -r submission/requirements.txt
python src/reproduce.py
python tests/run_regression.py --skill submission/scripts --label v03_known
python tests/v03/make_controls.py submission/scripts <grading-v2 目录>
python tests/v03/run_v03.py --skill submission/scripts --label v03
python tests/v03/run_holdout.py --skill submission/scripts --label v03
```
