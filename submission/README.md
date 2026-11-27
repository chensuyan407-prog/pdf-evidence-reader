# pdf-evidence-reader v0.4.0 — 科研 PDF 版式解析 + 证据定位 Skill

把科研 PDF 解析成可继续处理、可回指原文的结构化结果；让 Agent 按行列名称取表值、按页码区块引文、按页图坐标目读扫描页；
交付前由验证器分别检查“引文是否在原位置”和“答案是否被证据支持”。

## 文件

| 路径 | 作用 |
|---|---|
| `SKILL.md` | Agent 调用说明：流程、证据格式、验证判定含义、扫描页目读、诚信规则、异常处理 |
| `scripts/parse_pdf.py` | 版式解析：区块 → 页眉页脚/边注/图内文字分离 → 栏检测与阅读顺序 → 角色 → 章节树 → 表格 → 公式 → 段落拼接 → 输出 |
| `scripts/table_query.py` | 按行标识（`--row 列名=值`）和列表头路径（`--col 组头/叶列`）取单元格，返回值、页码、单元格坐标 |
| `scripts/find_evidence.py` | 文本检索（BM25 / 正则 / 章节 / 表 / 页 / 按编号取公式 `--equation N`） |
| `scripts/crop_region.py` | 按坐标或区域 id 裁剪页图并计算墨迹比例（`blank` 表示区域无可读内容） |
| `scripts/run_case.py` | 一条命令：找 PDF、解析、生成 `result.json` 骨架（含 `case_id`、契约字段、`field_status`）、`outline.md`（论文一页地图）、`structure_summary.json`，并打印可直接复制的后续命令 |
| `scripts/fill_result.py` | 命令式回填：`put`（表格单元格 / 公式 / 原文引文 / 裁图）、`unreported`、`note`、`show`、`finalize`、`repair`；写入前按验证器规则预检，原子写入并留备份，保护 `case_id`/`status` 等核心键 |
| `scripts/verify_result.py` | 逐字段核验：引文匹配 + 表格单元格重查 + 原页区域完整性 + 图像来源绑定；旧格式结果仍可读（缺来源记录时降级） |
| `scripts/mathlayout.py` | 按编号聚合跨块公式、解码数学字形名、重建上下限/重音/上下标/分式 |
| `scripts/provenance.py` | 裁图来源记录（源 PDF 哈希、页、bbox、裁图哈希）与核对 |
| `examples/minimal_example.md` | 最小调用示例（示例输出） |

## 安装

```bash
pip install "pymupdf>=1.23" numpy fonttools   # 测试环境：Python 3.13, PyMuPDF 1.28.2, numpy 2.4, fontTools 4.62（可选，用于 CFF/TrueType 字形名）
# 可选 OCR：tesseract-ocr (+ chi_sim)
```

## v0.2 方法要点

1. **表格**：横线按“线间区域不含正文行或题注”串成一张表（不再限制线间距）；无竖线时按 **词的左/右/中线对齐聚类** 得到叶列
   （不再依赖列间空白，能分开间距仅 2–5pt 的数字列）；只有少量值的稀疏列在表头上方有词时补出；无表头的纯文字列并入左侧行标签列；
   表头行按列中线中点分配，`\cmidrule` 下方的列构成组头跨列；每列给出 `path`（组头/叶列），每行给出 `records` 与 `key_columns`，每个单元格带 `bbox`。
   有框线但正文为段落或邻近 Figure 题注的盒子不算表。
2. **阅读顺序**：栏缝检测允许少量通栏块（作者行、标题）压过栏缝，取低占用区间中占用最小的最宽段；栏两侧都须有段落级文本块；表格块不参与栏缝计算。
3. **图内文字**：图片与矢量图簇区域（含 16pt 外扩）内的小字、短字、旋转字 → `figure_text`（如坐标轴名称），不进入正文；
   只有页面外缘 12% 带内的旋转字才是 `margin_note`（如 arXiv 侧标）。
4. **章节**：新增附录字母编号（`H`、`H.1`）；带页码结尾的目录行不当作标题；题注编号支持 `Table H.1`。
5. **公式**：按字体名恢复数学字母表（CMBX→`\mathbf`/𝐱，CMSY 大写→`\mathcal`/ℱ，MSBM→`\mathbb`，EUFM→`\mathfrak`），
   按字号与基线偏移恢复上下标；输出 `latex`、`unicode_math`、`math_fonts`、公式裁图 `crop`，以及不确定性说明。
6. **扫描页**：无 OCR 时输出页图与 XY-cut 版面区域；`crop_region.py` 裁剪并给出墨迹比例，供有视觉能力的 Agent 目读；
   字段状态区分 `text_layer` / `image_read` / `unreadable`，验证器把目读结果标为 `image_read_unverified`。
7. **验证器**：`table_cell` 证据按行列名称重查，答案与单元格不符时区分 `value_mismatch`、`column_mismatch`（答案在同一行其它列）、
   `row_mismatch`；文本证据只做字符串匹配并明确标注 `string_match_only`。
8. 修复：`--pages 1,3` 这类不连续页码不再崩溃。


## v0.3 变更（本轮首要目标：解析残缺时不把结果标成“已获原文支持”）

1. **表格数值**：表格改用字符级记号（`table_tokens`），按“字符间距 + 上下标层级”切词，再在行内合并
   `尾数 · 10^指数`、`10^指数`、`负号 + 数字`；数字后紧贴的上标（a、b、*）记为脚注而非指数；`[38]` 这类引用不当作数字；
   运算符（· × + − = / ±）把两侧紧邻的记号绑成一个表达式（`O(n² · d)`）。每个单元格保留 `parts`（每个原始字形及坐标）、
   `raw`（原样）和 `norm`（`value`、`canonical`、`pretty`）。没有自身表头的纯文字列按整格文本判断后并入行标签列。
   表体单元格居于两列之间且与两列都重叠时记为跨列（`colspan`），`records` 中两列共享同一值。
2. **table_query.py**：返回 `value`（原样）、`value_pretty`、`value_numeric`、`value_parts`、`cell_bbox`、`shared_across`、
   `footnote_marker`；可用组表头路径查询跨列单元格；行标签可省略末尾脚注标记。
3. **验证器**：逐字段分开报告 `value_equals_parsed_cell`、`source_region_complete`、`needs_visual_check`。
   按 SHA-256 找到原 PDF，重读单元格框周围字符：值越过单元格边缘（含末尾运算符、上标指数、紧邻运算符）、框内数字缺失、
   单位落在框外，即判 `parsed_cell_incomplete` 并给出复核位置；找不到 PDF 时降为 `…region_unchecked`。
   科学计数法按数值比较。新增 `equation` 证据类型。
4. **公式**：`mathlayout.py` 以编号行为锚点，在同栏内聚合数学行（遇正文行停止）；
   对无 Unicode 映射的数学字形，从嵌入字体程序取字形名（Type 1 解密 CharStrings、CFF charset、TrueType glyph order）；
   按主基线重建求和上下限、重音（˜ ˆ ¯ …）、上下标、分式（横线）与根号。区域含多条基线、主字号正文词、
   上下标无合理基字时，状态降为 `partial` 并列出原因；每个公式都有裁图。
5. **图像证据**：`crop_region.py` 写出 `<crop>.prov.json`（源 PDF 哈希、物理页、页面尺寸、bbox、dpi、裁图哈希）；
   验证器核对页码、bbox、文档、裁图哈希、越界，并在能找到 PDF 时重新渲染比对像素；路径相对 result.json，整体移动后仍可复查。
6. **效率**：`parse_pdf.py --cache`、`run_case.py --cache/--pages`；缓存键 = PDF 内容哈希 + 解析器版本 + 配置；
   `doc.json` 记录 `pages_parsed`/`pages_not_parsed`。只在有题注的页面运行 PyMuPDF find_tables 回退，只在有编号行的页面运行公式重建。
7. 所有脚本以 UTF-8 输出（v0.2 在中文 Windows 控制台遇到 − 等字符会崩溃）。

## v0.4 变更（本轮目标：真实论文结构难点 + Agent 低轮次、少犯错）

1. **章节层级**：一行标题被 MuPDF 拆成多条“行”时按基线合并计数（BN 的 §2、§3、§3.1–3.4，GloVe 的 §4.4、§4.7 此前全部漏检）；
   编号后只有符号的行（“1 N”）不再当标题；新增无编号加粗/加大单行标题（`kind: styled`，层级按相对字号推定）、
   “Appendix A: …”、“D.2. Title” 形式的附录编号；“Proof of Lemma 1” 不再被误判为目录行。行尾 “Batch-” + 大写开头的下一行保留连字符。
2. **图像形式的表格/图**：有文字层的页面中，内部没有文字层的嵌入图片（`raster_regions`）被识别并告警；靠近 “Table N” 题注时登记到
   `tables_as_images`，解析时直接生成带来源记录的裁图（可作 `image_region` 证据）；有 tesseract 时附 OCR 文本（标注未核验）。
   `table_query.py` 对这类表返回 `table_is_image` + 裁图路径，而不是 `table_not_found`。
3. **跨页续表**：题注 “Table 2 (continued)” 与原表关联（`continued_from`）；`table_query.py` 按原表题注查询时自动检索所有续表部分
   （无表头的续表借用原表列名），`--list` 汇总全部行键，`--markdown` 导出全部部分；行列查不到时给出 `did_you_mean`。
4. **公式**：分式与根号按“最外层横线优先”处理，分母中的根号不再被提前吃掉（Attention 公式 (1) v0.3 误标 `auto` 且结构错误，
   现为 `softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V`）；无法解释的横线（上划线等）使状态降为 `partial`；
   LaTeX 中 Unicode 数学斜体字母规范为普通字母、希腊字母规范为 `\alpha` 等命令，粗体保留 `\mathbf`；`\cdotmin` 这类粘连命令拆为 `\cdot min`。
5. **工作流**：`fill_result.py` 取代手改 JSON（DeepSeek Harness 手改大 JSON 易写坏语法、漏字段、覆盖 `case_id`）。每次 `put` 用验证器同一套规则预检，
   不通过则拒绝写入并给出原因/候选；`finalize` 为缺失字段写入 `not_determined` 说明、运行验证器、按结果设置 `success/partial/failed` 与
   `status_reason`（v0.5：只有答案与原文矛盾、主 PDF 无法解析或过半必填字段未确定时才为 `partial`；少量未确定/图像读取/需复核项写入 `warnings` 并保持 `success`；`finalize` 仅在答案与原文矛盾时 exit 1）；JSON 被手改坏时自动从备份恢复，`case_id` 与契约键从骨架恢复。旧流程（手改 + `verify_result.py`）保持可用。
6. **契约兼容**：契约里出现 `case_id`/`evidence`/`sources`/`status` 等与元数据同名的字段时，骨架保留元数据结构，验证器判为 `meta_field`
   （v0.3 会把 `evidence` 字段置 null 并判 `no_evidence` 失败）。
7. **导航与预算**：`outline.md` 汇总章节（含页码范围、区块 id）、每张表的列路径与行键、图像表、公式编号与状态、图、扫描页与告警；
   `structure_summary.json` 增加作者/单位、摘要、章节页码范围、图、页面地图。`SKILL.md` 给出技能目录与解释器自动探测、批量命令与 64 次请求预算。

## 测试

开发副本中的 `tests/`（不属于 Skill 资源）：`python tests/run_regression.py --skill submission/scripts --label new`，
`python tests/v03/run_v03.py --skill submission/scripts --label v03`，`python tests/v04/run_v04.py`（v0.4 新能力、工作流、契约形状、提交包限制），
详见 `tests/README.md` 与 `ITERATION_REPORT_v0.4.md`。

## 限制（v0.3 实测；v0.4 已解决的条目见上文，未变的条目保留）

- **无横线表格**：只靠横线检测表格；页面没有任何横线、且题注写成 “Figure N” 的表（留出 BN 第 8 页）完全检测不到；
  有 “Table N” 题注时才回退到 PyMuPDF find_tables。
- **表头右对齐的多词表头**（如 Chinchilla 表 3 的 “FLOPs (in Gopher unit)”）首词可能归入左列，列名变成 “FLOPs FLOPs”，按列名查询失败（返回 col_not_found，不返回错值）。
- **数学斜体字母**：字体把字母映射为 Unicode 数学斜体（𝐿、𝛼）时，LaTeX 中原样保留，未规范为 ASCII/\alpha。
- **公式嵌套上下标**：指数里的下标（L^{1/α_B} 中的 B）会丢失层级；多行/对齐公式、行内夹正文的公式标为 partial 并给裁图。
  按编号定位只适用于有编号的展示公式。
- **叠放表**（表中间再出现一组表头）：下半段表头被当作数据行，行标签重复，返回 row_ambiguous（不返回错值）。
- **验证器的“字距相邻”规则**有误报：文字标签与下一列文字间距小于一个词距时（GPT-3 表 H.1 的 “ARC (Challenge)” 与 “acc”，2.0 pt），
  会把正确的文字单元格判为可能残缺（实测 GPT-3 3019 个表体单元格中 4 个）。误报方向是“要求复核”，不会把错误标成支持。
- 扫描页无 OCR 时只能靠 Agent 目读；目读结果不做机器核验（来源一致只证明图来自该页）。OCR 质量未测（本机无 tesseract）。
- 测试覆盖：6 篇真实论文（ResNet、GPT‑3、BERT 扫描、Attention、GloVe、Hubble 扫描）为回归；Kaplan、BN 为冻结代码后的留出；
  Chinchilla 因留出探针暴露验证器漏洞后转为回归。其他出版社版式、三栏、真实低质量扫描未测。
- v0.4 补充：无编号标题按“加粗/加大 + 单行 + 后接正文”判定，层级按相对字号推定，非出版社样式信息；在 9 篇真实论文上新增的样式标题经人工抽查为真标题
  （如 ResNet 附录的 “PASCAL VOC”、“MS COCO”），但未做独立留出验证。
- 图像表只做定位、裁图和（可选）OCR，不做单元格结构重建；本机无 tesseract，OCR 路径未实测。
- 留出集（Kaplan、BN、Chinchilla）的公式检查在 v0.4 修改后从 26/55 升到 34/55；这些改动是在看过留出结果之后做的，留出对这部分已不再是盲测。
