# FSO 编码与交织仿真平台

[![源码验证](https://github.com/KristenYue/fso-coding-interleaving-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/KristenYue/fso-coding-interleaving-simulator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%20--%203.12-3776AB)
![NumPy](https://img.shields.io/badge/数值计算-NumPy-013243)
![SciPy](https://img.shields.io/badge/科学计算-SciPy-8CAAE6)
![RS/LDPC](https://img.shields.io/badge/信道编码-RS%20%7C%20LDPC-00599C)
![Monte Carlo](https://img.shields.io/badge/实验方法-Monte%20Carlo-6A5ACD)
[![开源许可](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

该项目为开发者硕士毕业设计，面向自由空间光通信（Free-Space Optical Communication，FSO）的编码、交织与湍流信道仿真平台。

项目围绕 IM/DD OOK 光通信链路，建模 AWGN、Gamma–Gamma 大气湍流、指向误差和固定相干块衰落，对比 Reed–Solomon 与 LDPC 编码在不同交织深度和信道相干条件下的误码性能、缓存代价和解码复杂度。

> 核心目标：建立一条从信道模型验证、编解码器验证、编码链路仿真、统计分析到独立复现的完整实验链，避免未经验证的组件直接进入最终性能对比。

---

## 10 秒了解项目

```text
AWGN 基准验证
      │
      ▼
Gamma–Gamma 湍流与指向误差
      │
      ▼
未编码 FSO 误码基线
      │
      ├───────────────┐
      ▼               ▼
RS 编解码验证      LDPC 编解码验证
      │               │
      ▼               ▼
固定相干信道 + 交织深度扫描
      │               │
      └───────┬───────┘
              ▼
      码率匹配的 RS/LDPC 对比
              │
              ▼
       编码增益与湍流敏感性
              │
              ▼
        独立实现路径复现
```

项目重点研究四个问题：

1. 不同码率的 RS 和 LDPC 如何进行公平的 `Eb/N0` 对比；
2. 交织深度 `D` 与物理信道相干长度 `Lc` 如何独立控制；
3. 同一衰落组内的相关帧如何计算可信的统计区间；
4. 主实验结论能否通过独立实现的信道与编解码路径进行复现。

---

# 工程规模

| 维度 | 当前内容 |
|---|---:|
| 主实验脚本 | **14 个** |
| 独立复现模块 | **12 个** |
| 实验链阶段 | **Task 01–10** |
| 已记录 Bad Case | **7 个** |
| RS 编码 | **RS(255,223)** |
| RS 码率 | **约 0.8745** |
| LDPC 编码 | **LDPC(512,449)** |
| LDPC 码率 | **约 0.8770** |
| 信道组件 | AWGN、Gamma–Gamma、指向误差、固定相干衰落 |
| 统计方法 | Wilson 区间、零误码上界、分组感知区间 |
| 可复现机制 | 固定随机种子、条件派生种子、断点续跑 |
| 自动检查 | 源码边界验证 + 全部实验模块编译检查 |

公开仓库包含完整实验源码和复现入口。CSV、NPZ、图片等生成结果由脚本在本地运行后产生，并保存在 Git 忽略目录中。

---

# 系统模型

## FSO 链路

项目研究的基本链路为：

```text
信息比特
   ↓
RS 或 LDPC 编码
   ↓
交织
   ↓
OOK 调制
   ↓
Gamma–Gamma 湍流
   +
指向误差
   +
AWGN
   ↓
解交织
   ↓
RS 或 LDPC 解码
   ↓
BER、置信区间与复杂度分析
```

接收信号可以抽象为：

```text
y = h × x + n
```

其中：

- `x`：OOK 调制后的发送符号；
- `h`：由大气湍流、指向误差和相干结构共同决定的信道增益；
- `n`：加性高斯白噪声；
- `y`：接收信号。

## 信道模型

| 模型 | 工程作用 |
|---|---|
| AWGN | 建立无衰落参考，独立验证编解码器 |
| Gamma–Gamma | 描述不同湍流强度下的光强起伏 |
| 零偏置指向误差 | 模拟光束抖动与接收孔径耦合损失 |
| 固定相干块衰落 | 控制一段连续符号共享同一衰落增益 |
| 未编码链路 | 提供编码增益对比基线 |

各信道组件先独立验证，再组合进入完整编码链路。

---

# 完整实验链

## Task 01：信道组件验证

入口：

```text
primary_experiments/01_channel_model_validation.py
```

验证：

- AWGN 参考链路；
- Gamma–Gamma 随机采样；
- 理论矩与样本矩的一致性；
- 指向误差采样；
- 组合信道的基础统计性质。

目的：

> 在引入编码和交织之前，先确认信道随机变量的生成与理论假设一致。

---

## Task 02：未编码 FSO 基线

入口：

```text
primary_experiments/02_uncoded_fso_ber.py
```

建立未编码 OOK 链路的 BER 基线，用于后续判断：

- 编码是否产生有效增益；
- 湍流对链路性能的影响；
- 指向误差是否改变性能趋势；
- 后续结果是否偏离合理数量级。

---

## Task 03：RS 编码 AWGN 验证

入口：

```text
primary_experiments/03_rs_awgn_validation.py
```

在没有湍流衰落的 AWGN 环境中验证：

- RS(255,223) 编码；
- 解码往返一致性；
- 可纠正错误数量边界；
- 码率归一化；
- BER 随 `Eb/N0` 的基本趋势。

先在简单信道中验证 RS，可以避免将编解码错误误认为湍流影响。

---

## Task 04：RS 编码与交织实验

相关入口：

```text
04_rs_fso_interleaving.py
04_rs_fso_symbol_interleaving.py
04c_rs_fso_fixed_coherence_interleaving.py
04d_rs_fso_fixed_coherence32_interleaving.py
```

核心实验变量：

- 交织深度 `D`；
- 信道相干长度 `Lc`；
- `Eb/N0`；
- 湍流强度；
- 指向误差；
- RS 解码结果。

最终分析使用 Task 04C/04D，因为它们将物理相干长度固定，再单独扫描交织深度。

---

## Task 05：相干长度与交织深度分析

入口：

```text
primary_experiments/05_rs_coherence_depth_analysis.py
```

将 BER 曲线转化为：

- 目标 BER 下所需的 `Eb/N0`；
- 不同交织深度的收益；
- 交织深度的边际收益；
- 缓存需求；
- 延迟与性能之间的取舍。

该阶段关注的不只是“深交织性能更好”，还要回答：

> 交织深度增加到什么程度后，额外缓存和延迟只带来很小收益？

---

## Task 06：LDPC 编码 AWGN 验证

入口：

```text
primary_experiments/06_ldpc_awgn_validation.py
```

验证：

- LDPC(512,449) 结构；
- 校验矩阵约束；
- 编码与解码往返；
- 无噪声正确恢复；
- AWGN 中的性能趋势；
- 与 RS 接近的有效码率。

LDPC 码率约为 0.8770，与 RS(255,223) 的约 0.8745 接近，从而降低冗余差异对编码对比的影响。

---

## Task 07：LDPC 固定相干信道实验

相关入口：

```text
07_ldpc_fso_fixed_coherence_interleaving.py
07b_ldpc_fso_group_aware_refinement.py
```

Task 07B 在固定相干信道中运行 LDPC，并保留：

- 交织组；
- 共享衰落关系；
- 独立组边界；
- 分组误码统计；
- 分组感知置信区间。

最终对比使用 Task 07B。

---

## Task 08：RS 与 LDPC 最终对比

入口：

```text
primary_experiments/08_rs_ldpc_final_comparison.py
```

在接近码率和匹配信道条件下比较：

- BER；
- 所需 `Eb/N0`；
- 交织深度；
- 缓存需求；
- 解码复杂度；
- 不同湍流条件下的稳定性。

该实验避免只比较某一个最佳工作点，而是同时展示性能与工程代价。

---

## Task 09：匹配信道下的编码增益

入口：

```text
primary_experiments/09_uncoded_fixed_coherence_coding_gain.py
```

在相同固定相干信道下建立未编码对照，避免将以下差异错误计入编码增益：

- 信道模型不同；
- 相干长度不同；
- 随机样本不同；
- `Eb/N0` 定义不同。

---

## Task 10：湍流敏感性

入口：

```text
primary_experiments/10_turbulence_sensitivity.py
```

将选定的 RS、LDPC 和交织配置放入不同湍流强度，分析：

- 性能排序是否稳定；
- 最优交织深度是否变化；
- 编码方案是否对信道参数敏感；
- 在更强湍流下是否出现误码平台。

---

# 四个核心技术设计

## 1. 码率归一化的 RS/LDPC 公平比较

### Bad Case

如果直接在相同的发送比特信噪比下比较不同码率编码：

```text
RS 码率 ≠ LDPC 码率
        ↓
每个信息比特承担的能量不同
        ↓
冗余差异被误认为编码结构增益
```

### 解决方案

项目选择码率接近的编码：

```text
RS(255,223)    R ≈ 0.8745
LDPC(512,449)  R ≈ 0.8770
```

并以信息比特 `Eb/N0` 为统一对比口径，在噪声方差计算中显式考虑码率。

同时使用共同随机数进行部分对比，使不同方案尽量经历相同的信道随机条件，减少由样本差异造成的比较噪声。

---

## 2. 将交织深度与信道相干长度解耦

### Bad Case

如果改变交织深度时同时改变信道衰落结构：

```text
D 增加
  +
Lc 同时变化
  ↓
BER 发生改善
  ↓
无法判断改善来自交织还是信道变化
```

### 解决方案

最终实验固定物理相干长度 `Lc`，独立扫描交织深度 `D`：

```text
固定 Lc
   ↓
D = 1, 2, 4, ...
   ↓
比较相同信道条件下的 BER
```

这样能够单独分析：

- 交织带来的时间/空间分散收益；
- 信道相干性对编码帧的影响；
- `D/Lc` 关系；
- 深交织的边际收益。

---

## 3. 相关帧的分组感知置信区间

### Bad Case

同一个交织组中的多个帧可能共享相同衰落增益。

如果把所有帧或比特都当作独立 Bernoulli 样本：

```text
相关样本被当作独立样本
        ↓
有效样本数量被高估
        ↓
置信区间过窄
        ↓
结果看起来比实际更稳定
```

### 解决方案

项目保留独立衰落组边界，并同时考虑：

- Wilson 置信区间；
- 独立组数量；
- 组间误码变化；
- 共享衰落造成的相关性；
- 分组感知的不确定性。

最终统计不仅报告总误码数，还保留误码来自多少个独立衰落组。

---

## 4. 零观测误码不等于零 BER

### Bad Case

蒙特卡洛实验中没有观测到误码，并不能证明真实 BER 等于零。

```text
有限样本中观测错误数 = 0
        ≠
真实误码概率 = 0
```

### 解决方案

对于零观测误码点：

- 不直接将 BER 表示为数学意义上的零；
- 报告有限样本下的上界；
- 标记该点受仿真样本数量限制；
- 避免在对数坐标中给出虚假的无限编码增益。

---

# 独立复现实验

主实验与复现实验使用不同的实现路径。

```text
主实验
├── 库支持的 RS/LDPC 实现
├── 主信道生成函数
└── Task 01–10 分阶段分析

独立复现
├── 独立的信道与编解码路径
├── 条件派生随机种子
├── 独立结果文件
└── 独立分析与制图入口
```

两条路径不共享生成结果，因此独立复现检查的是：

> 不同实现是否得到一致的趋势，而不是用第二个脚本重新绘制同一份数据。

## 复现模块

| 文件 | 作用 |
|---|---|
| `fso_lib.py` | 独立信道与编码基础实现 |
| `calc_baseline.py` | 计算基准工作点 |
| `run_rs_groupaware.py` | RS 分组感知实验 |
| `run_task10.py` | 湍流敏感性复现 |
| `run_supplementary.py` | 补充实验 |
| `analyse_task10.py` | Task 10 独立分析 |
| `make_task_map.py` | 生成实验关系图 |
| `make_causal_chain.py` | 生成因果关系分析 |
| `make_fig4_1.py` | 生成对应实验图 |
| `make_fig4_3.py` | 生成对应实验图 |
| `make_fig67.py` | 生成 RS/LDPC 对比图 |
| `qfunc.py` | Q 函数与辅助计算 |

---

# 可复现性设计

## 固定随机种子

主实验使用显式主种子，并为不同条件生成独立随机流，避免多个实验条件意外复用完全相同的随机序列。

## 条件派生种子

复现实验不使用 Python 内置 `hash()`。

因为内置 Hash 可能随进程和 `PYTHONHASHSEED` 改变。

项目使用：

```text
CRC32(明确的实验条件字符串)
```

派生每个实验条件的确定性种子。

相同条件在不同进程中仍能得到相同种子。

## 断点续跑

长时间蒙特卡洛扫描会在每个条件完成后立即追加结果。

重新启动时：

```text
读取已有结果
    ↓
识别已完成条件
    ↓
跳过已完成点
    ↓
继续剩余实验
```

避免因进程中断重新运行整个参数空间。

## 分离仿真与分析

分析脚本读取已保存结果，不会在绘图时静默重新执行长时间仿真。

这使以下过程相互独立：

```text
生成数据
→ 冻结结果
→ 统计分析
→ 制图
```

---

# 快速运行

## 环境要求

推荐：

- Python 3.10–3.12
- Git
- 支持 NumPy/SciPy 的本地环境

## 1. 创建虚拟环境

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

主要依赖：

```text
NumPy
SciPy
Matplotlib
reedsolo
pyldpc
```


## 2. 从信道验证开始

完成上一步的依赖安装后，在仓库根目录执行：

```powershell
Set-Location primary_experiments
& "..\.venv\Scripts\python.exe" "01_channel_model_validation.py"
```

运行未编码基线：

```powershell
& "..\.venv\Scripts\python.exe" "02_uncoded_fso_ber.py"
```

运行 RS AWGN 验证：

```powershell
& "..\.venv\Scripts\python.exe" "03_rs_awgn_validation.py"
```

运行 LDPC AWGN 验证：

```powershell
& "..\.venv\Scripts\python.exe" "06_ldpc_awgn_validation.py"
```

## 3. 运行独立复现入口

```powershell
Set-Location ..\replication
& "..\.venv\Scripts\python.exe" "fso_lib.py"
& "..\.venv\Scripts\python.exe" "calc_baseline.py" "Moderate"
```

正式 Task 04–10 包含蒙特卡洛参数扫描，运行时间明显高于组件验证任务。建议先检查脚本中的参数、随机种子和输出目录，再启动完整实验。

---

# 推荐实验顺序

第一次阅读或运行时，建议按照以下顺序：

```text
01_channel_model_validation.py
    ↓
02_uncoded_fso_ber.py
    ↓
03_rs_awgn_validation.py
    ↓
06_ldpc_awgn_validation.py
    ↓
04c/04d RS 固定相干实验
    ↓
07b LDPC 分组感知实验
    ↓
08_rs_ldpc_final_comparison.py
    ↓
09_uncoded_fixed_coherence_coding_gain.py
    ↓
10_turbulence_sensitivity.py
    ↓
replication/
```

这样任何异常都可以先在更小的模型边界中定位。

---

# Bad Case 与工程闭环

| 编号 | 发现的问题 | 影响 | 工程改进 |
|---|---|---|---|
| BC-001 | 不同码率使用不一致的 `Eb/N0` | 冗余差异被误认为编码增益 | 信息比特口径与码率感知噪声 |
| BC-002 | 交织深度与信道相干长度混合变化 | 无法分离设计变量影响 | 固定 `Lc`，独立扫描 `D` |
| BC-003 | 将相关帧当作独立样本 | 置信区间过度乐观 | 保留独立组边界并计算分组区间 |
| BC-004 | `uint8` 数组直接相减 | 负数回绕并污染误码统计 | 比较前转换为有符号整数 |
| BC-005 | 零观测错误直接写成零 BER | 有限样本被误认为零概率 | 报告有限样本上界 |
| BC-006 | 使用 Python `hash()` 派生种子 | 不同进程结果不一致 | 使用 CRC32 条件种子 |
| BC-007 | 长仿真结束时才保存结果 | 中断后丢失全部进度 | 每个条件完成后追加并支持跳过 |

完整记录见 [Bad Cases](docs/BAD_CASES.md)。

---

# CI 与公开源码检查

GitHub Actions 执行两类检查。

## 1. 公开源码边界验证

`tools/validate_repository.py` 会拒绝：

- PDF、Word、PPT 等文档；
- CSV、NPZ 等生成数据；
- PNG、JPG、SVG 等生成图片；
- 意外二进制文件；
- 私钥；
- AWS Access Key；
- GitHub Token；
- API Key 特征字符串。

这样可以保证公开仓库保持为干净的代码复现包。

## 2. 全量源码编译检查

CI 对以下目录执行：

```text
primary_experiments/
replication/
tools/
```

并编译全部 Python 模块，检查：

- 语法错误；
- 不完整文件；
- 导入阶段的基础问题；
- 公开代码树是否可被 Python 解析。

CI 不运行完整蒙特卡洛扫描，避免将长时间研究实验作为每次提交的常规流水线任务。

---

# 工程证据索引

| 想了解的内容 | 对应文件 |
|---|---|
| Task 01–10 的依赖关系 | [实验地图](docs/EXPERIMENT_MAP.md) |
| 关键模型和统计取舍 | [工程日志](docs/ENGINEERING_LOG.md) |
| 实验过程中发现的问题 | [Bad Cases](docs/BAD_CASES.md) |
| 主实验代码 | [primary_experiments](primary_experiments/) |
| 独立复现代码 | [replication](replication/) |
| 公开源码检查 | [validate_repository.py](tools/validate_repository.py) |
| Python 依赖 | [requirements.txt](requirements.txt) |
| CI 配置 | [GitHub Actions](.github/workflows/ci.yml) |

---

# 仓库结构

```text
fso-coding-interleaving-simulator/
├── primary_experiments/
│   ├── 01_channel_model_validation.py
│   ├── 02_uncoded_fso_ber.py
│   ├── 03_rs_awgn_validation.py
│   ├── 04*_rs_fso_*.py
│   ├── 05_rs_coherence_depth_analysis.py
│   ├── 06_ldpc_awgn_validation.py
│   ├── 07*_ldpc_fso_*.py
│   ├── 08_rs_ldpc_final_comparison.py
│   ├── 09_uncoded_fixed_coherence_coding_gain.py
│   └── 10_turbulence_sensitivity.py
├── replication/
│   ├── fso_lib.py
│   ├── calc_baseline.py
│   ├── run_rs_groupaware.py
│   ├── run_task10.py
│   ├── run_supplementary.py
│   └── make_*.py
├── docs/
│   ├── EXPERIMENT_MAP.md
│   ├── ENGINEERING_LOG.md
│   └── BAD_CASES.md
├── tools/
│   └── validate_repository.py
├── requirements.txt
└── .github/workflows/ci.yml
```

---

# 项目证明的工程能力

本项目技术亮点：

- 建立自由空间光通信链路模型；
- 使用 Gamma–Gamma 分布模拟大气湍流；
- 建模指向误差和固定相干块衰落；
- 实现未编码、RS 和 LDPC 链路仿真；
- 使用信息比特 `Eb/N0` 进行码率归一化比较；
- 将交织深度与物理相干长度独立控制；
- 识别相关样本对置信区间的影响；
- 使用有限样本上界处理零观测误码；
- 使用固定随机种子和共同随机数提高可复现性；
- 使用断点续跑保护长时间蒙特卡洛实验；
- 通过独立实现路径检查主要实验趋势；
- 将研究脚本整理为可阅读、可执行和可检查的工程仓库。

---

## 结果解释范围

该仓库提供完整的仿真源码、实验链、统计方法和独立复现入口。具体数值结果由使用者在选定参数、随机种子和蒙特卡洛预算下运行生成。

不同运行环境、样本数量和停止条件可能影响置信区间宽度，因此实验对比应同时保留参数、随机种子、有效样本数和统计区间。

## 开源许可

本项目使用 [MIT License](LICENSE)。
