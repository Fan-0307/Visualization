# VLM Attention Analysis

面向 VQA 任务的视觉语言模型可视分析系统。系统用于比较 Qwen2-VL、LLaVA、CLIP、BLIP2 等模型在不同问题类型上的表现，并通过样本诊断、错误模式、注意力扩散、逐层演化等视图辅助分析模型行为差异。

本仓库只维护前端系统代码和轻量级索引/统计文件。图片、attention JSON、逐层演化文件、热力图图片等大体积运行数据不随 Git 仓库提交，需要在本地单独解压或放置。

## 功能概览

- 模型对比：展示整体准确率、按问题类型准确率、关键摘要结论和交互式柱状图。
- 样本诊断：通过 Error Matrix 定位全错样本、部分错样本和高分歧样本，并展示末层视觉证据代理图。
- 错误模式：按模型查看答对/答错样本分布和问题类型分布。
- 注意力扩散：分析注意力从局部集中到扩散/转移的模式，并支持跨视图联动。
- 逐层演化：展示不同模型逐层视觉表征响应变化、演化曲线和相邻层差异。
- 聚合注意力与 Token 注意力：作为补充视图，查看 token 级注意力和聚合热力图。

注意：由于不同 VLM 架构的注意力定义并不完全一致，系统中的跨模型热力图应理解为“视觉证据/表征响应代理”，适合辅助诊断和比较，不应解释为严格因果归因或模型真实注视的像素区域。

## 技术栈

- Vue 3
- Vite 5
- D3.js
- Node.js / npm

## 环境要求

建议使用：

```bash
node --version   # 建议 >= 18
npm --version
```

如果需要重新运行数据处理脚本，还需要 Python 环境和 `requirements.txt` 中的依赖；仅运行前端系统不需要安装 Python 依赖。

## 安装依赖

在项目根目录执行：

```bash
cd /path/to/Visualization
npm install
```

仓库中同时存在 `pnpm-lock.yaml` 和 `package-lock.json`。当前最直接的运行方式是使用 npm；如果团队统一使用 pnpm，也可以自行切换，但不要混用安装器反复改动 lockfile。

## 准备运行数据

大体积数据不提交到 GitHub。系统运行至少需要以下目录或文件：

```text
Visualization/
├── public/
│   ├── data/
│   │   ├── attn/
│   │   └── layer_evo/
│   └── img/
└── src/
    └── data/
        ├── processed_data/
        │   ├── results.json
        │   ├── summary.json
        │   └── photos/
        ├── attn_index.json
        ├── layer_evo_index.json
        ├── layer_evo_stats.json
        └── layer_stats.json
```

如果拿到的是完整数据压缩包，例如：

```text
visualization_data.tar.gz
```

可在项目根目录解压：

```bash
cd /path/to/Visualization
tar -xzf visualization_data.tar.gz
```

如果拿到的是分片文件，先合并再解压：

```bash
cat visualization_data.tar.gz.part-* > visualization_data.tar.gz
tar -xzf visualization_data.tar.gz
```

如果需要手动放置，确保这些路径存在：

```text
public/data/attn/                  token-level attention JSON
public/data/layer_evo/             layer evolution result.json files
public/img/                        COCO/VQA images
src/data/processed_data/photos/    model heatmap images
src/data/processed_data/results.json
src/data/processed_data/summary.json
```

## 启动开发服务器

```bash
cd /path/to/Visualization
npm run dev -- --host 0.0.0.0
```

Vite 默认会给出类似下面的地址：

```text
http://localhost:5173/
```

在 JupyterHub 或远程服务器环境中，请使用平台提供的端口转发/代理地址访问 5173 端口。

## 生产构建

```bash
npm run build
```

构建产物会输出到：

```text
dist/
```

如果只修改文档，不需要重新构建；如果修改 `src/` 代码，提交前建议至少运行一次 `npm run build`。

## 项目结构

```text
Visualization/
├── data/                         数据预处理和 attention 提取脚本
├── public/                       Vite 静态资源目录，大体积运行数据放在这里
│   ├── data/attn/                token 级 attention JSON
│   ├── data/layer_evo/           逐层演化数据
│   └── img/                      原始图像
├── src/
│   ├── components/               各可视化视图组件
│   ├── data/                     轻量级索引、统计和 processedData 入口
│   ├── store/selection.js        跨视图联动状态
│   ├── utils/                    可视化主题等工具
│   ├── App.vue                   顶层布局和导航
│   └── main.js                   Vue 入口
├── visualization_data_package/   数据包说明占位目录
├── package.json
├── vite.config.js
└── README.md
```

## 主要数据说明

- `src/data/processed_data/results.json`：模型预测结果、问题、答案、问题类型等主表数据。
- `src/data/processed_data/photos/`：用于样本诊断的模型热力图图片。
- `public/data/attn/`：token 级 attention JSON，供聚合注意力和 Token 注意力视图使用。
- `public/data/layer_evo/`：逐层演化视图运行时读取的 `result.json`。
- `public/img/`：原始图片。系统会兼容 `COCO_val2014_000000xxxxxx.jpg` 和 `{image_id}.jpeg` 等不同命名方式。
- `src/data/*_index.json`、`src/data/*_stats.json`：前端索引和统计文件。

## 视图说明

### 模型对比

入口视图，适合答辩开场使用。它展示总体准确率、按问题类型的准确率差异，并用摘要卡片给出最佳模型、最弱模型和关键差异。

### 样本诊断

核心诊断视图。Error Matrix 支持按问题类型、错误数量和预测分歧度筛选排序。选中样本后显示问题、标准答案、模型预测、诊断摘要和末层视觉证据代理图。

### 错误模式

按模型聚合答对和答错样本，帮助快速观察某个模型在哪些问题类型或样本上更容易失败。

### 注意力扩散

展示注意力扩散/聚集模式，并支持和主流程视图联动，用于定位典型扩散样本和进一步钻取。

### 逐层演化

展示模型逐层视觉表征响应的变化，包括当前层热力图、演化曲线、相邻层差异等。该视图适合说明模型内部表征从浅层到深层的变化过程。

### 更多视图

`聚合注意力` 和 `Token 注意力` 使用另一条 token-level attention 数据线，样本覆盖和主流程不完全一致，因此被收纳在“更多”菜单中。

## 模型名称规范

界面中默认使用简称：

```text
Qwen2-VL
LLaVA
CLIP
BLIP2
```

鼠标悬停时显示完整工程名：

```text
Qwen2-VL-7B-Instruct
llava-1.5-7b-hf
clip-vit-base-patch32
blip2-opt-2.7b
```

## 常见问题

### 页面能打开，但图片或热力图不显示

优先检查：

```bash
ls public/img | head
ls public/data/layer_evo
ls src/data/processed_data/photos
```

如果图片命名和数据里的 `image_path` 不一致，逐层演化视图已经内置多路径回退；其他视图仍需要对应数据目录完整。

### 聚合注意力或 Token 注意力样本较少

这两个视图依赖 `public/data/attn/` 和 `src/data/attn_index.json`。它们与主流程 `results.json` 的样本线可能不同，因此属于补充分析视图。

### Tab 中的证据图是否等同于真实注意力

不完全等同。对于生成式 VLM 和 BLIP2 类结构，最后一层视觉表征已经不一定对应原始图像 patch。系统将其作为视觉证据/表征响应代理，用于对比和诊断，而不是严格的因果解释。

### GitHub 上为什么没有数据

数据目录体积较大，已经通过 `.gitignore` 排除。请通过课程提交包、服务器共享目录或数据压缩包单独分发。

## 提交前检查

建议提交代码前运行：

```bash
npm run build
git status --short
```

确认不要误提交以下大体积数据：

```text
public/
visualization_data.tar.gz
data_bundle.tar
src/data/processed_data/photos/
```

## 快速命令

```bash
# 安装
npm install

# 开发运行
npm run dev -- --host 0.0.0.0

# 构建
npm run build
```
