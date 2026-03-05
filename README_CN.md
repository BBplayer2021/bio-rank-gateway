<p align="center">
  <h1 align="center">Bio-Rank Gateway</h1>
  <p align="center"><strong>Find the best, code the rest. | 找到最好的，专注写代码。</strong></p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/actions"><img src="https://github.com/BBplayer2021/bio-rank-gateway/actions/workflows/main.yml/badge.svg" alt="GitHub Actions"></a>
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/stargazers"><img src="https://img.shields.io/github/stars/BBplayer2021/bio-rank-gateway?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/network/members"><img src="https://img.shields.io/github/forks/BBplayer2021/bio-rank-gateway?style=social" alt="GitHub Forks"></a>
  </p>
  <p align="center">
    <a href="https://bbplayer2021.github.io/bio-rank-gateway/"><img src="https://img.shields.io/badge/🔍_在线演示-访问站点-ff6b6b?style=for-the-badge" alt="Live Demo"></a>
  </p>
  <p align="center">
    <a href="README.md"><b>[English]</b></a> | <b>[中文]</b>
  </p>
</p>

全自动生物信息学工具排行门户，部署于 GitHub Pages。实时监测全球生信工具和流程的生命力，拒绝"生信考古"。

---

## 🌟 为什么选择 Bio-Rank Gateway？

生物信息学工具迭代极快，找到**真正活跃、可靠**的工具是一大痛点。Bio-Rank Gateway 通过数据驱动、零偏见的排行榜，帮你跳过"僵尸仓库"，直达金标准工具。

- **实时更新**：每周自动采集，告别过时的静态列表
- **纯生信聚焦**：领域专用算法，过滤噪声
- **循证排名**：基于增长趋势、维护状态、社区影响力的综合评分

---

## 🔥 核心功能

### 1. 组学热度指数 (Bio-Omics Heat Index v3.0)

业界首创的**五维热度模型**，量化组学领域发展态势：

| 维度 | 权重 | 数据来源 |
|------|------|----------|
| 学术影响力 (Academic) | 30% | PubMed YoY、OpenAlex 引用 |
| 预印本活跃度 (Preprint) | 20% | bioRxiv 发文量与增长率 |
| 技术生态 (Tech) | 25% | GitHub、Bioconductor、PyPI |
| 科研基金 (Funding) | 15% | NIH RePORTER |
| 社区关注度 (Community) | 10% | Google Trends、Semantic Scholar |

**亮点特性：**
- **10大组学类别**：Genomics、Transcriptomics、Metagenomics、Single-cell、Epigenetics、Proteomics、Metabolomics、Spatial Omics、Multi-omics、**BioAI** (NEW)
- **23个子领域**全覆盖：核心组学、高级组学、应用组学、医学组学、前沿技术
- **排名变动追踪**：类似 TIOBE 的位次变化指示器（▲上升 / ▼下降 / — 不变）
- **动能标签**：Rising Star / Hot / Growing / Stable / Cooling
- **交互式雷达图**：点击展开五维评分细节

### 2. 🤖 自动化流水线
- 每周日 UTC 0:00 自动采集数据
- GitHub Actions CI/CD 工作流
- 自动部署到 GitHub Pages

### 3. 💎 数据增强
- **安装命令检测**：自动识别 `conda install`、`pip install`、`docker pull`、`git clone`
- **预览图提取**：扫描 README 中的流程图/结果图
- **徽章生成**：为每个项目生成 Shields.io 徽章

### 4. 🎨 前端门户
- 现代卡片式设计
- 左侧边栏组学分类导航（BioAI 置顶 + HOT 标记）
- Pipeline / Utility 双赛道切换
- 一键安装代码块
- Giscus 评论系统

### 5. 📢 排名通知
- 追踪 Top 3 排名变化
- 检测新上榜项目
- 输出徽章 Markdown

---

## 📁 项目结构

```
bio-rank-gateway/
├── .github/workflows/       # GitHub Actions 自动化
│   └── main.yml
├── scripts/                 # Python 爬虫与算法
│   ├── bio_rank_gateway.py      # 工具排名引擎
│   └── omics_tiobe_index.py     # 组学热度指数引擎 (v3.0)
├── data/                    # JSON 数据存储
│   ├── ranking_report.json
│   └── ranking_history.json
├── docs/                    # GitHub Pages 部署
│   ├── index.html
│   ├── data/
│   │   ├── ranking_report.json
│   │   └── omics_index.json     # 热度指数数据 (23 字段)
│   └── assets/
├── README.md                # English
└── README_CN.md             # 中文
```

---

## 📊 评分公式

> ### Pipeline 赛道评分
> ```
> S = 5 × log10(Stars) + 周增长 × 2 + 环境加分(15) + 论文加分(5) + 维护感知(0.5)
> ```
> - **Stars**：GitHub 星标数（对数尺度）
> - **周增长**：过去 7 天的星标增量
> - **环境加分**：检测到 Docker/Conda 支持 +15 分
> - **论文加分**：关联论文 +5 分
> - **维护感知（反僵尸）**：超过 180 天无更新 ×0.5 惩罚
>
> ### Utility 赛道评分
> ```
> S = 8 × log10(Stars) + 周增长 × 2 + 论文加分(5) + 维护感知(0.5)
> ```

---

## 🛠️ 3 分钟快速上手

1. **Fork** 本仓库到你的 GitHub 账号
2. **启用 GitHub Pages**：进入 `Settings > Pages`，设置 source 为 `gh-pages` 分支
3. **坐等运行**：GitHub Actions 每周日自动采集数据并部署

### 可选：配置 Giscus 评论
1. 在仓库中启用 GitHub Discussions
2. 安装 Giscus 应用：https://giscus.app/
3. 更新 `docs/index.html` 中的 Giscus 配置

### 可选：提升 API 频率限制
- 工作流默认使用 GitHub Actions 提供的 `GITHUB_TOKEN`
- 如需更高限制，可添加 Personal Access Token 作为仓库 Secret

---

## 💻 本地开发

```bash
# 安装依赖
pip install requests

# 运行数据采集
cd scripts
python bio_rank_gateway.py

# 运行组学热度指数
python omics_tiobe_index.py

# 本地预览
cd ../docs
python -m http.server 8000
```

---

## 📡 API 频率限制

| 方式 | 限制 |
|------|------|
| 无 Token | 60 次/小时 |
| 使用 `GITHUB_TOKEN` | 5,000 次/小时 |

---

## 📄 License

MIT License

---

## 🤝 支持与反馈

如果觉得项目有帮助，请给个 Star 支持一下！帮助更多研究者发现合适的工具。

- 发现 Bug？[提交 Issue](https://github.com/BBplayer2021/bio-rank-gateway/issues)
- 有新想法？[发起 Discussion](https://github.com/BBplayer2021/bio-rank-gateway/discussions)
- 欢迎 Pull Request！
