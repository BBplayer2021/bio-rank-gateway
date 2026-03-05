<p align="center">
  <h1 align="center">Bio-Rank Gateway</h1>
  <p align="center"><strong>Find the best, code the rest.</strong></p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python 3.9+">
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/actions"><img src="https://github.com/BBplayer2021/bio-rank-gateway/actions/workflows/main.yml/badge.svg" alt="GitHub Actions"></a>
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/stargazers"><img src="https://img.shields.io/github/stars/BBplayer2021/bio-rank-gateway?style=social" alt="GitHub Stars"></a>
    <a href="https://github.com/BBplayer2021/bio-rank-gateway/network/members"><img src="https://img.shields.io/github/forks/BBplayer2021/bio-rank-gateway?style=social" alt="GitHub Forks"></a>
  </p>
  <p align="center">
    <a href="https://bbplayer2021.github.io/bio-rank-gateway/"><img src="https://img.shields.io/badge/🔍_Live_Demo-Visit_Site-ff6b6b?style=for-the-badge" alt="Live Demo"></a>
  </p>
  <p align="center">
    <b>[English]</b> Real-time monitoring of global bioinformatics tool vitality.<br>
    <a href="README_CN.md"><b>[中文]</b></a> 实时监测全球生信工具和流程的生命力，拒绝"生信考古"。
  </p>
</p>

A fully automated bioinformatics tool ranking portal deployed on GitHub Pages.

## 🌟 Why Bio-Rank Gateway?

In the fast-evolving field of bioinformatics, finding the most reliable and active tool is a challenge. Bio-Rank Gateway solves the "Discovery Problem" by providing a data-driven, zero-bias leaderboard. We help you skip the "zombie" repos and focus on the gold standards.

- **Live & Real-time**: No more static, outdated lists.
- **Pure Bioinfo**: Noise-free filtering using domain-specific algorithms.
- **Evidence-based**: Ranking based on growth, maintenance, and community impact.

## Features

### 1. 🔥 Bio-Omics Heat Index (v3.0)
- **Five-Dimension Model**: Academic(30%) + Preprint(20%) + Tech(25%) + Funding(15%) + Community(10%)
- **10 Omics Categories**: Genomics, Transcriptomics, Metagenomics, Single-cell, Epigenetics, Proteomics, Metabolomics, Spatial Omics, Multi-omics, **BioAI** (NEW)
- **23 Sub-fields**: Covering core, advanced, applied, medical, and technology tracks
- **Rank Change Tracking**: TIOBE-style position change indicator (▲/▼) showing movement between reporting periods
- **YoY Growth**: Rolling 12-month window comparison for trend analysis
- **Momentum Labels**: Rising Star / Hot / Growing / Stable / Cooling
- **Interactive Radar Charts**: Five-dimension breakdown on click-to-expand detail panels
- Data sources: PubMed, bioRxiv, GitHub, Bioconductor, PyPI, NIH RePORTER, Google Trends, Semantic Scholar, OpenAlex

### 2. 🤖 Automated Pipeline
- Weekly automated data collection (every Sunday at 0:00 UTC)
- GitHub Actions workflow for CI/CD
- Automatic deployment to GitHub Pages

### 3. 💎 Data Enrichment
- **Install Command Detection**: Regex matching for `conda install`, `pip install`, `docker pull`, `git clone`
- **Preview Image Extraction**: Scans README for workflow/report/plot images
- **Badge Generation**: Shields.io badge URLs for each project

### 4. 🎨 Frontend Portal
- Modern card-based design
- Left sidebar for omics category navigation (BioAI featured with HOT badge)
- Dual-track switching (Pipeline/Utility)
- Quick Install code blocks
- Giscus comment system integration

### 5. 📢 Notification System
- Tracks Top 3 rankings
- Detects new entries
- Outputs badge Markdown for manual notification

## Project Structure

```
bio-rank-gateway/
├── .github/workflows/   # GitHub Actions automation
│   └── main.yml
├── scripts/             # Python crawler and algorithm
│   ├── bio_rank_gateway.py      # Tool ranking engine
│   └── omics_tiobe_index.py     # Omics Heat Index engine (v3.0)
├── data/                # JSON data storage
│   ├── ranking_report.json
│   └── ranking_history.json
├── docs/                # GitHub Pages deployment
│   ├── index.html
│   ├── data/
│   │   ├── ranking_report.json
│   │   └── omics_index.json     # Heat Index data (23 fields)
│   └── assets/
└── README.md
```

## Scoring Formulas

> ### Pipeline Score
> ```
> S = 5 × log10(Stars) + Weekly_Growth × 2 + Env_Bonus(15) + Paper_Bonus(5) + Maintenance-Aware Scoring(0.5)
> ```
> - **Stars**: GitHub star count (logarithmic scale)
> - **Weekly_Growth**: Star increase in the past 7 days
> - **Env_Bonus**: +15 if Docker/Conda support detected
> - **Paper_Bonus**: +5 if an associated publication is found
> - **Maintenance-Aware Scoring (Anti-Zombie)**: ×0.5 penalty for repos inactive > 180 days
>
> ### Utility Score
> ```
> S = 8 × log10(Stars) + Weekly_Growth × 2 + Paper_Bonus(5) + Maintenance-Aware Scoring(0.5)
> ```

## 🛠️ Get Started in 3 Minutes

1. **Fork** this repository to your own GitHub account.
2. **Enable GitHub Pages**: Go to `Settings > Pages` and set the source to the `gh-pages` branch.
3. **Watch it run**: The GitHub Actions workflow will automatically collect data and deploy every Sunday.

### Optional: Configure Giscus Comments
1. Enable GitHub Discussions in your repository.
2. Install the Giscus app: https://giscus.app/
3. Update the Giscus configuration in `docs/index.html`.

### Optional: Boost API Rate Limits
- The workflow uses `GITHUB_TOKEN` automatically provided by GitHub Actions.
- For higher limits, add a personal access token as a repository secret.

## Local Development

```bash
# Install dependencies
pip install requests

# Run data collection
cd scripts
python bio_rank_gateway.py

# Serve locally
cd ../docs
python -m http.server 8000
```

## API Rate Limits

- Without token: 60 requests/hour
- With `GITHUB_TOKEN`: 5000 requests/hour

## License

MIT License

## 🤝 Support & Feedback

If you find this project useful, please give us a ⭐️! It helps more researchers discover the right tools.

- Found a bug? [Open an Issue](https://github.com/BBplayer2021/bio-rank-gateway/issues)
- Have an idea? [Start a Discussion](https://github.com/BBplayer2021/bio-rank-gateway/discussions)

## Contributing

Issues and Pull Requests are welcome!
