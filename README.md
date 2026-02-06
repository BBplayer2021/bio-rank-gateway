# Bio-Rank Gateway v13.0

A fully automated bioinformatics tool ranking portal deployed on GitHub Pages.

## Features

### 1. Automated Pipeline
- Weekly automated data collection (every Sunday at 0:00 UTC)
- GitHub Actions workflow for CI/CD
- Automatic deployment to GitHub Pages

### 2. Data Enrichment
- **Install Command Detection**: Regex matching for `conda install`, `pip install`, `docker pull`, `git clone`
- **Preview Image Extraction**: Scans README for workflow/report/plot images
- **Badge Generation**: Shields.io badge URLs for each project

### 3. Frontend Portal
- Modern card-based design
- Left sidebar for omics category navigation
- Dual-track switching (Pipeline/Utility)
- Quick Install code blocks
- Giscus comment system integration

### 4. Notification System
- Tracks Top 3 rankings
- Detects new entries
- Outputs badge Markdown for manual notification

## Project Structure

```
bio-rank-gateway/
├── .github/workflows/   # GitHub Actions automation
│   └── main.yml
├── scripts/             # Python crawler and algorithm
│   └── bio_rank_gateway.py
├── data/                # JSON data storage
│   ├── ranking_report.json
│   └── ranking_history.json
├── docs/                # GitHub Pages deployment
│   ├── index.html
│   ├── data/
│   └── assets/
└── README.md
```

## Scoring Formulas

### Pipeline Score
```
S = 5 × log10(Stars) + Weekly_Growth × 2 + Env_Bonus(15) + Paper_Bonus(5) + Zombie_Penalty(0.5)
```

### Utility Score
```
S = 8 × log10(Stars) + Weekly_Growth × 2 + Paper_Bonus(5) + Zombie_Penalty(0.5)
```

## Setup

### 1. Fork this repository

### 2. Enable GitHub Pages
- Go to Settings > Pages
- Set source to `gh-pages` branch

### 3. Configure Giscus (Optional)
1. Enable GitHub Discussions in your repository
2. Install Giscus app: https://giscus.app/
3. Update the Giscus configuration in `docs/index.html`

### 4. Set up GitHub Token
- The workflow uses `GITHUB_TOKEN` automatically provided by GitHub Actions
- For higher API rate limits, you can add a personal access token

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

## Contributing

Issues and Pull Requests are welcome!
