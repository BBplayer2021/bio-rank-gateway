#!/usr/bin/env python3
"""
Bio-Omics TIOBE Index
=====================
计算生信各组学领域的热度指数，类似 TIOBE 编程语言排行榜。

数据源:
  - PubMed (NCBI Entrez, 通过 Biopython): 2025-2026 年各领域文献新增量
  - bioRxiv: 过去 6 个月预印本活跃度（通过 PubMed biorxiv 索引 + bioRxiv API 总量）
  - GitHub Search API: 相关项目 Star 与新增仓库数

评分公式:
  Score = (学术文献 * 0.5) + (预印本 * 0.3) + (技术开发 * 0.2)
  归一化后得到各领域市场占有率 (Percentage Share %)。

趋势: 读取已有 docs/data/omics_index.json，对比上周数据计算 Change %。

输出: docs/data/omics_index.json

依赖 (需单独安装):
  pip install biopython requests
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

# 可选: Biopython Entrez（推荐）
try:
    from Bio import Entrez
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# 可选: requests（用于 GitHub / bioRxiv，否则用 urllib）
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "omics_index.json"

# 建议设置 GITHUB_TOKEN 以提高 GitHub API 配额，避免 403 rate limit
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "bio-rank-gateway@localhost")

# API 速率限制 (秒)
NCBI_DELAY = 0.4       # NCBI 建议无 key 时 ≤3 req/s
BIORXIV_DELAY = 1.0
GITHUB_DELAY = 2.0
API_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# 评分权重
WEIGHT_PUBMED = 0.50
WEIGHT_BIORXIV = 0.30
WEIGHT_GITHUB = 0.20

# NCBI
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# bioRxiv API: details 支持 interval 如 180d
BIORXIV_BASE = "https://api.biorxiv.org/details/biorxiv"

# GitHub
GITHUB_SEARCH = "https://api.github.com/search/repositories"

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.info


# ============================================================
# 核心词库: 20+ 组学领域，每领域配置精准关键词用于 API 检索
# ============================================================

OMICS_DICT = {
    # --- 基础组学 (由 Core 升级) ---
    "Genomics": {
        "keywords": [
            "genomics", "WGS", "genome assembly", "T2T-Genome", "Hi-C",
            "Haplotype genome", "variant calling", "VCF", "GWAS", "BSA"
        ],
        "category": "Core Omics",
    },
    "Transcriptomics": {
        "keywords": [
            "transcriptomics", "RNA-seq", "lncRNA", "circRNA", "miRNA",
            "full-length transcriptomics", "gene expression", "isoform"
        ],
        "category": "Core Omics",
    },
    "Proteomics": {
        "keywords": [
            "proteomics", "mass spectrometry", "protein identification",
            "TMT", "iTRAQ", "label-free quantification", "LC-MS", "DIA", "DDA"
        ],
        "category": "Core Omics",
    },
    "Metabolomics": {
        "keywords": [
            "metabolomics", "metabolic profiling", "untargeted metabolomics",
            "LC-MS metabolomics", "GC-MS", "metabolite", "MZmine", "XCMS"
        ],
        "category": "Core Omics",
    },
    # --- 进阶/热门 ---
    "Single-cell Omics": {
        "keywords": [
            "single-cell RNA-seq", "scRNA-seq", "single-cell genomics",
            "single-cell multiomics", "10x Genomics", "Scanpy", "Seurat"
        ],
        "category": "Advanced",
    },
    "Spatial Omics": {
        "keywords": [
            "spatial transcriptomics", "spatial proteomics", "spatial metabolomics",
            "Visium", "Stereo-seq", "MALDI-imaging"
        ],
        "category": "Advanced",
    },
    "Pangenomics": {
        "keywords": [
            "pangenomics", "pangenome", "graph genome", "comparative genomics"
        ],
        "category": "Advanced",  # 从 Technology 升级
    },
    # --- 临床与转化医学 (新增分类) ---
    "Clinical Genomics": {
        "keywords": [
            "WES", "exome sequencing", "liquid biopsy", "ctDNA",
            "cancer genomics", "rare disease genomics", "precision oncology"
        ],
        "category": "Medical",
    },
    # --- 微生物与环境 ---
    "Metagenomics": {
        "keywords": [
            "metagenomics", "shotgun metagenomics", "MAGs",
            "16S/18S/ITS", "virome", "culturomics"
        ],
        "category": "Advanced",
    },
    "Epigenomics": {
        "keywords": [
            "epigenomics", "DNA methylation", "ChIP-seq", "ATAC-seq",
            "Hi-C", "chromatin", "CUT&Tag", "bisulfite sequencing"
        ],
        "category": "Advanced",
    },
    "Lipidomics": {
        "keywords": [
            "lipidomics", "lipid profiling", "lipidome", "phospholipid",
            "fatty acid", "lipid metabolism"
        ],
        "category": "Advanced",
    },
    "Multi-omics": {
        "keywords": [
            "multi-omics", "integrative omics", "multi-omics integration",
            "pan-omics", "multiomic"
        ],
        "category": "Advanced",
    },
    # --- 应用/技术 ---
    "Microbiome": {
        "keywords": [
            "microbiome", "gut microbiome", "16S rRNA", "microbiome analysis",
            "dysbiosis", "microbiota"
        ],
        "category": "Applied",
    },
    "Pharmacogenomics": {
        "keywords": [
            "pharmacogenomics", "drug response genomics", "precision medicine",
            "pharmacogenetics"
        ],
        "category": "Applied",
    },
    "Phylogenomics": {
        "keywords": [
            "phylogenomics", "phylogenetic", "comparative genomics",
            "molecular evolution", "species tree"
        ],
        "category": "Applied",
    },
    "Structural Biology": {
        "keywords": [
            "structural biology", "cryo-EM", "protein structure",
            "AlphaFold", "structure prediction"
        ],
        "category": "Applied",
    },
    "Immunogenomics": {
        "keywords": [
            "immunogenomics", "TCR sequencing", "BCR repertoire",
            "immune repertoire", "V(D)J"
        ],
        "category": "Applied",
    },
    "Synthetic Biology": {
        "keywords": [
            "synthetic biology", "genetic circuit", "genome engineering",
            "synthetic genome"
        ],
        "category": "Applied",
    },
    "Glycomics": {
        "keywords": [
            "glycomics", "glycoproteomics", "glycan", "glycosylation"
        ],
        "category": "Applied",
    },
    "Radiomics": {
        "keywords": [
            "radiomics", "radiomic features", "imaging biomarkers",
            "radiogenomics"
        ],
        "category": "Applied",
    },
    "Nutrigenomics": {
        "keywords": [
            "nutrigenomics", "nutritional genomics", "diet-gene",
            "nutrigenetics"
        ],
        "category": "Applied",
    },
    "Long-read Sequencing": {
        "keywords": [
            "long-read sequencing", "Nanopore", "PacBio", "HiFi",
            "long-read genome", "third-generation sequencing"
        ],
        "category": "Technology",
    },
    "CRISPR Genomics": {
        "keywords": [
            "CRISPR", "CRISPR screen", "genome editing", "CRISPR-Cas9",
            "CRISPR-Cas"
        ],
        "category": "Technology",
    },
}


# ============================================================
# 工具: 限速 + 重试的 HTTP 请求
# ============================================================

def _http_get(url: str, headers: Optional[dict] = None, timeout: int = API_TIMEOUT) -> Optional[dict]:
    """统一 HTTP GET，支持 requests 或 urllib，带重试与异常处理。"""
    headers = headers or {"User-Agent": "Bio-Omics-TIOBE-Index/1.0"}
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            if HAS_REQUESTS:
                r = requests.get(url, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            log("  Request attempt %d failed: %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
    if last_err:
        log("  All retries failed for: %s", url[:80])
    return None


def _rate_limit(seconds: float) -> None:
    time.sleep(seconds)


# ============================================================
# 数据抓取: PubMed (Biopython Entrez 优先)
# ============================================================

def _pubmed_esearch_term(keywords: list[str], year_start: int = 2025, year_end: int = 2026) -> str:
    """构建 PubMed 检索式：关键词 OR + 日期范围。"""
    term = " OR ".join(f'"{k}"' for k in keywords)
    date_filter = f'("{year_start}"[Date - Publication] : "{year_end}"[Date - Publication])'
    return f"({term}) AND {date_filter}"


def fetch_pubmed_count(query: str) -> int:
    """查询 PubMed 2025-2026 年文献量。优先使用 Biopython Entrez。"""
    if HAS_BIOPYTHON:
        try:
            Entrez.email = ENTREZ_EMAIL
            Entrez.retries = 2
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0, rettype="count")
            record = Entrez.read(handle)
            handle.close()
            count = record.get("Count", 0)
            return int(count) if count is not None else 0
        except Exception as e:
            log("  Entrez error: %s", e)
            return 0
    # Fallback: 直接 URL
    params = f"?db=pubmed&term={quote_plus(query)}&retmax=0&retmode=json"
    data = _http_get(NCBI_ESEARCH + params)
    if data and "esearchresult" in data:
        return int(data["esearchresult"].get("count", 0))
    return 0


def fetch_biorxiv_count(keywords: list[str]) -> int:
    """过去 6 个月各领域预印本量：通过 PubMed 预印本出版类型检索（Preprint[pt]）。
    注意：PubMed 中 biorxiv[filter] 无效，需用 \"Preprint\"[pt]。"""
    term = " OR ".join(f'"{k}"' for k in keywords)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_filter = (
        f'("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : '
        f'"{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
    )
    # PubMed 预印本用出版类型 "Preprint"[pt]，不是 biorxiv[filter]
    query = f'({term}) AND {date_filter} AND "Preprint"[pt]'
    return fetch_pubmed_count(query)


def fetch_biorxiv_total_last_six_months() -> int:
    """调用 bioRxiv API 获取过去 6 个月预印本总数（用于报告说明）。
    接口: /details/biorxiv/180d/0 ，返回 messages 中含 cursor/count。"""
    url = f"{BIORXIV_BASE}/180d/0"
    data = _http_get(url)
    if not data:
        return 0
    # messages 可能为 list 或 dict；常见为 [{"cursor":"0","count":"12345"}, ...]
    messages = data.get("messages") or data.get("message") or []
    if isinstance(messages, dict):
        messages = [messages]
    for m in messages:
        if isinstance(m, dict):
            count = m.get("count") or m.get("total")
            if count is not None:
                return int(count)
    return 0


# ============================================================
# 数据抓取: GitHub
# ============================================================

def fetch_github_activity(main_keyword: str) -> int:
    """相关关键词的 GitHub 活跃度：近 6 个月新仓库(stars>5) * 2 + 总匹配仓库(stars>10)。"""
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Bio-Omics-Index/1.0"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    new_repos = 0
    q_new = f"{main_keyword} bioinformatics created:>{six_months_ago} stars:>5"
    url_new = f"{GITHUB_SEARCH}?q={quote_plus(q_new)}&per_page=1"
    data = _http_get(url_new, headers=headers)
    if data:
        new_repos = data.get("total_count", 0)
    _rate_limit(GITHUB_DELAY)

    total_repos = 0
    q_total = f"{main_keyword} bioinformatics stars:>10"
    url_total = f"{GITHUB_SEARCH}?q={quote_plus(q_total)}&per_page=1"
    data2 = _http_get(url_total, headers=headers)
    if data2:
        total_repos = data2.get("total_count", 0)

    return new_repos * 2 + total_repos


# ============================================================
# 指数算法: 加权得分 + 归一化市场占有率
# ============================================================

def calculate_scores(raw_data: dict[str, dict]) -> list[dict[str, Any]]:
    """
    Score = (学术文献 * 0.5) + (预印本 * 0.3) + (技术开发 * 0.2)
    归一化得到 Percentage Share %，并排序赋 rank。
    """
    results = []
    total_score = 0.0
    for field, counts in raw_data.items():
        score = (
            WEIGHT_PUBMED * counts["pubmed"]
            + WEIGHT_BIORXIV * counts["biorxiv"]
            + WEIGHT_GITHUB * counts["github"]
        )
        results.append({
            "field": field,
            "category": counts.get("category", ""),
            "pubmed_count": counts["pubmed"],
            "biorxiv_count": counts["biorxiv"],
            "github_activity": counts["github"],
            "raw_score": round(score, 2),
        })
        total_score += score

    for item in results:
        item["share_pct"] = round(item["raw_score"] / total_score * 100, 2) if total_score > 0 else 0.0

    results.sort(key=lambda x: x["share_pct"], reverse=True)
    for i, item in enumerate(results, 1):
        item["rank"] = i
    return results


# ============================================================
# 趋势: 对比已有 omics_index.json 计算 Change %
# ============================================================

def load_previous_share(history_path: Path) -> dict[str, float]:
    """读取已有 omics_index.json 中最近一期 rankings 的 share_pct。"""
    if not history_path.exists():
        return {}
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        prev = {}
        for item in old.get("rankings", []):
            prev[item["field"]] = item.get("share_pct", 0)
        return prev
    except Exception as e:
        log("  Warning: could not load history %s: %s", history_path, e)
        return {}


def attach_trend(rankings: list[dict], history_path: Path) -> None:
    """在原 rankings 上就地添加 change / change_str。"""
    prev = load_previous_share(history_path)
    for item in rankings:
        prev_share = prev.get(item["field"])
        if prev_share is not None:
            change = round(item["share_pct"] - prev_share, 2)
            item["change"] = change
            item["change_str"] = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
        else:
            item["change"] = None
            item["change_str"] = "NEW"


# ============================================================
# 主流程: 采集 -> 计分 -> 趋势 -> 输出
# ============================================================

def collect_all_data() -> dict[str, dict]:
    """按 OMICS_DICT 依次拉取 PubMed、bioRxiv、GitHub 数据，严格限速。"""
    raw_data = {}
    n = len(OMICS_DICT)
    for i, (field, config) in enumerate(OMICS_DICT.items(), 1):
        keywords = config["keywords"]
        category = config.get("category", "")
        log("[%d/%d] %s", i, n, field)

        pubmed_query = _pubmed_esearch_term(keywords)
        pubmed_count = fetch_pubmed_count(pubmed_query)
        log("  PubMed: %s", pubmed_count)
        _rate_limit(NCBI_DELAY)

        biorxiv_count = fetch_biorxiv_count(keywords)
        log("  bioRxiv: %s", biorxiv_count)
        _rate_limit(NCBI_DELAY)

        github_activity = fetch_github_activity(keywords[0])
        log("  GitHub: %s", github_activity)
        _rate_limit(GITHUB_DELAY)

        raw_data[field] = {
            "pubmed": pubmed_count,
            "biorxiv": biorxiv_count,
            "github": github_activity,
            "category": category,
        }
    return raw_data


def generate_omics_index() -> dict[str, Any]:
    """生成 Bio-Omics TIOBE Index 并写入 docs/data/omics_index.json。"""
    log("=" * 60)
    log("Bio-Omics TIOBE Index Generator")
    log("=" * 60)
    if not HAS_BIOPYTHON:
        log("Warning: Biopython not installed. Using URL fallback for PubMed. pip install biopython")

    # 1) 数据采集
    log("\n[Phase 1] Collecting data (PubMed, bioRxiv, GitHub)...")
    raw_data = collect_all_data()

    # 2) 计分与归一化
    log("\n[Phase 2] Calculating scores and share %%...")
    rankings = calculate_scores(raw_data)

    # 3) 趋势
    log("\n[Phase 3] Trend (compare with existing %s)...", OUTPUT_PATH)
    attach_trend(rankings, OUTPUT_PATH)

    # 4) 可选: bioRxiv 总量（仅说明用）
    biorxiv_total = 0
    try:
        _rate_limit(BIORXIV_DELAY)
        biorxiv_total = fetch_biorxiv_total_last_six_months()
    except Exception as e:
        log("  Skip biorxiv total: %s", e)

    report = {
        "generated_at": datetime.now().isoformat(),
        "version": "1.0",
        "methodology": {
            "weights": {"pubmed": WEIGHT_PUBMED, "biorxiv": WEIGHT_BIORXIV, "github": WEIGHT_GITHUB},
            "pubmed_period": "2025-01-01 to 2026-12-31",
            "biorxiv_period": "last 6 months (PubMed Preprint[pt])",
            "github_period": "last 6 months new repos (stars>5) + total (stars>10)",
            "biorxiv_total_6m": biorxiv_total,
        },
        "total_fields": len(rankings),
        "rankings": rankings,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("\nSaved: %s", OUTPUT_PATH)

    # 控制台 Top 20
    log("\n" + "=" * 60)
    log("Bio-Omics TIOBE Index (Top 20)")
    log("=" * 60)
    log("%-5s %-24s %8s %10s %8s %8s %8s", "Rank", "Field", "Share%", "Change", "PubMed", "bioRxiv", "GitHub")
    log("-" * 85)
    for item in rankings[:20]:
        ch = item.get("change_str", "N/A")
        log(
            "%-5s %-24s %7.2f%% %10s %8s %8s %8s",
            item["rank"],
            item["field"][:24],
            item["share_pct"],
            ch,
            item["pubmed_count"],
            item["biorxiv_count"],
            item["github_activity"],
        )
    log("\nGenerated at: %s", report["generated_at"])
    return report


# ---------------------------------------------------------------------------
# Top 20 分布逻辑说明
# ---------------------------------------------------------------------------
# 1. 排名依据: 各领域按加权总分排序，总分 = 学术(0.5) + 预印本(0.3) + 技术(0.2)。
# 2. 学术主导: 权重 50% 来自 2025-2026 年 PubMed 文献量，传统强领域（Genomics、
#    Proteomics、Single-cell）文献基数大，易居前列。
# 3. 预印本 30%: 反映近期未正式发表的热度，新兴/前沿方向（如 Spatial Omics、
#    CRISPR）预印本占比高时排名会提升。
# 4. 技术 20%: GitHub 近 6 个月新仓库与总仓库数，工具/流程活跃的领域（如
#    Single-cell、Metagenomics）会获得加成。
# 5. 归一化: 所有领域得分之和为 100%，Share% 即“市场占有率”，便于与 TIOBE 类比。
# 6. 趋势: Change% 为相对上周（或上次运行）的 share_pct 差值，NEW 表示新入榜或历史无数据。
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_omics_index()
