#!/usr/bin/env python3
"""
Bio-Omics Heat Index (v2.0)
===========================
计算生信各组学领域的「增长热度指数」，重点反映领域发展势头而非存量规模。

核心改进 (v2.0):
  - 引入 YoY 增长率：对比 2025 vs 2024 PubMed 发文量，捕捉增长动能
  - 引入 Bioconductor 下载量：反映工具实际使用需求
  - 新权重公式：增长率 40% + 绝对量 30% + 工具下载 30%

数据源:
  - PubMed (NCBI Entrez): 年度文献量 + YoY 增长率
  - bioRxiv: 过去 6 个月预印本活跃度
  - GitHub Search API: 新增仓库与活跃度
  - Bioconductor: R 包下载量统计

输出: docs/data/omics_index.json

依赖:
  pip install biopython requests
"""

from __future__ import annotations

import json
import logging
import os
import re
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
BIOC_DELAY = 0.3       # Bioconductor 限速
API_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# ═══════════════════════════════════════════════════════════════
# 新权重公式 (v2.0): 以增长热度为核心
# ═══════════════════════════════════════════════════════════════
WEIGHT_GROWTH = 0.40       # 增长率权重 (YoY + bioRxiv 增长)
WEIGHT_VOLUME = 0.30       # 绝对量权重 (PubMed + GitHub)
WEIGHT_DOWNLOADS = 0.30    # 工具下载权重 (Bioconductor)

# 子权重分配
WEIGHT_PUBMED_YOY = 0.70        # 增长率中 PubMed YoY 占比
WEIGHT_BIORXIV_GROWTH = 0.30    # 增长率中 bioRxiv 占比

WEIGHT_PUBMED_VOL = 0.60        # 绝对量中 PubMed 占比
WEIGHT_GITHUB_VOL = 0.40        # 绝对量中 GitHub 占比

# NCBI
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# bioRxiv API
BIORXIV_BASE = "https://api.biorxiv.org/details/biorxiv"

# GitHub
GITHUB_SEARCH = "https://api.github.com/search/repositories"

# Bioconductor
BIOC_STATS_BASE = "https://bioconductor.org/packages/stats/bioc"

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.info


# ============================================================
# 核心词库: 20+ 组学领域，每领域配置精准关键词用于 API 检索
# 新增 bioc_packages: Bioconductor 代表性包列表（用于下载量统计）
# ============================================================

OMICS_DICT = {
    # --- 基础组学 (由 Core 升级) ---
    "Genomics": {
        "keywords": [
            "genomics", "WGS", "genome assembly", "T2T-Genome", "Hi-C",
            "Haplotype genome", "variant calling", "VCF", "GWAS", "BSA"
        ],
        "category": "Core Omics",
        "bioc_packages": ["GenomicRanges", "VariantAnnotation", "GenomicFeatures", "Rsamtools", "BSgenome", "rtracklayer"],
    },
    "Transcriptomics": {
        "keywords": [
            "transcriptomics", "RNA-seq", "lncRNA", "circRNA", "miRNA",
            "full-length transcriptomics", "gene expression", "isoform"
        ],
        "category": "Core Omics",
        "bioc_packages": ["DESeq2", "edgeR", "limma", "tximport", "clusterProfiler", "DOSE"],
    },
    "Proteomics": {
        "keywords": [
            "proteomics", "mass spectrometry", "protein identification",
            "TMT", "iTRAQ", "label-free quantification", "LC-MS", "DIA", "DDA"
        ],
        "category": "Core Omics",
        "bioc_packages": ["MSnbase", "MSstats", "mzR", "Spectra", "ProtGenerics"],
    },
    "Metabolomics": {
        "keywords": [
            "metabolomics", "metabolic profiling", "untargeted metabolomics",
            "LC-MS metabolomics", "GC-MS", "metabolite", "MZmine", "XCMS"
        ],
        "category": "Core Omics",
        "bioc_packages": ["xcms", "CAMERA", "MSnbase", "MetaboAnalystR"],
    },
    # --- 进阶/热门 ---
    "Single-cell Omics": {
        "keywords": [
            "single-cell RNA-seq", "scRNA-seq", "single-cell genomics",
            "single-cell multiomics", "10x Genomics", "Scanpy", "Seurat"
        ],
        "category": "Advanced",
        "bioc_packages": ["SingleCellExperiment", "scater", "scran", "DropletUtils", "SingleR", "celldex", "Seurat"],
    },
    "Spatial Omics": {
        "keywords": [
            "spatial transcriptomics", "spatial proteomics", "spatial metabolomics",
            "Visium", "Stereo-seq", "MALDI-imaging"
        ],
        "category": "Advanced",
        "bioc_packages": ["SpatialExperiment", "spatialLIBD", "Giotto", "SpatialFeatureExperiment"],
    },
    "Pangenomics": {
        "keywords": [
            "pangenomics", "pangenome", "graph genome", "comparative genomics"
        ],
        "category": "Advanced",
        "bioc_packages": ["GenomicRanges", "Biostrings"],
    },
    # --- 临床与转化医学 (新增分类) ---
    "Clinical Genomics": {
        "keywords": [
            "WES", "exome sequencing", "liquid biopsy", "ctDNA",
            "cancer genomics", "rare disease genomics", "precision oncology"
        ],
        "category": "Medical",
        "bioc_packages": ["maftools", "TCGAbiolinks", "CNVkit", "COSMIC.67"],
    },
    # --- 微生物与环境 ---
    "Metagenomics": {
        "keywords": [
            "metagenomics", "shotgun metagenomics", "MAGs",
            "16S/18S/ITS", "virome", "culturomics"
        ],
        "category": "Advanced",
        "bioc_packages": ["phyloseq", "dada2", "microbiome", "metagenomeSeq", "curatedMetagenomicData"],
    },
    "Epigenomics": {
        "keywords": [
            "epigenomics", "DNA methylation", "ChIP-seq", "ATAC-seq",
            "Hi-C", "chromatin", "CUT&Tag", "bisulfite sequencing"
        ],
        "category": "Advanced",
        "bioc_packages": ["minfi", "methylKit", "ChIPseeker", "DiffBind", "ATACseqQC", "csaw"],
    },
    "Lipidomics": {
        "keywords": [
            "lipidomics", "lipid profiling", "lipidome", "phospholipid",
            "fatty acid", "lipid metabolism"
        ],
        "category": "Advanced",
        "bioc_packages": ["lipidr", "LOBSTAHS"],
    },
    "Multi-omics": {
        "keywords": [
            "multi-omics", "integrative omics", "multi-omics integration",
            "pan-omics", "multiomic"
        ],
        "category": "Advanced",
        "bioc_packages": ["mixOmics", "MOFA2", "MultiAssayExperiment", "omicade4"],
    },
    # --- 应用/技术 ---
    "Microbiome": {
        "keywords": [
            "microbiome", "gut microbiome", "16S rRNA", "microbiome analysis",
            "dysbiosis", "microbiota"
        ],
        "category": "Applied",
        "bioc_packages": ["phyloseq", "dada2", "microbiome", "MicrobiotaProcess"],
    },
    "Pharmacogenomics": {
        "keywords": [
            "pharmacogenomics", "drug response genomics", "precision medicine",
            "pharmacogenetics"
        ],
        "category": "Applied",
        "bioc_packages": ["PharmacoGx", "DrugVsDisease"],
    },
    "Phylogenomics": {
        "keywords": [
            "phylogenomics", "phylogenetic", "comparative genomics",
            "molecular evolution", "species tree"
        ],
        "category": "Applied",
        "bioc_packages": ["ggtree", "treeio", "phangorn"],
    },
    "Structural Biology": {
        "keywords": [
            "structural biology", "cryo-EM", "protein structure",
            "AlphaFold", "structure prediction"
        ],
        "category": "Applied",
        "bioc_packages": ["bio3d", "Rpdb"],
    },
    "Immunogenomics": {
        "keywords": [
            "immunogenomics", "TCR sequencing", "BCR repertoire",
            "immune repertoire", "V(D)J"
        ],
        "category": "Applied",
        "bioc_packages": ["immunarch", "scRepertoire", "alakazam"],
    },
    "Synthetic Biology": {
        "keywords": [
            "synthetic biology", "genetic circuit", "genome engineering",
            "synthetic genome"
        ],
        "category": "Applied",
        "bioc_packages": ["Biostrings", "seqinr"],
    },
    "Glycomics": {
        "keywords": [
            "glycomics", "glycoproteomics", "glycan", "glycosylation"
        ],
        "category": "Applied",
        "bioc_packages": [],
    },
    "Radiomics": {
        "keywords": [
            "radiomics", "radiomic features", "imaging biomarkers",
            "radiogenomics"
        ],
        "category": "Applied",
        "bioc_packages": [],
    },
    "Nutrigenomics": {
        "keywords": [
            "nutrigenomics", "nutritional genomics", "diet-gene",
            "nutrigenetics"
        ],
        "category": "Applied",
        "bioc_packages": [],
    },
    "Long-read Sequencing": {
        "keywords": [
            "long-read sequencing", "Nanopore", "PacBio", "HiFi",
            "long-read genome", "third-generation sequencing"
        ],
        "category": "Technology",
        "bioc_packages": ["NanoMethViz", "nanopohr"],
    },
    "CRISPR Genomics": {
        "keywords": [
            "CRISPR", "CRISPR screen", "genome editing", "CRISPR-Cas9",
            "CRISPR-Cas"
        ],
        "category": "Technology",
        "bioc_packages": ["CRISPRseek", "crisprScore", "crisprDesign"],
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

def _pubmed_esearch_term(keywords: list[str], year_start: int, year_end: int) -> str:
    """构建 PubMed 检索式：关键词 OR + 日期范围。"""
    term = " OR ".join(f'"{k}"' for k in keywords)
    date_filter = f'("{year_start}"[Date - Publication] : "{year_end}"[Date - Publication])'
    return f"({term}) AND {date_filter}"


def fetch_pubmed_count(query: str) -> int:
    """查询 PubMed 文献量。优先使用 Biopython Entrez。"""
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


def fetch_pubmed_yoy(keywords: list[str]) -> tuple[int, int, float]:
    """
    获取 PubMed YoY 增长率。
    返回: (count_2025, count_2024, yoy_growth_rate)
    """
    current_year = datetime.now().year
    prev_year = current_year - 1
    
    query_current = _pubmed_esearch_term(keywords, current_year, current_year)
    query_prev = _pubmed_esearch_term(keywords, prev_year, prev_year)
    
    count_current = fetch_pubmed_count(query_current)
    _rate_limit(NCBI_DELAY)
    count_prev = fetch_pubmed_count(query_prev)
    
    # 计算 YoY 增长率，限制在 -50% ~ +200%
    if count_prev > 0:
        yoy = (count_current - count_prev) / count_prev * 100
    else:
        yoy = 100.0 if count_current > 0 else 0.0
    yoy = max(-50.0, min(200.0, yoy))
    
    return count_current, count_prev, round(yoy, 2)


def fetch_biorxiv_count(keywords: list[str]) -> int:
    """过去 6 个月各领域预印本量：通过 PubMed 预印本出版类型检索（Preprint[pt]）。"""
    term = " OR ".join(f'"{k}"' for k in keywords)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_filter = (
        f'("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : '
        f'"{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
    )
    query = f'({term}) AND {date_filter} AND "Preprint"[pt]'
    return fetch_pubmed_count(query)


def fetch_biorxiv_total_last_six_months() -> int:
    """调用 bioRxiv API 获取过去 6 个月预印本总数（用于报告说明）。"""
    url = f"{BIORXIV_BASE}/180d/0"
    data = _http_get(url)
    if not data:
        return 0
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
# 数据抓取: Bioconductor 下载量
# ============================================================

# 缓存已获取的包下载量，避免重复请求
_bioc_download_cache: dict[str, int] = {}


def fetch_bioc_package_downloads(package: str) -> int:
    """
    获取单个 Bioconductor 包的年度下载量。
    数据源: https://bioconductor.org/packages/stats/bioc/{package}/{package}_stats.tab
    """
    if package in _bioc_download_cache:
        return _bioc_download_cache[package]
    
    url = f"{BIOC_STATS_BASE}/{package}/{package}_stats.tab"
    try:
        if HAS_REQUESTS:
            r = requests.get(url, timeout=API_TIMEOUT)
            if r.status_code != 200:
                _bioc_download_cache[package] = 0
                return 0
            content = r.text
        else:
            req = Request(url, headers={"User-Agent": "Bio-Omics-Index/2.0"})
            with urlopen(req, timeout=API_TIMEOUT) as resp:
                content = resp.read().decode("utf-8")
        
        # 解析 TSV: Year Month Nb_of_distinct_IPs Nb_of_downloads
        # 取最近 12 个月的下载总量
        lines = content.strip().split("\n")
        if len(lines) < 2:
            _bioc_download_cache[package] = 0
            return 0
        
        total_downloads = 0
        count = 0
        for line in reversed(lines[1:]):  # 跳过表头，从最新月份开始
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    downloads = int(parts[3])
                    total_downloads += downloads
                    count += 1
                    if count >= 12:  # 只取最近 12 个月
                        break
                except ValueError:
                    continue
        
        _bioc_download_cache[package] = total_downloads
        return total_downloads
    
    except Exception as e:
        log("  Bioc stats error for %s: %s", package, e)
        _bioc_download_cache[package] = 0
        return 0


def fetch_bioc_downloads_for_field(packages: list[str]) -> int:
    """获取某领域所有 Bioconductor 包的年度下载总量。"""
    if not packages:
        return 0
    
    total = 0
    for pkg in packages:
        downloads = fetch_bioc_package_downloads(pkg)
        total += downloads
        _rate_limit(BIOC_DELAY)
    
    return total


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
# 指数算法 v2.0: 增长热度优先
# ============================================================

def normalize_values(values: list[float]) -> list[float]:
    """Min-Max 归一化到 [0, 1]，避免绝对量主导。"""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.5] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def calculate_momentum_tier(yoy_growth: float, biorxiv_growth: float) -> str:
    """根据增长率计算动能等级。"""
    avg_growth = yoy_growth * 0.7 + biorxiv_growth * 0.3
    if avg_growth > 30:
        return "Rising Star"
    elif avg_growth > 15:
        return "Hot"
    elif avg_growth > 5:
        return "Growing"
    elif avg_growth > -5:
        return "Stable"
    else:
        return "Cooling"


def calculate_scores_v2(raw_data: dict[str, dict]) -> list[dict[str, Any]]:
    """
    v2.0 评分公式:
    Heat Score = 增长率(40%) + 绝对量(30%) + 工具下载(30%)
    
    增长率 = PubMed YoY(70%) + bioRxiv 相对热度(30%)
    绝对量 = PubMed(60%) + GitHub(40%)
    工具下载 = Bioconductor 年度下载量
    
    所有子指标先 Min-Max 归一化到 [0, 1]，再加权求和。
    """
    fields = list(raw_data.keys())
    n = len(fields)
    
    # 提取各维度原始值
    yoy_rates = [raw_data[f]["yoy_rate"] for f in fields]
    biorxiv_counts = [raw_data[f]["biorxiv"] for f in fields]
    pubmed_counts = [raw_data[f]["pubmed_current"] for f in fields]
    github_counts = [raw_data[f]["github"] for f in fields]
    bioc_downloads = [raw_data[f]["bioc_downloads"] for f in fields]
    
    # 归一化
    norm_yoy = normalize_values(yoy_rates)
    norm_biorxiv = normalize_values(biorxiv_counts)
    norm_pubmed = normalize_values(pubmed_counts)
    norm_github = normalize_values(github_counts)
    norm_bioc = normalize_values(bioc_downloads)
    
    results = []
    for i, field in enumerate(fields):
        data = raw_data[field]
        
        # 计算各维度得分 (归一化后)
        growth_score = (
            norm_yoy[i] * WEIGHT_PUBMED_YOY +
            norm_biorxiv[i] * WEIGHT_BIORXIV_GROWTH
        )
        volume_score = (
            norm_pubmed[i] * WEIGHT_PUBMED_VOL +
            norm_github[i] * WEIGHT_GITHUB_VOL
        )
        download_score = norm_bioc[i]
        
        # 总分
        heat_score = (
            growth_score * WEIGHT_GROWTH +
            volume_score * WEIGHT_VOLUME +
            download_score * WEIGHT_DOWNLOADS
        )
        
        # 动能等级
        # bioRxiv 增长用相对排名近似（归一化值 * 50 模拟增长率）
        biorxiv_pseudo_growth = norm_biorxiv[i] * 50
        momentum = calculate_momentum_tier(data["yoy_rate"], biorxiv_pseudo_growth)
        
        results.append({
            "field": field,
            "category": data.get("category", ""),
            # 原始数据
            "pubmed_current": data["pubmed_current"],
            "pubmed_prev": data["pubmed_prev"],
            "yoy_rate": data["yoy_rate"],
            "biorxiv_count": data["biorxiv"],
            "github_activity": data["github"],
            "bioc_downloads": data["bioc_downloads"],
            # 归一化分数 (调试用)
            "growth_score": round(growth_score, 4),
            "volume_score": round(volume_score, 4),
            "download_score": round(download_score, 4),
            # 总分
            "heat_score": round(heat_score, 4),
            # 动能等级
            "momentum": momentum,
        })
    
    # 排序并计算 Share%
    total_heat = sum(r["heat_score"] for r in results)
    for r in results:
        r["share_pct"] = round(r["heat_score"] / total_heat * 100, 2) if total_heat > 0 else 0.0
    
    results.sort(key=lambda x: x["heat_score"], reverse=True)
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
    """
    按 OMICS_DICT 依次拉取所有数据源，严格限速。
    
    采集项:
      - PubMed YoY (当年 + 去年 + 增长率)
      - bioRxiv 6 个月预印本
      - GitHub 活跃度
      - Bioconductor 下载量
    """
    raw_data = {}
    n = len(OMICS_DICT)
    
    for i, (field, config) in enumerate(OMICS_DICT.items(), 1):
        keywords = config["keywords"]
        category = config.get("category", "")
        bioc_packages = config.get("bioc_packages", [])
        
        log("[%d/%d] %s", i, n, field)
        
        # 1) PubMed YoY
        pubmed_current, pubmed_prev, yoy_rate = fetch_pubmed_yoy(keywords)
        log("  PubMed: %s (2025) / %s (2024) → YoY: %+.1f%%", pubmed_current, pubmed_prev, yoy_rate)
        _rate_limit(NCBI_DELAY)
        
        # 2) bioRxiv
        biorxiv_count = fetch_biorxiv_count(keywords)
        log("  bioRxiv (6m): %s", biorxiv_count)
        _rate_limit(NCBI_DELAY)
        
        # 3) GitHub
        github_activity = fetch_github_activity(keywords[0])
        log("  GitHub: %s", github_activity)
        _rate_limit(GITHUB_DELAY)
        
        # 4) Bioconductor
        bioc_downloads = fetch_bioc_downloads_for_field(bioc_packages)
        log("  Bioconductor (%d pkgs): %s downloads/year", len(bioc_packages), bioc_downloads)
        
        raw_data[field] = {
            "pubmed_current": pubmed_current,
            "pubmed_prev": pubmed_prev,
            "yoy_rate": yoy_rate,
            "biorxiv": biorxiv_count,
            "github": github_activity,
            "bioc_downloads": bioc_downloads,
            "category": category,
        }
    
    return raw_data


def generate_omics_index() -> dict[str, Any]:
    """生成 Bio-Omics Heat Index v2.0 并写入 docs/data/omics_index.json。"""
    log("=" * 70)
    log("Bio-Omics Heat Index v2.0")
    log("=" * 70)
    log("New formula: Growth(40%%) + Volume(30%%) + Downloads(30%%)")
    log("")
    
    if not HAS_BIOPYTHON:
        log("Warning: Biopython not installed. Using URL fallback for PubMed.")
    
    # 1) 数据采集
    log("\n[Phase 1] Collecting data (PubMed YoY, bioRxiv, GitHub, Bioconductor)...")
    raw_data = collect_all_data()
    
    # 2) 计分与归一化 (v2.0)
    log("\n[Phase 2] Calculating Heat Scores (v2.0 algorithm)...")
    rankings = calculate_scores_v2(raw_data)
    
    # 3) 趋势
    log("\n[Phase 3] Trend (compare with existing %s)...", OUTPUT_PATH)
    attach_trend(rankings, OUTPUT_PATH)
    
    # 4) bioRxiv 总量
    biorxiv_total = 0
    try:
        _rate_limit(BIORXIV_DELAY)
        biorxiv_total = fetch_biorxiv_total_last_six_months()
    except Exception as e:
        log("  Skip biorxiv total: %s", e)
    
    current_year = datetime.now().year
    report = {
        "generated_at": datetime.now().isoformat(),
        "version": "2.0",
        "methodology": {
            "formula": "Heat = Growth(40%) + Volume(30%) + Downloads(30%)",
            "weights": {
                "growth": WEIGHT_GROWTH,
                "volume": WEIGHT_VOLUME,
                "downloads": WEIGHT_DOWNLOADS,
            },
            "sub_weights": {
                "growth_pubmed_yoy": WEIGHT_PUBMED_YOY,
                "growth_biorxiv": WEIGHT_BIORXIV_GROWTH,
                "volume_pubmed": WEIGHT_PUBMED_VOL,
                "volume_github": WEIGHT_GITHUB_VOL,
            },
            "pubmed_period": f"{current_year} vs {current_year - 1}",
            "biorxiv_period": "last 6 months (PubMed Preprint[pt])",
            "github_period": "last 6 months new repos + total (stars>10)",
            "bioconductor_period": "last 12 months downloads",
            "biorxiv_total_6m": biorxiv_total,
        },
        "total_fields": len(rankings),
        "rankings": rankings,
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("\nSaved: %s", OUTPUT_PATH)
    
    # 控制台输出
    log("\n" + "=" * 100)
    log("Bio-Omics Heat Index v2.0 (Top 20)")
    log("=" * 100)
    log("%-4s %-22s %7s %8s %10s %8s %8s %10s %10s",
        "Rank", "Field", "Share%", "YoY%", "Momentum", "PubMed", "bioRxiv", "GitHub", "Bioc DL")
    log("-" * 100)
    for item in rankings[:20]:
        ch = item.get("change_str", "N/A")
        yoy_str = f"{item['yoy_rate']:+.1f}%"
        log(
            "%-4s %-22s %6.2f%% %8s %10s %8s %8s %10s %10s",
            item["rank"],
            item["field"][:22],
            item["share_pct"],
            yoy_str,
            item["momentum"],
            item["pubmed_current"],
            item["biorxiv_count"],
            item["github_activity"],
            item["bioc_downloads"],
        )
    log("\nGenerated at: %s", report["generated_at"])
    return report


# ---------------------------------------------------------------------------
# v2.0 算法说明
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
