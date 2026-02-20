#!/usr/bin/env python3
"""
Bio-Omics Heat Index (v3.0)
===========================
计算生信各组学领域的「五维度热度指数」，全面反映领域学术热度、技术生态和发展势头。

核心算法 (v3.0):
  Heat Score = 学术增长力(30%) + 预印本活跃度(20%) + 技术开发势能(25%)
             + 资金信号(15%) + 社区关注度(10%)

子维度:
  - 学术增长力: PubMed YoY(70%) + 被引/高影响论文(30%, Phase 2)
  - 预印本活跃度: bioRxiv 量(60%) + bioRxiv 增长(40%)
  - 技术开发势能: GitHub(30%) + Bioconductor(30%) + PyPI(40%)
  - 资金信号: NIH 项目数(100%)
  - 社区关注度: Google Trends(50%) + Semantic Scholar(50%)

数据源:
  - PubMed (NCBI Entrez): 年度文献量 + YoY 增长率
  - bioRxiv: 过去 6 个月预印本活跃度
  - GitHub Search API: 新增仓库与活跃度
  - Bioconductor: R 包下载量统计
  - PyPI Stats: Python 包下载量统计
  - NIH RePORTER: 近 2 年资助项目数
  - Google Trends: 相对搜索热度
  - Semantic Scholar: 学术搜索结果数 (社区关注度)

输出: docs/data/omics_index.json

依赖:
  pip install biopython requests pytrends (pytrends 可选)
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
HISTORY_PATH = PROJECT_ROOT / "docs" / "data" / "biorxiv_history.json"

# 建议设置 GITHUB_TOKEN 以提高 GitHub API 配额，避免 403 rate limit
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "bio-rank-gateway@localhost")
# Semantic Scholar (免费, 无需 API key, 100 req/5min)

# API 速率限制 (秒)
NCBI_DELAY = 0.4       # NCBI 建议无 key 时 ≤3 req/s
BIORXIV_DELAY = 1.0
GITHUB_DELAY = 2.0
BIOC_DELAY = 0.3       # Bioconductor 限速
PYPI_DELAY = 0.2       # PyPI Stats 限速
TRENDS_DELAY = 3.0     # Google Trends 限速 (需 ≥3s 避免 429)
NIH_DELAY = 0.5        # NIH RePORTER 限速
SCHOLAR_DELAY = 3.0    # Semantic Scholar 限速 (100 req/5min → ~3s safe)
OPENALEX_DELAY = 0.15  # OpenAlex 限速 (polite pool: 10 req/s)
API_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# ═══════════════════════════════════════════════════════════════
# 五维度热度模型 (v3.0)
# ═══════════════════════════════════════════════════════════════
# Heat Index = 学术增长力(30%) + 预印本活跃度(20%) + 技术开发势能(25%) 
#            + 资金信号(15%) + 社区关注度(10%)

# 一级维度权重
WEIGHT_ACADEMIC = 0.30       # 学术增长力
WEIGHT_PREPRINT = 0.20       # 预印本活跃度
WEIGHT_TECH = 0.25           # 技术开发势能
WEIGHT_FUNDING = 0.15        # 资金信号 (NIH RePORTER)
WEIGHT_COMMUNITY = 0.10      # 社区关注度 (Google Trends + Scholar Results)

# 学术增长力子权重
WEIGHT_PUBMED_YOY = 0.70         # PubMed YoY 增长率
WEIGHT_CITATION_MOMENTUM = 0.30  # 高影响力论文/被引 (Phase 2: OpenAlex)

# 预印本活跃度子权重
WEIGHT_BIORXIV_COUNT = 0.60      # bioRxiv 6 个月发文量
WEIGHT_BIORXIV_GROWTH = 0.40     # bioRxiv 增长率

# 技术开发势能子权重 (R 30% / Python 40%)
WEIGHT_GITHUB = 0.30             # GitHub 活跃度
WEIGHT_BIOC = 0.30               # Bioconductor (R 生态)
WEIGHT_PYPI = 0.40               # PyPI (Python 生态)

# 社区关注度子权重
WEIGHT_GTRENDS = 0.50            # Google Trends (相对热度趋势)
WEIGHT_SCHOLAR = 0.50            # Semantic Scholar Results (学术搜索量)

# API 端点
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIORXIV_BASE = "https://api.biorxiv.org/details/biorxiv"
GITHUB_SEARCH = "https://api.github.com/search/repositories"
BIOC_STATS_BASE = "https://bioconductor.org/packages/stats/bioc"
PYPI_STATS_BASE = "https://pypistats.org/api/packages"
NIH_REPORTER_BASE = "https://api.reporter.nih.gov/v2/projects/search"
BING_SEARCH_BASE = "https://api.bing.microsoft.com/v7.0/search"
OPENALEX_WORKS_BASE = "https://api.openalex.org/works"
SCHOLAR_SEARCH_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.info


# ============================================================
# 核心词库: 23 组学领域（v3.0 全面补充版）
# 每领域 8-12 个关键词 + Bioconductor/PyPI 包列表
# ============================================================

OMICS_DICT = {
    # ═══════════════════════════════════════════════════════════
    # Core Omics (基础组学)
    # ═══════════════════════════════════════════════════════════
    "Genomics": {
        "keywords": [
            "genomics", "whole genome sequencing", "WGS", "genome assembly",
            "variant calling", "SNP", "VCF", "GWAS", "genome annotation",
            "reference genome", "de novo assembly", "structural variant"
        ],
        "category": "Core Omics",
        "bioc_packages": ["GenomicRanges", "VariantAnnotation", "GenomicFeatures", 
                         "Rsamtools", "BSgenome", "rtracklayer"],
        "pypi_packages": ["pysam", "cyvcf2", "pyvcf", "biopython", "pyfaidx"],
    },
    "Transcriptomics": {
        "keywords": [
            "transcriptomics", "RNA-seq", "RNA sequencing", "gene expression",
            "differential expression", "lncRNA", "circRNA", "miRNA",
            "transcriptome", "isoform", "alternative splicing", "mRNA"
        ],
        "category": "Core Omics",
        "bioc_packages": ["DESeq2", "edgeR", "limma", "tximport", 
                         "clusterProfiler", "DOSE", "fgsea"],
        "pypi_packages": ["htseq", "pysam", "pydeseq2", "gseapy", "rpy2"],
    },
    "Proteomics": {
        "keywords": [
            "proteomics", "mass spectrometry", "LC-MS", "protein identification",
            "TMT", "iTRAQ", "label-free quantification", "DIA", "DDA",
            "peptide", "protein quantification", "shotgun proteomics"
        ],
        "category": "Core Omics",
        "bioc_packages": ["MSnbase", "MSstats", "mzR", "Spectra", "ProtGenerics"],
        "pypi_packages": ["pyteomics", "pyopenms", "spectrum_utils", "ms2pip"],
    },
    "Metabolomics": {
        "keywords": [
            "metabolomics", "metabolic profiling", "untargeted metabolomics",
            "targeted metabolomics", "LC-MS metabolomics", "GC-MS", 
            "metabolite identification", "metabolome", "XCMS", "MZmine",
            "metabolic pathway", "flux analysis"
        ],
        "category": "Core Omics",
        "bioc_packages": ["xcms", "CAMERA", "MSnbase", "MetaboAnalystR"],
        "pypi_packages": ["pyopenms", "ms2deepscore", "matchms", "spec2vec"],
    },
    
    # ═══════════════════════════════════════════════════════════
    # Advanced (进阶/热门)
    # ═══════════════════════════════════════════════════════════
    "Single-cell Omics": {
        "keywords": [
            "single-cell RNA-seq", "scRNA-seq", "single-cell sequencing",
            "single-cell multiomics", "10x Genomics", "single-cell ATAC",
            "cell clustering", "trajectory analysis", "cell annotation",
            "droplet-based", "Smart-seq", "single-cell transcriptomics"
        ],
        "category": "Advanced",
        "bioc_packages": ["SingleCellExperiment", "scater", "scran", 
                         "DropletUtils", "SingleR", "celldex", "Seurat"],
        "pypi_packages": ["scanpy", "anndata", "scvi-tools", "cellxgene", 
                         "leidenalg", "scrublet", "velocyto"],
    },
    "Spatial Omics": {
        "keywords": [
            "spatial transcriptomics", "spatial proteomics", "spatial omics",
            "Visium", "10x Xenium", "Stereo-seq", "CosMx", "MERFISH",
            "seqFISH", "MALDI-imaging", "spatial gene expression",
            "tissue mapping", "spatial single-cell"
        ],
        "category": "Advanced",
        "bioc_packages": ["SpatialExperiment", "spatialLIBD", "Giotto", 
                         "SpatialFeatureExperiment"],
        "pypi_packages": ["squidpy", "scanpy", "stlearn", "spatialdata", "tangram"],
    },
    "Pangenomics": {
        "keywords": [
            "pangenomics", "pangenome", "pan-genome", "graph genome",
            "core genome", "accessory genome", "pangenome analysis",
            "comparative genomics", "gene presence absence", "COG",
            "pangenome graph", "population genomics"
        ],
        "category": "Advanced",
        "bioc_packages": ["GenomicRanges", "Biostrings", "DECIPHER"],
        "pypi_packages": ["ppanggolin"],
    },
    "Metagenomics": {
        "keywords": [
            "metagenomics", "shotgun metagenomics", "metagenomic sequencing",
            "MAGs", "metagenome-assembled genome", "taxonomic profiling",
            "functional metagenomics", "16S rRNA", "amplicon sequencing",
            "virome", "metagenome binning", "environmental DNA"
        ],
        "category": "Advanced",
        "bioc_packages": ["phyloseq", "dada2", "microbiome", 
                         "metagenomeSeq", "curatedMetagenomicData"],
        "pypi_packages": ["metaphlan", "humann"],
    },
    "Epigenomics": {
        "keywords": [
            "epigenomics", "epigenetics", "DNA methylation", "ChIP-seq",
            "ATAC-seq", "histone modification", "chromatin accessibility",
            "CUT&Tag", "CUT&RUN", "bisulfite sequencing", "WGBS",
            "chromatin state", "enhancer", "promoter methylation"
        ],
        "category": "Advanced",
        "bioc_packages": ["minfi", "methylKit", "ChIPseeker", "DiffBind", 
                         "ATACseqQC", "csaw", "bsseq"],
        "pypi_packages": ["deeptools", "pygenometracks", "macs2", "methylpy"],
    },
    "Lipidomics": {
        "keywords": [
            "lipidomics", "lipid profiling", "lipidome", "lipid metabolism",
            "phospholipid", "sphingolipid", "fatty acid", "lipid class",
            "lipid species", "lipid quantification", "membrane lipid",
            "lipid biomarker"
        ],
        "category": "Advanced",
        "bioc_packages": ["lipidr", "LOBSTAHS"],
        "pypi_packages": [],  # lipidcreator/lipidspace 仅有 Conda
    },
    "Multi-omics": {
        "keywords": [
            "multi-omics", "multiomics", "integrative omics", "omics integration",
            "multi-omics integration", "pan-omics", "data integration",
            "multi-modal", "cross-omics", "systems biology",
            "network analysis", "pathway integration"
        ],
        "category": "Advanced",
        "bioc_packages": ["mixOmics", "MOFA2", "MultiAssayExperiment", "omicade4"],
        "pypi_packages": ["mofapy2", "scikit-fusion", "tensorly"],
    },
    
    # ═══════════════════════════════════════════════════════════
    # Medical (临床与转化医学)
    # ═══════════════════════════════════════════════════════════
    "Clinical Genomics": {
        "keywords": [
            "clinical genomics", "WES", "whole exome sequencing", "exome",
            "liquid biopsy", "ctDNA", "circulating tumor DNA", "cancer genomics",
            "rare disease genomics", "precision oncology", "germline variant",
            "somatic mutation", "clinical sequencing"
        ],
        "category": "Medical",
        "bioc_packages": ["maftools", "TCGAbiolinks", "CNVkit", "AnnotationHub"],
        "pypi_packages": [],  # oncokb-annotator 等仅有 Conda/GitHub
    },
    
    # ═══════════════════════════════════════════════════════════
    # Applied (应用/技术)
    # ═══════════════════════════════════════════════════════════
    "Microbiome": {
        "keywords": [
            "microbiome", "gut microbiome", "microbiota", "16S rRNA",
            "microbiome analysis", "dysbiosis", "microbial community",
            "oral microbiome", "skin microbiome", "vaginal microbiome",
            "host-microbe interaction", "microbiome diversity"
        ],
        "category": "Applied",
        "bioc_packages": ["phyloseq", "dada2", "microbiome", "MicrobiotaProcess"],
        "pypi_packages": ["scikit-bio", "biom-format", "emperor"],  # qiime2 仅 Conda
    },
    "Pharmacogenomics": {
        "keywords": [
            "pharmacogenomics", "pharmacogenetics", "drug response",
            "drug metabolism", "precision medicine", "personalized medicine",
            "CYP450", "drug-gene interaction", "adverse drug reaction",
            "therapeutic drug monitoring", "ADME", "pharmacokinetics"
        ],
        "category": "Applied",
        "bioc_packages": ["PharmacoGx", "DrugVsDisease"],
        "pypi_packages": ["pharmpy"],  # ddinter 不在 PyPI
    },
    "Phylogenomics": {
        "keywords": [
            "phylogenomics", "phylogenetics", "phylogenetic analysis",
            "molecular evolution", "species tree", "gene tree",
            "phylogenetic tree", "evolutionary genomics", "divergence time",
            "ancestral reconstruction", "ortholog", "synteny"
        ],
        "category": "Applied",
        "bioc_packages": ["ggtree", "treeio", "phangorn", "ape"],
        "pypi_packages": ["biopython", "ete3", "dendropy", "toytree"],
    },
    "Structural Biology": {
        "keywords": [
            "structural biology", "protein structure", "cryo-EM",
            "cryo-electron microscopy", "AlphaFold", "structure prediction",
            "protein folding", "molecular docking", "homology modeling",
            "X-ray crystallography", "NMR spectroscopy", "PDB"
        ],
        "category": "Applied",
        "bioc_packages": ["bio3d", "Rpdb"],
        "pypi_packages": ["biopython", "mdanalysis", "prody", "openmm"],
                         # pymol/alphafold/rosettafold 仅 Conda
    },
    "Immunogenomics": {
        "keywords": [
            "immunogenomics", "TCR sequencing", "TCR repertoire",
            "BCR repertoire", "immune repertoire", "V(D)J recombination",
            "T cell receptor", "B cell receptor", "immunome",
            "neoantigen", "HLA typing", "immune profiling"
        ],
        "category": "Applied",
        "bioc_packages": ["immunarch", "scRepertoire", "alakazam"],
        "pypi_packages": ["changeo", "tcrdist3", "immuneml"],  # pyir 不在 PyPI
    },
    "Synthetic Biology": {
        "keywords": [
            "synthetic biology", "genetic circuit", "genome engineering",
            "synthetic genome", "gene synthesis", "metabolic engineering",
            "DNA assembly", "BioBricks", "genetic parts", "biosensor",
            "cell factory", "pathway engineering"
        ],
        "category": "Applied",
        "bioc_packages": ["Biostrings", "seqinr"],
        "pypi_packages": ["dnachisel", "pydna"],  # sboltools/teselagen 不在 PyPI
    },
    "Glycomics": {
        "keywords": [
            "glycomics", "glycoproteomics", "glycan", "glycosylation",
            "N-glycan", "O-glycan", "glycome", "carbohydrate",
            "lectin", "glycan structure", "glycan analysis",
            "glycoprotein", "sialic acid"
        ],
        "category": "Applied",
        "bioc_packages": [],  # 无专用 R 包
        "pypi_packages": ["glycowork", "glypy"],  # glycopeptidegraphms 不在 PyPI
    },
    "Radiomics": {
        "keywords": [
            "radiomics", "radiomic features", "imaging biomarkers",
            "radiogenomics", "medical imaging", "CT radiomics",
            "MRI radiomics", "PET radiomics", "texture analysis",
            "image-based phenotyping", "tumor heterogeneity",
            "imaging genomics"
        ],
        "category": "Applied",
        "bioc_packages": [],  # 无专用 R 包
        "pypi_packages": ["pyradiomics", "simpleitk", "nibabel", "dicom2nifti"],
    },
    "Nutrigenomics": {
        "keywords": [
            "nutrigenomics", "nutrigenetics", "nutritional genomics",
            "diet-gene interaction", "food genomics", "nutrient metabolism",
            "dietary intervention", "personalized nutrition",
            "metabolic response", "food metabolome", "dietary biomarker",
            "nutrition and genetics"
        ],
        "category": "Applied",
        "bioc_packages": [],  # 无专用 R 包
        "pypi_packages": ["fooddata"],  # PyPI 包较少
    },
    
    # ═══════════════════════════════════════════════════════════
    # Technology (技术)
    # ═══════════════════════════════════════════════════════════
    "Long-read Sequencing": {
        "keywords": [
            "long-read sequencing", "Nanopore", "Oxford Nanopore",
            "PacBio", "HiFi", "SMRT sequencing", "third-generation sequencing",
            "long-read RNA-seq", "direct RNA sequencing", "ultra-long reads",
            "structural variant detection", "phasing"
        ],
        "category": "Technology",
        "bioc_packages": ["NanoMethViz", "nanopohr", "Longread"],
        "pypi_packages": ["nanofilt", "nanoplot", "medaka"],
                         # pomoxis/dorado/pbmm2 仅 Conda
    },
    "CRISPR Genomics": {
        "keywords": [
            "CRISPR", "CRISPR-Cas9", "CRISPR screen", "genome editing",
            "CRISPR-Cas", "sgRNA", "guide RNA", "gene knockout",
            "base editing", "prime editing", "CRISPR activation",
            "CRISPR interference", "Cas12", "Cas13"
        ],
        "category": "Technology",
        "bioc_packages": ["CRISPRseek", "crisprScore", "crisprDesign"],
        "pypi_packages": [],  # crispresso2/mageck 仅 Conda
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
    获取 PubMed YoY 增长率（滚动 12 个月同期对比）。

    比较窗口:
      - 当前期: 过去 12 个月 (today-365d ~ today)
      - 对比期: 前一个 12 个月 (today-730d ~ today-366d)

    返回: (count_current_12m, count_prev_12m, yoy_growth_rate%)
    """
    today = datetime.now()

    # 当前期: 过去 12 个月
    cur_start = (today - timedelta(days=365)).strftime("%Y/%m/%d")
    cur_end = today.strftime("%Y/%m/%d")

    # 对比期: 12-24 个月前
    prev_start = (today - timedelta(days=730)).strftime("%Y/%m/%d")
    prev_end = (today - timedelta(days=366)).strftime("%Y/%m/%d")

    term = " OR ".join(f'"{k}"' for k in keywords)
    query_current = (
        f'({term}) AND ("{cur_start}"[Date - Publication] : '
        f'"{cur_end}"[Date - Publication])'
    )
    query_prev = (
        f'({term}) AND ("{prev_start}"[Date - Publication] : '
        f'"{prev_end}"[Date - Publication])'
    )

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
# bioRxiv 历史数据持久化 (用于计算增长率)
# ============================================================

def load_biorxiv_history() -> dict[str, dict]:
    """
    加载 bioRxiv 历史数据。
    
    返回格式:
    {
        "2025-02-20": {
            "Genomics": 850,
            "Transcriptomics": 420,
            ...
        },
        "2025-02-13": { ... },
        ...
    }
    """
    if not HISTORY_PATH.exists():
        return {}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log("Warning: Failed to load biorxiv history: %s", e)
        return {}


def save_biorxiv_history(history: dict[str, dict]) -> None:
    """保存 bioRxiv 历史数据，保留最近 12 周。"""
    # 按日期排序，只保留最近 12 条
    sorted_dates = sorted(history.keys(), reverse=True)[:12]
    trimmed = {d: history[d] for d in sorted_dates}
    
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("Warning: Failed to save biorxiv history: %s", e)


def calculate_biorxiv_growth(
    current_counts: dict[str, int],
    history: dict[str, dict]
) -> dict[str, float]:
    """
    计算 bioRxiv 增长率。
    
    算法: 对比上周数据 (最近一条历史记录)。
    如果没有历史数据，返回 0.0 (无法计算增长)。
    
    返回: {"Genomics": 5.2, "Transcriptomics": -2.1, ...}
    """
    growth_rates = {}
    
    # 获取上一周的数据 (最新的历史记录)
    sorted_dates = sorted(history.keys(), reverse=True)
    if not sorted_dates:
        # 没有历史数据，所有增长率为 0
        for field in current_counts:
            growth_rates[field] = 0.0
        return growth_rates
    
    prev_date = sorted_dates[0]
    prev_counts = history[prev_date]
    
    for field, current in current_counts.items():
        prev = prev_counts.get(field, 0)
        if prev > 0:
            growth = (current - prev) / prev * 100
            # 限制在 -50% ~ +200%
            growth = max(-50.0, min(200.0, growth))
        else:
            growth = 100.0 if current > 0 else 0.0
        growth_rates[field] = round(growth, 2)
    
    return growth_rates


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
# 数据抓取: PyPI 下载量
# ============================================================

_pypi_download_cache: dict[str, int] = {}


def fetch_pypi_package_downloads(package: str) -> int:
    """
    获取单个 PyPI 包的近 6 个月下载量。
    数据源: https://pypistats.org/api/packages/{package}/recent
    返回: last_month * 6 的近似值（API 只返回近期数据）
    """
    if package in _pypi_download_cache:
        return _pypi_download_cache[package]
    
    url = f"{PYPI_STATS_BASE}/{package}/recent"
    try:
        data = _http_get(url)
        if not data or "data" not in data:
            _pypi_download_cache[package] = 0
            return 0
        
        # recent API 返回 last_day, last_week, last_month
        last_month = data["data"].get("last_month", 0)
        # 用 last_month * 6 近似 6 个月下载量
        downloads_6m = last_month * 6
        
        _pypi_download_cache[package] = downloads_6m
        return downloads_6m
    
    except Exception as e:
        log("  PyPI stats error for %s: %s", package, e)
        _pypi_download_cache[package] = 0
        return 0


def fetch_pypi_downloads_for_field(packages: list[str]) -> int:
    """获取某领域所有 PyPI 包的 6 个月下载总量。"""
    if not packages:
        return 0
    
    total = 0
    for pkg in packages:
        downloads = fetch_pypi_package_downloads(pkg)
        total += downloads
        _rate_limit(PYPI_DELAY)
    
    return total


# ============================================================
# 数据抓取: NIH RePORTER (资金信号)
# ============================================================

_nih_cache: dict[str, int] = {}


def fetch_nih_funding_projects(keywords: list[str]) -> int:
    """
    查询 NIH RePORTER 获取近 2 年资助项目数量。
    API: https://api.reporter.nih.gov/v2/projects/search
    """
    cache_key = keywords[0] if keywords else ""
    if cache_key in _nih_cache:
        return _nih_cache[cache_key]
    
    # 构建搜索词
    search_text = " OR ".join(keywords[:5])  # 取前 5 个关键词
    current_year = datetime.now().year
    
    payload = {
        "criteria": {
            "advanced_text_search": {
                "operator": "or",
                "search_field": "all",
                "search_text": search_text
            },
            "fiscal_years": [current_year - 1, current_year]
        },
        "offset": 0,
        "limit": 1,  # 只需要 total count
        "sort_field": "project_start_date",
        "sort_order": "desc"
    }
    
    try:
        if HAS_REQUESTS:
            headers = {"Content-Type": "application/json"}
            r = requests.post(NIH_REPORTER_BASE, json=payload, headers=headers, timeout=API_TIMEOUT)
            if r.status_code != 200:
                _nih_cache[cache_key] = 0
                return 0
            data = r.json()
        else:
            # urllib fallback
            req_data = json.dumps(payload).encode("utf-8")
            req = Request(
                NIH_REPORTER_BASE,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "Bio-Omics-Index/3.0"}
            )
            with urlopen(req, timeout=API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        
        total_count = data.get("meta", {}).get("total", 0)
        _nih_cache[cache_key] = total_count
        return total_count
    
    except Exception as e:
        log("  NIH RePORTER error: %s", e)
        _nih_cache[cache_key] = 0
        return 0


# ============================================================
# 数据抓取: Google Trends (社区关注度)
# ============================================================

_gtrends_cache: dict[str, float] = {}


def fetch_google_trends_score(keywords: list[str]) -> float:
    """
    获取 Google Trends 相对热度分数 (0-100)。
    使用 pytrends 库，如未安装则返回 0。
    """
    cache_key = keywords[0] if keywords else ""
    if cache_key in _gtrends_cache:
        return _gtrends_cache[cache_key]
    
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log("  pytrends not installed, skipping Google Trends")
        _gtrends_cache[cache_key] = 0.0
        return 0.0
    
    try:
        # 取前 3 个关键词（Google Trends 限制 5 个）
        kw_list = keywords[:3]
        
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m', geo='', gprop='')
        
        interest = pytrends.interest_over_time()
        if interest.empty:
            _gtrends_cache[cache_key] = 0.0
            return 0.0
        
        # 计算平均热度（各关键词的平均值的平均）
        avg_scores = []
        for kw in kw_list:
            if kw in interest.columns:
                avg_scores.append(interest[kw].mean())
        
        score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
        _gtrends_cache[cache_key] = score
        return score
    
    except Exception as e:
        log("  Google Trends error: %s", e)
        _gtrends_cache[cache_key] = 0.0
        return 0.0


# ============================================================
# 数据抓取: Bing Web Search (社区关注度 - 搜索量)
# ============================================================
# Bing Web Search API 返回 totalEstimatedMatches，反映关键词绝对搜索热度。
# 免费层: 1000 次/月，足够周更需求 (23 领域 × 3 关键词 = 69 次/周)
# API Key: 需设置环境变量 BING_API_KEY
# ============================================================

_bing_cache: dict[str, int] = {}


def fetch_bing_search_volume(keywords: list[str]) -> int:
    """
    获取 Bing 搜索结果数量，反映关键词绝对热度。
    
    API: https://api.bing.microsoft.com/v7.0/search
    返回: totalEstimatedMatches (搜索结果估算总数)
    
    Args:
        keywords: 关键词列表，取前 3 个进行查询
        
    Returns:
        搜索结果总数的平均值，若 API 不可用则返回 0
    """
    cache_key = keywords[0] if keywords else ""
    if cache_key in _bing_cache:
        return _bing_cache[cache_key]
    
    if not BING_API_KEY:
        log("  Bing API key not set, returning 0")
        _bing_cache[cache_key] = 0
        return 0
    
    headers = {
        "Ocp-Apim-Subscription-Key": BING_API_KEY,
        "User-Agent": "Bio-Omics-Index/3.0"
    }
    
    total_matches = 0
    query_count = 0
    
    # 查询前 3 个关键词，加 "bioinformatics" 限定领域
    for kw in keywords[:3]:
        query = f"{kw} bioinformatics"
        url = f"{BING_SEARCH_BASE}?q={quote_plus(query)}&count=1&mkt=en-US"
        
        try:
            if HAS_REQUESTS:
                r = requests.get(url, headers=headers, timeout=API_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    matches = data.get("webPages", {}).get("totalEstimatedMatches", 0)
                    total_matches += matches
                    query_count += 1
                elif r.status_code == 401:
                    log("  Bing API: Invalid key")
                    break
                elif r.status_code == 403:
                    log("  Bing API: Rate limit exceeded")
                    break
            else:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    matches = data.get("webPages", {}).get("totalEstimatedMatches", 0)
                    total_matches += matches
                    query_count += 1
            
            _rate_limit(BING_DELAY)
            
        except Exception as e:
            log("  Bing search error for '%s': %s", kw, e)
    
    avg_matches = total_matches // query_count if query_count > 0 else 0
    _bing_cache[cache_key] = avg_matches
    return avg_matches


# ============================================================
# 数据抓取: Semantic Scholar 搜索结果数 (替代 Bing)
# ============================================================
# Semantic Scholar API 免费、无需 API key
# 返回匹配论文总数，作为学术社区关注度的代理指标
# Rate limit: 100 requests / 5 min (无 key)
# ============================================================

_scholar_cache: dict[str, int] = {}


def fetch_scholar_results_count(keywords: list[str]) -> int:
    """
    获取 Semantic Scholar 搜索结果总数，反映学术社区对该领域的关注规模。
    
    API: https://api.semanticscholar.org/graph/v1/paper/search
    返回: total (匹配论文总数)
    
    取前 3 个关键词分别查询，返回平均结果数。
    """
    cache_key = keywords[0] if keywords else ""
    if cache_key in _scholar_cache:
        return _scholar_cache[cache_key]
    
    headers = {
        "User-Agent": "Bio-Omics-Index/3.0 (mailto:bio-rank-gateway@github.io)"
    }
    
    total_results = 0
    query_count = 0
    
    for kw in keywords[:3]:
        query = f"{kw} bioinformatics"
        url = f"{SCHOLAR_SEARCH_BASE}?query={quote_plus(query)}&limit=1&fields=title"
        
        try:
            if HAS_REQUESTS:
                r = requests.get(url, headers=headers, timeout=API_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    total = data.get("total", 0)
                    total_results += total
                    query_count += 1
                elif r.status_code == 429:
                    log("  Scholar API: Rate limit, backing off")
                    _rate_limit(10)
                    continue
            else:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    total = data.get("total", 0)
                    total_results += total
                    query_count += 1
            
            _rate_limit(SCHOLAR_DELAY)
            
        except Exception as e:
            log("  Scholar search error for '%s': %s", kw, e)
    
    avg_results = total_results // query_count if query_count > 0 else 0
    _scholar_cache[cache_key] = avg_results
    return avg_results


# ============================================================
# 数据抓取: OpenAlex 引用动量 (Citation Momentum)
# ============================================================

_openalex_cache: dict[str, float] = {}


def fetch_openalex_citation_momentum(keywords: list[str]) -> float:
    """
    获取领域的引用动量 (Citation Momentum)。
    
    算法:
      - 查询过去 12 个月发表的论文 (按 keywords 搜索)
      - 获取 top 20 高被引论文
      - 返回这些论文的平均 cited_by_count
    
    这个指标反映该领域近期论文的学术影响力。
    高引用动量 = 该领域论文被快速引用 = 学术热度高。
    
    数据源: OpenAlex API (免费, 无需 API Key, polite pool 需 mailto)
    """
    cache_key = "+".join(sorted(keywords[:3]))
    if cache_key in _openalex_cache:
        return _openalex_cache[cache_key]
    
    # 构建搜索词 (取前 3 个关键词)
    search_terms = " ".join(keywords[:3])
    
    # 日期范围: 过去 12 个月
    today = datetime.now()
    from_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    
    # 构建 URL: 搜索 + 日期过滤 + 按引用数降序 + 取前 20
    params = (
        f"?search={quote_plus(search_terms)}"
        f"&filter=from_publication_date:{from_date},to_publication_date:{to_date}"
        f"&sort=cited_by_count:desc"
        f"&per_page=20"
        f"&select=cited_by_count"
        f"&mailto={quote_plus(ENTREZ_EMAIL)}"
    )
    url = OPENALEX_WORKS_BASE + params
    
    try:
        data = _http_get(url)
        if not data or "results" not in data:
            _openalex_cache[cache_key] = 0.0
            return 0.0
        
        results = data["results"]
        if not results:
            _openalex_cache[cache_key] = 0.0
            return 0.0
        
        # 计算 top N 论文的平均引用数
        citations = [r.get("cited_by_count", 0) for r in results]
        avg_citations = sum(citations) / len(citations) if citations else 0.0
        
        _openalex_cache[cache_key] = round(avg_citations, 2)
        return _openalex_cache[cache_key]
        
    except Exception as e:
        log("  OpenAlex error: %s", e)
        _openalex_cache[cache_key] = 0.0
        return 0.0


# ============================================================
# 指数算法 v3.0: 五维度热度模型
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


def calculate_scores_v3(raw_data: dict[str, dict]) -> list[dict[str, Any]]:
    """
    v3.0 五维度热度模型:
    
    Heat Score = 
        学术增长力(30%) + 预印本活跃度(20%) + 技术开发势能(25%)
      + 资金信号(15%) + 社区关注度(10%)
    
    子维度:
      - 学术增长力: PubMed YoY(70%) + OpenAlex 引用动量(30%)
      - 预印本活跃度: bioRxiv 量(60%) + bioRxiv 增长(40%)
      - 技术开发势能: GitHub(30%) + Bioconductor(30%) + PyPI(40%)
      - 资金信号: NIH 项目数(100%)
      - 社区关注度: Google Trends(50%) + Scholar Results(50%)
    
    所有子指标先 Min-Max 归一化到 [0, 1]，再加权求和。
    """
    fields = list(raw_data.keys())
    
    # 提取各维度原始值
    yoy_rates = [raw_data[f]["yoy_rate"] for f in fields]
    biorxiv_counts = [raw_data[f]["biorxiv"] for f in fields]
    biorxiv_growth = [raw_data[f].get("biorxiv_growth", 0.0) for f in fields]
    pubmed_counts = [raw_data[f]["pubmed_current"] for f in fields]
    github_counts = [raw_data[f]["github"] for f in fields]
    bioc_downloads = [raw_data[f]["bioc_downloads"] for f in fields]
    pypi_downloads = [raw_data[f]["pypi_downloads"] for f in fields]
    nih_projects = [raw_data[f]["nih_projects"] for f in fields]
    gtrends_scores = [raw_data[f]["gtrends_score"] for f in fields]
    scholar_results = [raw_data[f]["scholar_results"] for f in fields]
    citation_momentum = [raw_data[f]["citation_momentum"] for f in fields]
    
    # 归一化
    norm_yoy = normalize_values(yoy_rates)
    norm_biorxiv = normalize_values(biorxiv_counts)
    norm_biorxiv_growth = normalize_values(biorxiv_growth)
    norm_pubmed = normalize_values(pubmed_counts)
    norm_github = normalize_values(github_counts)
    norm_bioc = normalize_values(bioc_downloads)
    norm_pypi = normalize_values(pypi_downloads)
    norm_nih = normalize_values(nih_projects)
    norm_gtrends = normalize_values(gtrends_scores)
    norm_scholar = normalize_values(scholar_results)
    norm_citation = normalize_values(citation_momentum)
    
    results = []
    for i, field in enumerate(fields):
        data = raw_data[field]
        
        # 1. 学术增长力 (30%) = YoY(70%) + Citation Momentum(30%)
        academic_score = (
            norm_yoy[i] * WEIGHT_PUBMED_YOY +
            norm_citation[i] * WEIGHT_CITATION_MOMENTUM
        )
        
        # 2. 预印本活跃度 (20%) = bioRxiv数量(60%) + bioRxiv增长率(40%)
        preprint_score = (
            norm_biorxiv[i] * WEIGHT_BIORXIV_COUNT +
            norm_biorxiv_growth[i] * WEIGHT_BIORXIV_GROWTH
        )
        
        # 3. 技术开发势能 (25%)
        tech_score = (
            norm_github[i] * WEIGHT_GITHUB +
            norm_bioc[i] * WEIGHT_BIOC +
            norm_pypi[i] * WEIGHT_PYPI
        )
        
        # 4. 资金信号 (15%)
        funding_score = norm_nih[i]
        
        # 5. 社区关注度 (10%) = Google Trends(50%) + Scholar Results(50%)
        community_score = (
            norm_gtrends[i] * WEIGHT_GTRENDS +
            norm_scholar[i] * WEIGHT_SCHOLAR
        )
        
        # 总分
        heat_score = (
            academic_score * WEIGHT_ACADEMIC +
            preprint_score * WEIGHT_PREPRINT +
            tech_score * WEIGHT_TECH +
            funding_score * WEIGHT_FUNDING +
            community_score * WEIGHT_COMMUNITY
        )
        
        # 动能等级 (使用实际 bioRxiv 增长率)
        actual_biorxiv_growth = data.get("biorxiv_growth", 0.0)
        momentum = calculate_momentum_tier(data["yoy_rate"], actual_biorxiv_growth)
        
        results.append({
            "field": field,
            "category": data.get("category", ""),
            # 原始数据
            "pubmed_current": data["pubmed_current"],
            "pubmed_prev": data["pubmed_prev"],
            "yoy_rate": data["yoy_rate"],
            "biorxiv_count": data["biorxiv"],
            "biorxiv_growth": actual_biorxiv_growth,
            "github_activity": data["github"],
            "bioc_downloads": data["bioc_downloads"],
            "pypi_downloads": data["pypi_downloads"],
            "nih_projects": data["nih_projects"],
            "gtrends_score": round(data["gtrends_score"], 2),
            "scholar_results": data["scholar_results"],
            "citation_momentum": data["citation_momentum"],
            # 五维度分数 (调试用)
            "academic_score": round(academic_score, 4),
            "preprint_score": round(preprint_score, 4),
            "tech_score": round(tech_score, 4),
            "funding_score": round(funding_score, 4),
            "community_score": round(community_score, 4),
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
    
    v3.0 采集项 (10 项指标):
      - PubMed YoY (滚动12个月 vs 前12个月 增长率)
      - bioRxiv 6 个月预印本
      - GitHub 活跃度
      - Bioconductor 下载量 (R 生态)
      - PyPI 下载量 (Python 生态)
      - NIH 资助项目数 (资金信号)
      - Google Trends 热度 (社区关注度)
      - Semantic Scholar 搜索结果数 (社区关注度)
      - OpenAlex 引用动量 (学术增长力)
    """
    raw_data = {}
    n = len(OMICS_DICT)
    
    for i, (field, config) in enumerate(OMICS_DICT.items(), 1):
        keywords = config["keywords"]
        category = config.get("category", "")
        bioc_packages = config.get("bioc_packages", [])
        pypi_packages = config.get("pypi_packages", [])
        
        log("[%d/%d] %s", i, n, field)
        
        # 1) PubMed YoY
        pubmed_current, pubmed_prev, yoy_rate = fetch_pubmed_yoy(keywords)
        log("  PubMed 12m: %d vs prev 12m: %d → YoY: %+.1f%%",
            pubmed_current, pubmed_prev, yoy_rate)
        _rate_limit(NCBI_DELAY)
        
        # 2) bioRxiv
        biorxiv_count = fetch_biorxiv_count(keywords)
        log("  bioRxiv (6m): %s", biorxiv_count)
        _rate_limit(NCBI_DELAY)
        
        # 3) GitHub
        github_activity = fetch_github_activity(keywords[0])
        log("  GitHub: %s", github_activity)
        _rate_limit(GITHUB_DELAY)
        
        # 4) Bioconductor (R 生态)
        bioc_downloads = fetch_bioc_downloads_for_field(bioc_packages)
        log("  Bioconductor (%d pkgs): %s downloads/year", len(bioc_packages), bioc_downloads)
        
        # 5) PyPI (Python 生态)
        pypi_downloads = fetch_pypi_downloads_for_field(pypi_packages)
        log("  PyPI (%d pkgs): %s downloads/6m", len(pypi_packages), pypi_downloads)
        
        # 6) NIH RePORTER (资金信号)
        nih_projects = fetch_nih_funding_projects(keywords)
        log("  NIH Projects (2y): %s", nih_projects)
        _rate_limit(NIH_DELAY)
        
        # 7) Google Trends (社区关注度)
        gtrends_score = fetch_google_trends_score(keywords)
        log("  Google Trends: %.1f", gtrends_score)
        _rate_limit(TRENDS_DELAY)
        
        # 8) Semantic Scholar (社区关注度 - 学术搜索量)
        scholar_results = fetch_scholar_results_count(keywords)
        log("  Scholar Results: %s papers", scholar_results)
        _rate_limit(SCHOLAR_DELAY)
        
        # 9) OpenAlex Citation Momentum (学术增长力子维度)
        citation_momentum = fetch_openalex_citation_momentum(keywords)
        log("  OpenAlex Citations: %.1f avg (top 20 papers, 12m)", citation_momentum)
        _rate_limit(OPENALEX_DELAY)
        
        raw_data[field] = {
            "pubmed_current": pubmed_current,
            "pubmed_prev": pubmed_prev,
            "yoy_rate": yoy_rate,
            "biorxiv": biorxiv_count,
            "github": github_activity,
            "bioc_downloads": bioc_downloads,
            "pypi_downloads": pypi_downloads,
            "nih_projects": nih_projects,
            "gtrends_score": gtrends_score,
            "scholar_results": scholar_results,
            "citation_momentum": citation_momentum,
            "category": category,
        }
    
    return raw_data


def generate_omics_index() -> dict[str, Any]:
    """生成 Bio-Omics Heat Index v3.0 并写入 docs/data/omics_index.json。"""
    log("=" * 70)
    log("Bio-Omics Heat Index v3.0")
    log("=" * 70)
    log("Five-Dimension Model: Academic(30%%) + Preprint(20%%) + Tech(25%%) + Funding(15%%) + Community(10%%)")
    log("")
    
    if not HAS_BIOPYTHON:
        log("Warning: Biopython not installed. Using URL fallback for PubMed.")
    
    # 1) 数据采集 (v3.0: 10 项指标)
    log("\n[Phase 1] Collecting data (10 indicators)...")
    log("  - PubMed YoY, bioRxiv, GitHub, OpenAlex Citations")
    log("  - Bioconductor (R), PyPI (Python)")
    log("  - NIH RePORTER, Google Trends, Semantic Scholar")
    raw_data = collect_all_data()
    
    # 1.5) bioRxiv 增长率 (需要历史数据)
    log("\n[Phase 1.5] Calculating bioRxiv growth rates...")
    biorxiv_history = load_biorxiv_history()
    current_biorxiv = {field: data["biorxiv"] for field, data in raw_data.items()}
    biorxiv_growth = calculate_biorxiv_growth(current_biorxiv, biorxiv_history)
    
    # 将增长率合并到 raw_data
    for field in raw_data:
        raw_data[field]["biorxiv_growth"] = biorxiv_growth.get(field, 0.0)
    
    # 保存当前 bioRxiv 数据到历史
    today_str = datetime.now().strftime("%Y-%m-%d")
    biorxiv_history[today_str] = current_biorxiv
    save_biorxiv_history(biorxiv_history)
    log("  bioRxiv history saved (%d weeks)", len(biorxiv_history))
    
    # 2) 计分与归一化 (v3.0 五维度模型)
    log("\n[Phase 2] Calculating Heat Scores (v3.0 Five-Dimension Model)...")
    rankings = calculate_scores_v3(raw_data)
    
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
        "version": "3.0",
        "methodology": {
            "formula": "Heat = Academic(30%) + Preprint(20%) + Tech(25%) + Funding(15%) + Community(10%)",
            "weights": {
                "academic": WEIGHT_ACADEMIC,
                "preprint": WEIGHT_PREPRINT,
                "tech": WEIGHT_TECH,
                "funding": WEIGHT_FUNDING,
                "community": WEIGHT_COMMUNITY,
            },
            "sub_weights": {
                "academic_pubmed_yoy": WEIGHT_PUBMED_YOY,
                "academic_citation": WEIGHT_CITATION_MOMENTUM,
                "preprint_biorxiv_count": WEIGHT_BIORXIV_COUNT,
                "preprint_biorxiv_growth": WEIGHT_BIORXIV_GROWTH,
                "tech_github": WEIGHT_GITHUB,
                "tech_bioconductor": WEIGHT_BIOC,
                "tech_pypi": WEIGHT_PYPI,
                "community_gtrends": WEIGHT_GTRENDS,
                "community_scholar": WEIGHT_SCHOLAR,
            },
            "pubmed_period": "rolling 12-month vs previous 12-month",
            "citation_period": "top 20 papers (12 months) avg cited_by_count via OpenAlex",
            "biorxiv_period": "last 6 months (PubMed Preprint[pt])",
            "biorxiv_growth_period": "week-over-week comparison from history",
            "github_period": "last 6 months new repos + total (stars>10)",
            "bioconductor_period": "last 12 months downloads",
            "pypi_period": "last 6 months downloads (estimated)",
            "nih_period": f"{current_year - 1}-{current_year} funded projects",
            "gtrends_period": "last 12 months relative interest",
            "scholar_period": "total matching papers via Semantic Scholar API",
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
    log("\n" + "=" * 120)
    log("Bio-Omics Heat Index v3.0 (Top 20)")
    log("=" * 120)
    log("%-4s %-20s %6s %7s %9s %7s %6s %9s %9s %6s %8s %8s",
        "Rank", "Field", "Share%", "YoY%", "Momentum", "PubMed", "bioRxiv", "GitHub", "BiocDL", "PyPI", "NIH", "Scholar")
    log("-" * 120)
    for item in rankings[:20]:
        yoy_str = f"{item['yoy_rate']:+.1f}%"
        # 简化大数字显示
        pypi_str = f"{item['pypi_downloads'] // 1000}k" if item['pypi_downloads'] >= 1000 else str(item['pypi_downloads'])
        bioc_str = f"{item['bioc_downloads'] // 1000}k" if item['bioc_downloads'] >= 1000 else str(item['bioc_downloads'])
        scholar_str = f"{item['scholar_results'] // 1000}k" if item['scholar_results'] >= 1000 else str(item['scholar_results'])
        log(
            "%-4s %-20s %5.2f%% %7s %9s %7s %7s %9s %9s %6s %8s %8s",
            item["rank"],
            item["field"][:20],
            item["share_pct"],
            yoy_str,
            item["momentum"],
            item["pubmed_current"],
            item["biorxiv_count"],
            item["github_activity"],
            bioc_str,
            pypi_str,
            item["nih_projects"],
            scholar_str,
        )
    log("\nGenerated at: %s", report["generated_at"])
    return report


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1. 排名依据: 各领域按五维度加权总分排序
#    Heat Score = 学术(30%) + 预印本(20%) + 技术(25%) + 资金(15%) + 社区(10%)
#
# 2. 学术增长力 (30%): 
#    - PubMed YoY 增长率 (70%): 当年 vs 去年发文量变化
#    - 被引/高影响论文 (30%): Phase 2 通过 OpenAlex 实现
#
# 3. 预印本活跃度 (20%):
#    - bioRxiv 6 个月发文量 (60%): 反映近期研究热度
#    - bioRxiv 增长率 (40%): 需要历史数据，Phase 2 实现
#
# 4. 技术开发势能 (25%):
#    - GitHub 活跃度 (30%): 近 6 个月新仓库 + 总仓库数
#    - Bioconductor R 包下载 (30%): 反映 R 生态工具使用
#    - PyPI Python 包下载 (40%): 反映 Python 生态工具使用
#
# 5. 资金信号 (15%):
#    - NIH RePORTER 项目数: 近 2 年资助项目数量
#
# 6. 社区关注度 (10%):
#    - Google Trends (50%): 12 个月相对搜索热度
#    - Bing Search (50%): 搜索结果数量
#
# 7. 归一化: 所有子指标 Min-Max 归一化到 [0, 1]，避免绝对量主导
#
# 8. 动能等级: 基于 YoY 增长率分级
#    - Rising Star (>30%), Hot (>15%), Growing (>5%), Stable (>-5%), Cooling (<-5%)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_omics_index()
