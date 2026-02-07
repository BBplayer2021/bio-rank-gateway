#!/usr/bin/env python3
"""
Bio-Rank Gateway v13.0
全自动化生信排行榜门户系统

功能:
1. 数据抓取与评分
2. 安装命令识别
3. 预览图抓取
4. 勋章URL生成
5. 新进榜通知
"""

import requests
import time
import sqlite3
import base64
import json
import math
import re
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN: Optional[str] = os.environ.get("GITHUB_TOKEN")
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "data" / "bio_toolbox.db"
DATA_DIR = PROJECT_DIR / "data"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 关键词字典
KEYWORDS = {
    "Genomics": [
        "WGS", "Variant Calling", "Genome Assembly", "Structural Variant",
        "GATK", "Haplotype", "Pangenome", "VCF", "BAM", "FASTQ",
        "BWA", "Bowtie", "samtools", "bcftools", "CRISPR"
    ],
    "Transcriptomics": [
        "RNA-seq", "Differential Expression", "Alternative Splicing",
        "Isoform", "DESeq2", "edgeR", "kallisto", "salmon", "STAR aligner",
        "featureCounts", "HTSeq", "transcript assembly"
    ],
    "Metagenomics": [
        "Metagenome", "16S rRNA", "Taxonomic Profiling", "Metabolic Reconstruction",
        "MAGs", "Kraken", "MetaPhlAn", "QIIME", "Mothur", "amplicon",
        "microbiome", "OTU", "ASV"
    ],
    "Single-cell": [
        "scRNA-seq", "single-cell", "10x Genomics", "Scanpy", "Seurat",
        "Cell Ranger", "spatial transcriptomics", "UMAP", "trajectory",
        "cell clustering", "droplet"
    ],
    "Epigenetics": [
        "ATAC-seq", "ChIP-seq", "DNA Methylation", "Hi-C", "Chromatin",
        "Cut&Tag", "CUT&RUN", "bisulfite", "WGBS", "peak calling",
        "MACS2", "Homer"
    ]
}

# Pipeline 检测关键词
PIPELINE_KEYWORDS = ["nextflow", "snakemake", "cwl", "wdl", "workflow", "pipeline", "nf-core"]

# 细分领域标签映射
UTILITY_LABELS = {
    "Alignment": ["bwa", "bowtie", "star", "hisat", "minimap", "align"],
    "Variant Calling": ["gatk", "variant", "snp", "vcf", "bcftools", "freebayes", "deepvariant"],
    "Differential Expression": ["deseq", "edger", "limma", "differential", "pydeseq"],
    "Clustering": ["cluster", "leiden", "louvain", "umap", "tsne", "pca"],
    "Cell Annotation": ["celltypist", "scgate", "annotation", "marker", "cell type"],
    "Trajectory": ["trajectory", "pseudotime", "velocity", "monocle", "dynamo"],
    "Quality Control": ["qc", "fastqc", "multiqc", "quality", "trimming"],
    "Assembly": ["assembly", "spades", "megahit", "flye", "canu"],
    "Taxonomic": ["kraken", "metaphlan", "taxonom", "classifier", "16s"],
    "Visualization": ["plot", "visual", "track", "igv", "genome browser"],
    "Format Conversion": ["convert", "bam", "sam", "fastq", "format"],
    "Peak Calling": ["macs", "peak", "chip-seq", "atac-seq", "homer"],
    "Methylation": ["bismark", "methylat", "bisulfite", "cpg"],
    "Quantification": ["salmon", "kallisto", "rsem", "count", "tpm", "fpkm"],
    "Fusion Detection": ["fusion", "arriba", "star-fusion", "fusioncatcher"],
    "CNV Analysis": ["cnv", "copy number", "infercnv", "numbat"],
    "Cell Communication": ["ligand", "receptor", "nichenet", "cellchat", "communication"]
}

# 排除关键词
EXCLUDE_KEYWORDS = ["notes", "exercise", "homework", "tutorial", "learning", "course", "awesome-"]

# Pipeline 组织白名单
PIPELINE_ORG_WHITELIST = [
    "nf-core", "snakemake-workflows", "nextflow-io", "bcbio", "snakepipes",
    "bioconda", "qbic-pipelines", "hoelzer-lab"
]

# 强制归类白名单/黑名单
FORCE_PIPELINE_REPOS = [
    "yongxinliu/easymetagenome", "shujiahuang/ilus", "metagenome-atlas/atlas",
    "ebi-gene-expression-group/scxa-tertiary-workflow", "sequana/sequana",
    "maxplanck-ie/snakepipes",
]

FORCE_UTILITY_REPOS = [
    "lh3/bwa", "broadinstitute/gatk", "scverse/pydeseq2", "saeyslab/nichenetr",
    "aertslab/pyscenic", "pachterlab/kallisto", "alexdobin/star",
    "thelovelab/deseq2", "gpertea/stringtie", "bwa-mem2/bwa-mem2",
    "deeptools/deeptools", "trinityrnaseq/trinityrnaseq", "pachterlab/gget",
    "broadinstitute/infercnv", "marbl/mash", "ecogenomics/checkm",
]

# 端到端流程关键词
END_TO_END_KEYWORDS = [
    "end-to-end", "from raw data", "one-stop", "complete analysis",
    "fastq to", "raw to results", "from fastq", "automated pipeline",
    "complete workflow", "full pipeline", "comprehensive pipeline",
    "turn-key", "all-in-one", "ready-to-use pipeline"
]

# Pipeline 语义关键词
PIPELINE_SEMANTIC_KEYWORDS = [
    "pipeline", "workflow", "automated", "ngs pipeline",
    "analysis pipeline", "processing pipeline", "bioinformatics pipeline"
]

# R/Python 库特征
LIBRARY_INDICATORS = [
    "setup.py", "setup.cfg", "pyproject.toml", "description",
    "r package", "python package", "cran", "bioconductor package",
    "pip install", "install.packages"
]


# ============================================================
# 工具函数
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def get_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def is_excluded(repo: dict) -> bool:
    description = (repo.get("description") or "").lower()
    name = repo.get("name", "").lower()
    for kw in EXCLUDE_KEYWORDS:
        if kw in description or kw in name:
            return True
    return False


# ============================================================
# 数据库操作
# ============================================================

def init_database(reset: bool = False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if reset:
        cursor.execute("DROP TABLE IF EXISTS snapshots")
        cursor.execute("DROP TABLE IF EXISTS repositories")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repositories (
            id INTEGER PRIMARY KEY,
            full_name TEXT UNIQUE NOT NULL,
            url TEXT,
            description TEXT,
            language TEXT,
            license TEXT,
            topics TEXT,
            created_at TEXT,
            first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            project_type TEXT,
            has_paper INTEGER DEFAULT 0,
            has_docker INTEGER DEFAULT 0,
            has_conda_env INTEGER DEFAULT 0,
            sub_label TEXT,
            tech_stack TEXT,
            install_commands TEXT,
            preview_images TEXT,
            badge_url TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL,
            stars INTEGER,
            forks INTEGER,
            watchers INTEGER,
            open_issues INTEGER,
            pushed_at TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (repo_id) REFERENCES repositories(id)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repo_category ON repositories(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repo_type ON repositories(project_type)")
    
    conn.commit()
    conn.close()
    log(f"[DB] Initialized: {DB_PATH}")


def save_repo_with_snapshot(repo: dict, category: str, project_type: str, 
                            has_paper: bool, has_docker: bool, has_conda_env: bool,
                            sub_label: str, install_commands: list = None,
                            preview_images: list = None, badge_url: str = "") -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    full_name = repo.get("full_name")
    topics = json.dumps(repo.get("topics", []))
    license_info = repo.get("license") or {}
    license_name = license_info.get("name", "") if isinstance(license_info, dict) else ""
    
    cursor.execute("""
        INSERT INTO repositories (id, full_name, url, description, language, license, topics, 
                                  created_at, category, project_type, has_paper, has_docker, 
                                  has_conda_env, sub_label, install_commands, preview_images, badge_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(full_name) DO UPDATE SET
            description = excluded.description,
            language = excluded.language,
            topics = excluded.topics,
            category = excluded.category,
            project_type = excluded.project_type,
            has_paper = excluded.has_paper,
            has_docker = excluded.has_docker,
            has_conda_env = excluded.has_conda_env,
            sub_label = excluded.sub_label,
            install_commands = excluded.install_commands,
            preview_images = excluded.preview_images,
            badge_url = excluded.badge_url
    """, (
        repo.get("id"), full_name, repo.get("html_url"), repo.get("description"),
        repo.get("language"), license_name, topics, repo.get("created_at"),
        category, project_type, 
        1 if has_paper else 0, 1 if has_docker else 0, 1 if has_conda_env else 0,
        sub_label, json.dumps(install_commands or []), json.dumps(preview_images or []), badge_url
    ))
    
    cursor.execute("SELECT id FROM repositories WHERE full_name = ?", (full_name,))
    repo_id = cursor.fetchone()[0]
    
    cursor.execute("""
        INSERT INTO snapshots (repo_id, stars, forks, watchers, open_issues, pushed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        repo_id,
        repo.get("stargazers_count", 0),
        repo.get("forks_count", 0),
        repo.get("watchers_count", 0),
        repo.get("open_issues_count", 0),
        repo.get("pushed_at")
    ))
    
    conn.commit()
    conn.close()
    return repo_id


def get_weekly_star_growth(repo_id: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT stars FROM snapshots WHERE repo_id = ? ORDER BY fetched_at DESC LIMIT 2
    """, (repo_id,))
    rows = cursor.fetchall()
    conn.close()
    if len(rows) >= 2:
        return max(0, rows[0][0] - rows[1][0])
    return 0


def get_all_repos_for_ranking() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            r.id, r.full_name, r.url, r.description, r.category, 
            r.project_type, r.has_paper, r.has_docker, r.has_conda_env,
            r.sub_label, r.topics, r.license, r.tech_stack,
            r.install_commands, r.preview_images, r.badge_url,
            s.stars, s.forks, s.open_issues, s.pushed_at
        FROM repositories r
        JOIN (
            SELECT repo_id, stars, forks, open_issues, pushed_at,
                   ROW_NUMBER() OVER (PARTITION BY repo_id ORDER BY fetched_at DESC) as rn
            FROM snapshots
        ) s ON r.id = s.repo_id AND s.rn = 1
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


# ============================================================
# GitHub API
# ============================================================

def search_with_pagination(query: str, min_results: int = 50) -> list:
    url = f"{GITHUB_API_BASE}/search/repositories"
    all_items = []
    page = 1
    per_page = 100
    
    while len(all_items) < min_results and page <= 5:
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page}
        try:
            response = requests.get(url, headers=get_headers(), params=params, timeout=30)
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            
            if response.status_code == 403:
                log(f"    [Rate Limited] Waiting...")
                time.sleep(60)
                continue
            
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                break
            
            all_items.extend(items)
            log(f"    [Page {page}] Got {len(items)} items (Total: {len(all_items)}, API: {remaining})")
            
            if len(items) < per_page:
                break
            page += 1
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            log(f"    [Error] {e}")
            break
    
    return all_items


def get_readme_content(owner: str, repo_name: str) -> str:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/readme"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            content = response.json().get("content", "")
            return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        pass
    return ""


# ============================================================
# 数据增强: 安装命令/预览图/勋章
# ============================================================

def extract_install_commands(readme_content: str) -> List[Dict[str, str]]:
    """从README中提取安装命令"""
    commands = []
    patterns = [
        (r'conda install\s+[\w\-\.]+(?:\s+[\w\-\.=<>]+)*', 'conda'),
        (r'pip install\s+[\w\-\.]+(?:\[[\w,]+\])?', 'pip'),
        (r'docker pull\s+[\w\-\./:]+', 'docker'),
        (r'git clone\s+https?://[\w\-\./:]+', 'git'),
        (r'mamba install\s+[\w\-\.]+(?:\s+[\w\-\.=<>]+)*', 'mamba'),
        (r'brew install\s+[\w\-\.]+', 'brew'),
    ]
    
    for pattern, cmd_type in patterns:
        matches = re.findall(pattern, readme_content, re.IGNORECASE)
        for match in matches[:3]:  # 每种类型最多取3个
            commands.append({
                "type": cmd_type,
                "command": match.strip()
            })
    
    return commands[:5]  # 最多返回5个命令


def extract_preview_images(readme_content: str, repo_url: str) -> List[str]:
    """从README中提取预览图和Logo"""
    images = []
    logo_images = []  # Logo图片优先级最高
    
    # 从repo_url提取owner和repo名
    # https://github.com/owner/repo -> owner/repo
    repo_full_name = ""
    if "github.com" in repo_url:
        parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
        if len(parts) >= 2:
            repo_full_name = f"{parts[0]}/{parts[1]}"
    
    def convert_to_raw_url(url: str) -> str:
        """将GitHub相对路径或blob路径转换为raw URL"""
        url = url.strip()
        
        # 已经是完整URL
        if url.startswith('http://') or url.startswith('https://'):
            # 转换blob URL为raw URL
            if 'github.com' in url and '/blob/' in url:
                url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            return url
        
        # GitHub绝对路径格式: /owner/repo/raw/branch/path 或 /owner/repo/blob/branch/path
        if url.startswith('/') and '/raw/' in url:
            return f"https://raw.githubusercontent.com{url.replace('/raw/', '/', 1)}"
        
        if url.startswith('/') and '/blob/' in url:
            # /owner/repo/blob/master/path -> https://raw.githubusercontent.com/owner/repo/master/path
            url = url.replace('/blob/', '/', 1)
            return f"https://raw.githubusercontent.com{url}"
        
        # 相对路径格式: ./assets/logo.png 或 assets/logo.png
        if repo_full_name:
            clean_path = url.lstrip('./')
            # 尝试多个分支
            return f"https://raw.githubusercontent.com/{repo_full_name}/master/{clean_path}"
        
        return url
    
    # Logo关键词 - 最高优先级
    logo_keywords = ['logo', 'banner', 'header', 'brand']
    
    # 优先关键词
    priority_keywords = ['workflow', 'report', 'plot', 'result', 'output', 'diagram', 
                         'overview', 'pipeline', 'screenshot', 'example', 'figure', 'dag']
    
    # 排除的徽章关键词
    badge_keywords = ['badge', 'shields.io', 'travis', 'codecov', 'circleci', 'coveralls',
                      'github.io/badge', 'img.shields', 'badgen.net', 'fury.io']
    
    # 1. 提取<picture>元素中的图片 (nf-core风格)
    picture_pattern = r'<picture[^>]*>.*?<img[^>]+src=["\']([^"\']+)["\'].*?</picture>'
    for url in re.findall(picture_pattern, readme_content, re.DOTALL | re.IGNORECASE):
        url_lower = url.lower()
        if any(badge in url_lower for badge in badge_keywords):
            continue
        
        full_url = convert_to_raw_url(url)
        is_logo = any(kw in url_lower for kw in logo_keywords)
        
        if is_logo:
            if full_url not in logo_images:
                logo_images.append(full_url)
        elif full_url not in images:
            images.append(full_url)
    
    # 2. 提取<source>标签中的图片
    source_pattern = r'<source[^>]+srcset=["\']([^"\']+)["\']'
    for url in re.findall(source_pattern, readme_content, re.IGNORECASE):
        url_lower = url.lower()
        if any(badge in url_lower for badge in badge_keywords):
            continue
        
        full_url = convert_to_raw_url(url)
        is_logo = any(kw in url_lower for kw in logo_keywords)
        
        if is_logo:
            if full_url not in logo_images:
                logo_images.append(full_url)
    
    # 3. 提取HTML <img>标签 (包括GitHub风格的相对路径)
    html_pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    for url in re.findall(html_pattern, readme_content, re.IGNORECASE):
        url_lower = url.lower()
        if any(badge in url_lower for badge in badge_keywords):
            continue
        
        full_url = convert_to_raw_url(url)
        is_logo = any(kw in url_lower for kw in logo_keywords)
        is_priority = any(kw in url_lower for kw in priority_keywords)
        
        if is_logo:
            if full_url not in logo_images:
                logo_images.append(full_url)
        elif is_priority:
            if full_url not in images:
                images.insert(0, full_url)
        else:
            if full_url not in images:
                images.append(full_url)
    
    # 4. 提取<a>标签中href指向的图片 (MpGAP风格)
    a_img_pattern = r'<a[^>]+href=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg|webp))["\']'
    for url in re.findall(a_img_pattern, readme_content, re.IGNORECASE):
        url_lower = url.lower()
        if any(badge in url_lower for badge in badge_keywords):
            continue
        
        full_url = convert_to_raw_url(url)
        is_logo = any(kw in url_lower for kw in logo_keywords)
        
        if is_logo:
            if full_url not in logo_images:
                logo_images.append(full_url)
        elif full_url not in images:
            images.append(full_url)
    
    # 5. 提取Markdown图片语法
    md_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    for alt, url in re.findall(md_pattern, readme_content):
        alt_lower = alt.lower()
        url_lower = url.lower()
        
        if any(badge in url_lower for badge in badge_keywords):
            continue
        
        full_url = convert_to_raw_url(url)
        is_logo = any(kw in alt_lower or kw in url_lower for kw in logo_keywords)
        is_priority = any(kw in alt_lower or kw in url_lower for kw in priority_keywords)
        
        if is_logo:
            if full_url not in logo_images:
                logo_images.append(full_url)
        elif is_priority:
            if full_url not in images:
                images.insert(0, full_url)
        else:
            if full_url not in images:
                images.append(full_url)
    
    # Logo图片放在最前面，然后是其他图片
    result = logo_images + images
    
    # 去重并返回前5张
    seen = set()
    unique_result = []
    for img in result:
        if img not in seen:
            seen.add(img)
            unique_result.append(img)
    
    return unique_result[:5]


def generate_badge_url(full_name: str, stars: int, language: str = "") -> str:
    """生成Shields.io勋章URL"""
    # Stars勋章
    stars_badge = f"https://img.shields.io/github/stars/{full_name}?style=flat-square&logo=github"
    return stars_badge


def generate_all_badges(full_name: str, stars: int, language: str = "", 
                        has_docker: bool = False, has_conda: bool = False) -> Dict[str, str]:
    """生成所有勋章URL"""
    badges = {
        "stars": f"https://img.shields.io/github/stars/{full_name}?style=flat-square&logo=github",
        "forks": f"https://img.shields.io/github/forks/{full_name}?style=flat-square&logo=github",
        "issues": f"https://img.shields.io/github/issues/{full_name}?style=flat-square",
        "license": f"https://img.shields.io/github/license/{full_name}?style=flat-square",
        "last_commit": f"https://img.shields.io/github/last-commit/{full_name}?style=flat-square",
    }
    
    if language:
        badges["language"] = f"https://img.shields.io/github/languages/top/{full_name}?style=flat-square"
    
    if has_docker:
        badges["docker"] = "https://img.shields.io/badge/docker-available-blue?style=flat-square&logo=docker"
    
    if has_conda:
        badges["conda"] = "https://img.shields.io/badge/conda-available-green?style=flat-square&logo=anaconda"
    
    return badges


# ============================================================
# 分类器
# ============================================================

def detect_project_type(repo: dict, readme_content: str = "") -> str:
    """分类器: 端到端能力判定 + 白名单/黑名单机制"""
    description = (repo.get("description") or "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    name = repo.get("name", "").lower()
    full_name = repo.get("full_name", "").lower()
    owner = full_name.split("/")[0] if "/" in full_name else ""
    readme_lower = readme_content.lower()
    readme_head = readme_lower[:2000]
    combined_text = f"{description} {readme_head}"
    
    # 1. 强制 Utility (黑名单)
    if full_name in FORCE_UTILITY_REPOS:
        return "Utility"
    
    # 2. 强制 Pipeline (白名单)
    if full_name in FORCE_PIPELINE_REPOS:
        return "Pipeline"
    
    # 3. 组织白名单
    for org in PIPELINE_ORG_WHITELIST:
        if owner == org or org in full_name:
            return "Pipeline"
    
    # 4. 一票否决权: R/Python 库检测
    library_score = 0
    for indicator in LIBRARY_INDICATORS:
        if indicator in readme_lower:
            library_score += 1
    
    if library_score >= 2:
        has_e2e = any(kw in combined_text for kw in END_TO_END_KEYWORDS)
        if not has_e2e:
            return "Utility"
    
    # 5. 端到端关键词检测
    for kw in END_TO_END_KEYWORDS:
        if kw in combined_text:
            return "Pipeline"
    
    # 6. 语义特征评分
    pipeline_score = 0
    
    for kw in ["nextflow", "snakemake"]:
        if kw in description:
            pipeline_score += 5
        if kw in name:
            pipeline_score += 4
        if kw in topics:
            pipeline_score += 4
        if kw in readme_head:
            pipeline_score += 2
    
    config_patterns = ["nextflow.config", "main.nf", "snakefile", ".smk", ".wdl"]
    for pattern in config_patterns:
        if pattern in readme_lower:
            pipeline_score += 3
    
    for kw in PIPELINE_SEMANTIC_KEYWORDS:
        if kw in description:
            pipeline_score += 2
        if kw in readme_head[:500]:
            pipeline_score += 1
    
    pipeline_topics = ["pipeline", "workflow", "nextflow", "snakemake", "cwl", "wdl"]
    for kw in pipeline_topics:
        if kw in topics:
            pipeline_score += 3
    
    dir_patterns = ["workflows/", "rules/", "modules/", "subworkflows/"]
    for pattern in dir_patterns:
        if pattern in readme_lower:
            pipeline_score += 2
    
    return "Pipeline" if pipeline_score >= 5 else "Utility"


def detect_environment_support(readme_content: str) -> Tuple[bool, bool]:
    """检测 Docker 和 Conda 环境支持"""
    readme_lower = readme_content.lower()
    
    has_docker = any(kw in readme_lower for kw in [
        "dockerfile", "docker pull", "docker run", "docker-compose",
        "container", "singularity", "biocontainer"
    ])
    
    has_conda_env = any(kw in readme_lower for kw in [
        "environment.yml", "environment.yaml", "conda env create",
        "conda install", "bioconda", "mamba install"
    ])
    
    return has_docker, has_conda_env


def detect_has_paper(readme_content: str) -> bool:
    readme_lower = readme_content.lower()
    patterns = [r"10\.\d{4,}/", r"pubmed", r"pmid", r"doi\.org", r"citation", 
                r"cite this", r"published in", r"biorxiv", r"arxiv"]
    return any(re.search(p, readme_lower) for p in patterns)


def detect_sub_label(repo: dict, readme_content: str = "") -> str:
    """检测 Utility 细分领域标签"""
    description = (repo.get("description") or "").lower()
    name = repo.get("name", "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    readme_lower = readme_content.lower()
    full_text = f"{description} {name} {' '.join(topics)} {readme_lower[:2000]}"
    
    for label, keywords in UTILITY_LABELS.items():
        if any(kw in full_text for kw in keywords):
            return label
    
    return "General"


def classify_category(repo: dict, search_category: str) -> str:
    description = (repo.get("description") or "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    name = repo.get("name", "").lower()
    full_text = f"{description} {name} {' '.join(topics)}"
    
    matched_categories = []
    
    if any(kw in full_text for kw in ["single-cell", "scrna", "10x", "scanpy", "seurat", 
                                       "cell ranger", "droplet", "spatial transcriptomics"]):
        matched_categories.append("Single-cell")
    
    if any(kw in full_text for kw in ["metagenom", "16s", "microbiome", "taxonom", "kraken", 
                                       "metaphlan", "qiime", "mothur", "amplicon"]):
        matched_categories.append("Metagenomics")
    
    if any(kw in full_text for kw in ["atac-seq", "chip-seq", "methylat", "hi-c", "chromatin",
                                       "cut&tag", "bisulfite", "epigenom"]):
        matched_categories.append("Epigenetics")
    
    if any(kw in full_text for kw in ["rna-seq", "rnaseq", "transcript", "differential expression",
                                       "deseq", "edger", "kallisto", "salmon"]):
        matched_categories.append("Transcriptomics")
    
    if not matched_categories:
        return search_category if search_category in KEYWORDS else "Genomics"
    
    return matched_categories[0]


def get_multi_categories(repo: dict) -> list:
    """获取项目的所有匹配类别（一库多标）"""
    description = (repo.get("description") or "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    name = repo.get("name", "").lower()
    full_text = f"{description} {name} {' '.join(topics)}"
    
    matched_categories = []
    
    if any(kw in full_text for kw in ["single-cell", "scrna", "10x", "scanpy", "seurat", 
                                       "cell ranger", "droplet", "spatial transcriptomics"]):
        matched_categories.append("Single-cell")
    
    if any(kw in full_text for kw in ["metagenom", "16s", "microbiome", "taxonom", "kraken", 
                                       "metaphlan", "qiime", "mothur", "amplicon"]):
        matched_categories.append("Metagenomics")
    
    if any(kw in full_text for kw in ["atac-seq", "chip-seq", "methylat", "hi-c", "chromatin",
                                       "cut&tag", "bisulfite", "epigenom"]):
        matched_categories.append("Epigenetics")
    
    if any(kw in full_text for kw in ["rna-seq", "rnaseq", "transcript", "differential expression",
                                       "deseq", "edger", "kallisto", "salmon"]):
        matched_categories.append("Transcriptomics")
    
    if not matched_categories:
        return ["Genomics"]
    
    return matched_categories


# ============================================================
# 评分系统
# ============================================================

def calculate_pipeline_score(stars: int, weekly_growth: int, has_docker: bool, has_conda: bool, 
                           pushed_at: str = "", open_issues: int = 0, has_paper: bool = False) -> float:
    """Pipeline 评分公式"""
    star_component = 5 * math.log10(stars) if stars > 0 else 0
    growth_component = weekly_growth * 2
    env_bonus = 15 if (has_docker or has_conda) else 0
    paper_bonus = 5 if has_paper else 0
    
    base_score = star_component + growth_component + env_bonus + paper_bonus
    
    if pushed_at:
        try:
            pushed_dt = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
            days_since_update = (datetime.now(timezone.utc) - pushed_dt).days
            if days_since_update > 730:
                base_score *= 0.5
        except:
            pass
    
    return round(base_score, 2)


def calculate_utility_score(stars: int, weekly_growth: int, pushed_at: str = "", 
                          open_issues: int = 0, has_paper: bool = False) -> float:
    """Utility 评分公式"""
    star_component = 8 * math.log10(stars) if stars > 0 else 0
    growth_component = weekly_growth * 2
    paper_bonus = 5 if has_paper else 0
    
    base_score = star_component + growth_component + paper_bonus
    
    if pushed_at:
        try:
            pushed_dt = datetime.fromisoformat(pushed_at.replace('Z', '+00:00'))
            days_since_update = (datetime.now(timezone.utc) - pushed_dt).days
            if days_since_update > 730:
                base_score *= 0.5
        except:
            pass
    
    return round(base_score, 2)


# ============================================================
# 深度搜索
# ============================================================

def depth_search(quick_mode: bool = False):
    log("\n" + "=" * 70)
    log("[Deep Search] Dual-track collection...")
    log("Criteria: pushed:>2025-01-01 AND stars:>20")
    log("=" * 70)
    
    cutoff_date = "2025-01-01"
    total_found = 0
    seen_repos = set()
    
    keywords_to_search = KEYWORDS
    if quick_mode:
        keywords_to_search = {
            "Genomics": ["GATK", "BWA"],
            "Transcriptomics": ["RNA-seq", "DESeq2"],
            "Metagenomics": ["Metagenome"],
            "Single-cell": ["scRNA-seq", "Scanpy"]
        }
    
    category_pipelines_count = {cat: 0 for cat in KEYWORDS.keys()}
    max_pipelines_per_category = 20
    
    bio_terms = ["bioinformatics", "genomics", "transcriptomics", "metagenomics", 
                 "epigenetics", "sequencing", "alignment", "assembly", "variant", 
                 "expression", "analysis", "pipeline", "workflow"]
    
    for category, keywords in keywords_to_search.items():
        log(f"\n[Category] {category}")
        log("-" * 60)
        
        if category_pipelines_count[category] >= max_pipelines_per_category:
            log(f"  [Skip] {category} reached {category_pipelines_count[category]} Pipelines")
            continue
        
        for keyword in keywords:
            query = f"{keyword} pushed:>{cutoff_date} stars:>20"
            log(f"  Keyword: {keyword}")
            
            repos = search_with_pagination(query, min_results=50 if not quick_mode else 20)
            
            for repo in repos:
                full_name = repo.get("full_name")
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)
                
                if is_excluded(repo):
                    continue
                
                description = (repo.get("description") or "").lower()
                topics = [t.lower() for t in repo.get("topics", [])]
                
                has_bio_term = any(term in description or term in topics for term in bio_terms)
                if not has_bio_term:
                    continue
                
                owner = repo.get("owner", {}).get("login", "")
                repo_name = repo.get("name", "")
                readme = get_readme_content(owner, repo_name)
                
                final_category = classify_category(repo, category)
                project_type = detect_project_type(repo, readme)
                
                if project_type == "Pipeline" and category_pipelines_count[final_category] >= max_pipelines_per_category:
                    continue
                
                has_paper = detect_has_paper(readme)
                has_docker, has_conda = detect_environment_support(readme)
                sub_label = detect_sub_label(repo, readme) if project_type == "Utility" else ""
                
                # 数据增强
                install_commands = extract_install_commands(readme)
                preview_images = extract_preview_images(readme, repo.get("html_url", ""))
                badge_url = generate_badge_url(full_name, repo.get("stargazers_count", 0), repo.get("language", ""))
                
                save_repo_with_snapshot(repo, final_category, project_type, 
                                       has_paper, has_docker, has_conda, sub_label,
                                       install_commands, preview_images, badge_url)
                total_found += 1
                
                if project_type == "Pipeline":
                    category_pipelines_count[final_category] += 1
                
                stars = repo.get("stargazers_count", 0)
                type_mark = "[P]" if project_type == "Pipeline" else f"[U:{sub_label}]"
                env_mark = " [D]" if has_docker else ""
                env_mark += " [C]" if has_conda else ""
                log(f"    + {full_name} (*{stars}) {type_mark}{env_mark} -> {final_category}")
                
                time.sleep(0.3)
            
            time.sleep(3)
    
    log(f"\n[Search Complete] Total: {total_found} repositories")
    log(f"Pipeline distribution: {category_pipelines_count}")
    return total_found


# ============================================================
# 生成排行榜报告
# ============================================================

def generate_ranking_report():
    log("\n" + "=" * 70)
    log("[Rankings] Generating report...")
    log("=" * 70)
    
    repos = get_all_repos_for_ranking()
    
    # 计算评分
    for repo in repos:
        weekly_growth = get_weekly_star_growth(repo["id"])
        repo["weekly_growth"] = weekly_growth
        
        if repo["project_type"] == "Pipeline":
            repo["score"] = calculate_pipeline_score(
                repo["stars"], weekly_growth, 
                bool(repo["has_docker"]), bool(repo["has_conda_env"]),
                repo.get("pushed_at", ""), repo.get("open_issues", 0), bool(repo["has_paper"])
            )
        else:
            repo["score"] = calculate_utility_score(
                repo["stars"], weekly_growth,
                repo.get("pushed_at", ""), repo.get("open_issues", 0), bool(repo["has_paper"])
            )
    
    # 按类别和类型分组
    categories = ["Genomics", "Transcriptomics", "Metagenomics", "Single-cell", "Epigenetics"]
    ranking = {}
    
    for cat in categories:
        cat_repos = []
        for repo in repos:
            repo_categories = get_multi_categories(repo)
            if cat in repo_categories:
                cat_repos.append(repo)
        
        pipelines = sorted([r for r in cat_repos if r["project_type"] == "Pipeline"],
                          key=lambda x: x["score"], reverse=True)[:20]
        utilities = sorted([r for r in cat_repos if r["project_type"] == "Utility"],
                          key=lambda x: x["score"], reverse=True)[:10]
        
        ranking[cat] = {
            "total_count": len(cat_repos),
            "pipeline_count": len([r for r in cat_repos if r["project_type"] == "Pipeline"]),
            "utility_count": len([r for r in cat_repos if r["project_type"] == "Utility"]),
            "top_20_pipelines": [_format_repo(r, i+1) for i, r in enumerate(pipelines)],
            "top_10_utilities": [_format_repo(r, i+1) for i, r in enumerate(utilities)]
        }
        
        log(f"\n[{cat}]")
        log(f"  --- Top 20 Pipelines ---")
        for i, r in enumerate(pipelines[:5], 1):
            env = "[D]" if r["has_docker"] else ""
            env += "[C]" if r["has_conda_env"] else ""
            log(f"    {i:2}. {r['full_name']} (S={r['score']:.1f}, *{r['stars']}) {env}")
        
        log(f"  --- Top 10 Utilities ---")
        for i, r in enumerate(utilities[:5], 1):
            label = f"[{r['sub_label']}]" if r.get("sub_label") else ""
            log(f"    {i:2}. {r['full_name']} (S={r['score']:.1f}, *{r['stars']}) {label}")
    
    # 生成红黑榜
    red_black_lists = _generate_red_black_lists(repos)
    
    # 检测新进榜项目
    new_entries = _detect_new_entries(ranking)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "version": "13.0",
        "scoring_formulas": {
            "pipeline": "S = 5 * log10(Stars) + Weekly_Growth * 2 + Env_Bonus(15) + Zombie_Penalty(0.5)",
            "utility": "S = 8 * log10(Stars) + Weekly_Growth * 2 + Zombie_Penalty(0.5)"
        },
        "total_repositories": len(repos),
        "categories": ranking,
        "red_black_lists": red_black_lists,
        "new_entries": new_entries,
        "summary": {
            "total_pipelines": len([r for r in repos if r["project_type"] == "Pipeline"]),
            "total_utilities": len([r for r in repos if r["project_type"] == "Utility"]),
            "with_docker": len([r for r in repos if r["has_docker"]]),
            "with_conda": len([r for r in repos if r["has_conda_env"]]),
            "paper_linked": len([r for r in repos if r["has_paper"]])
        }
    }
    
    # 保存 JSON
    output_path = DATA_DIR / "ranking_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"\nJSON Report: {output_path}")
    
    # 保存历史记录用于新进榜检测
    _save_ranking_history(ranking)
    
    # 输出新进榜通知
    if new_entries:
        _print_new_entry_notifications(new_entries)
    
    return report


def _generate_red_black_lists(repos):
    """生成红黑榜"""
    growth_repos = [(r, r["weekly_growth"]) for r in repos if r["weekly_growth"] > 0]
    red_list = sorted(growth_repos, key=lambda x: x[1], reverse=True)[:5]
    
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    black_list_candidates = []
    for repo in repos:
        try:
            pushed_dt = datetime.fromisoformat(repo["pushed_at"].replace('Z', '+00:00'))
            if repo["stars"] > 0 and (repo["open_issues"] / repo["stars"]) > 0.15 and pushed_dt < six_months_ago:
                days_inactive = (datetime.now(timezone.utc) - pushed_dt).days
                black_list_candidates.append((repo, repo["open_issues"], days_inactive, repo["stars"]))
        except:
            continue
    
    black_list = sorted(black_list_candidates, key=lambda x: (-(x[1]/x[3]), -x[2]))[:5]
    
    return {
        "fastest_growth": [{"repo": r[0], "growth": r[1]} for r in red_list],
        "maintenance_warning": [{"repo": r[0], "issues": r[1], "inactive_days": r[2], "stars": r[3]} for r in black_list]
    }


def _format_repo(r: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "name": r["full_name"],
        "url": r["url"],
        "description": r["description"],
        "stars": r["stars"],
        "weekly_growth": r["weekly_growth"],
        "score": r["score"],
        "has_paper": bool(r["has_paper"]),
        "has_docker": bool(r["has_docker"]),
        "has_conda": bool(r["has_conda_env"]),
        "sub_label": r.get("sub_label", ""),
        "topics": json.loads(r["topics"]) if r["topics"] else [],
        "license": r.get("license", "Unknown"),
        "open_issues": r.get("open_issues", 0),
        "pushed_at": r.get("pushed_at", ""),
        "tech_stack": json.loads(r["tech_stack"]) if r.get("tech_stack") else [],
        "install_commands": json.loads(r["install_commands"]) if r.get("install_commands") else [],
        "preview_images": json.loads(r["preview_images"]) if r.get("preview_images") else [],
        "badges": generate_all_badges(r["full_name"], r["stars"], r.get("language", ""),
                                      bool(r["has_docker"]), bool(r["has_conda_env"]))
    }


def _save_ranking_history(ranking: dict):
    """保存排名历史用于新进榜检测"""
    history_path = DATA_DIR / "ranking_history.json"
    
    current_top3 = {}
    for cat, data in ranking.items():
        current_top3[cat] = {
            "pipelines": [p["name"] for p in data["top_20_pipelines"][:3]],
            "utilities": [u["name"] for u in data["top_10_utilities"][:3]]
        }
    
    history = {"timestamp": datetime.now().isoformat(), "top3": current_top3}
    
    # 加载历史并追加
    all_history = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                all_history = json.load(f)
        except:
            all_history = []
    
    all_history.append(history)
    
    # 只保留最近10次记录
    all_history = all_history[-10:]
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)


def _detect_new_entries(ranking: dict) -> List[Dict]:
    """检测新进榜项目"""
    history_path = DATA_DIR / "ranking_history.json"
    
    if not history_path.exists():
        return []
    
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_history = json.load(f)
    except:
        return []
    
    if len(all_history) < 1:
        return []
    
    # 获取上次的Top3
    last_record = all_history[-1]
    last_top3 = last_record.get("top3", {})
    
    new_entries = []
    
    for cat, data in ranking.items():
        current_pipelines = [p["name"] for p in data["top_20_pipelines"][:3]]
        current_utilities = [u["name"] for u in data["top_10_utilities"][:3]]
        
        last_pipelines = last_top3.get(cat, {}).get("pipelines", [])
        last_utilities = last_top3.get(cat, {}).get("utilities", [])
        
        # 检测新进榜的Pipeline
        for name in current_pipelines:
            if name not in last_pipelines:
                repo_data = next((p for p in data["top_20_pipelines"] if p["name"] == name), None)
                if repo_data:
                    new_entries.append({
                        "category": cat,
                        "type": "Pipeline",
                        "repo": repo_data
                    })
        
        # 检测新进榜的Utility
        for name in current_utilities:
            if name not in last_utilities:
                repo_data = next((u for u in data["top_10_utilities"] if u["name"] == name), None)
                if repo_data:
                    new_entries.append({
                        "category": cat,
                        "type": "Utility",
                        "repo": repo_data
                    })
    
    return new_entries


def _print_new_entry_notifications(new_entries: List[Dict]):
    """输出新进榜通知"""
    log("\n" + "=" * 70)
    log("[NEW ENTRIES] The following projects entered Top 3 this week:")
    log("=" * 70)
    
    for entry in new_entries:
        repo = entry["repo"]
        log(f"\n  Category: {entry['category']}")
        log(f"  Type: {entry['type']}")
        log(f"  Name: {repo['name']}")
        log(f"  Stars: {repo['stars']} | Score: {repo['score']:.1f}")
        log(f"  URL: {repo['url']}")
        
        # 生成推荐发送的Issue勋章代码
        log(f"\n  [Recommended Badge Markdown]:")
        log(f"  Congratulations! Your project **{repo['name']}** has entered the Bio-Rank Gateway Top 3!")
        log(f"  ![Bio-Rank Badge](https://img.shields.io/badge/Bio--Rank-Top%203%20{entry['category']}-brightgreen)")
        log("-" * 50)


# ============================================================
# 主函数
# ============================================================

def main():
    log("=" * 70)
    log("Bio-Rank Gateway v13.0")
    log("=" * 70)
    
    init_database()
    
    # 执行深度搜索
    depth_search(quick_mode=False)
    
    # 生成排行榜
    generate_ranking_report()
    
    log("\n[Complete] Bio-Rank Gateway update finished.")


if __name__ == "__main__":
    main()
