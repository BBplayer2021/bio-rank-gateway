#!/usr/bin/env python3
"""
Bio-Rank Gateway Weekly Report Generator
生成周报长图海报 (9:16 比例，适合手机端分享)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Installing...")
    os.system(f"{sys.executable} -m pip install Pillow")
    from PIL import Image, ImageDraw, ImageFont

try:
    import qrcode
except ImportError:
    print("qrcode not installed. Installing...")
    os.system(f"{sys.executable} -m pip install qrcode[pil]")
    import qrcode

# ============== 配置 ==============
POSTER_WIDTH = 1080
POSTER_HEIGHT = 1920  # 9:16 比例
WEBSITE_URL = "https://bbplayer2021.github.io/bio-rank-gateway/"

# 颜色主题
COLORS = {
    "bg_gradient_top": "#1a202c",
    "bg_gradient_bottom": "#2d3748",
    "primary": "#38a169",
    "secondary": "#4299e1",
    "accent": "#ed8936",
    "text_white": "#ffffff",
    "text_muted": "#a0aec0",
    "card_bg": "#2d3748",
    "gold": "#ffd700",
    "silver": "#c0c0c0",
    "bronze": "#cd7f32",
}

# 分类图标映射 (使用emoji或文字作为备用)
CATEGORY_ICONS = {
    "Genomics": "DNA",
    "Transcriptomics": "RNA",
    "Metagenomics": "MET",
    "Single-cell": "SC",
    "Epigenetics": "EPI",
    "Proteomics": "PRO",
    "Metabolomics": "MTB",
}


def hex_to_rgb(hex_color: str) -> tuple:
    """将十六进制颜色转为RGB元组"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_gradient_background(width: int, height: int) -> Image.Image:
    """创建渐变背景"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    top_color = hex_to_rgb(COLORS["bg_gradient_top"])
    bottom_color = hex_to_rgb(COLORS["bg_gradient_bottom"])
    
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    return img


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """获取字体，优先使用系统字体"""
    font_paths = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    
    if bold:
        font_paths = [
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ] + font_paths
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    
    # 回退到默认字体
    return ImageFont.load_default()


def fetch_logo(url: str, size: tuple = (80, 80)) -> Image.Image | None:
    """从URL获取Logo图片"""
    if not url:
        return None
    try:
        headers = {'User-Agent': 'Bio-Rank-Gateway/1.0'}
        req = Request(url, headers=headers)
        with urlopen(req, timeout=5) as response:
            img_data = response.read()
        img = Image.open(BytesIO(img_data))
        img = img.convert('RGBA')
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Failed to fetch logo from {url}: {e}")
        return None


def create_initials_avatar(name: str, size: int = 80) -> Image.Image:
    """创建首字母头像"""
    colors = ['#4299e1', '#48bb78', '#ed8936', '#9f7aea', '#ed64a6', '#38b2ac']
    color_index = sum(ord(c) for c in name) % len(colors)
    bg_color = hex_to_rgb(colors[color_index])
    
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制圆形背景
    draw.ellipse([0, 0, size-1, size-1], fill=bg_color)
    
    # 绘制文字
    initials = name[:2].upper()
    font = get_font(size // 3, bold=True)
    
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - bbox[1]
    
    draw.text((x, y), initials, fill=(255, 255, 255), font=font)
    
    return img


def draw_rounded_rect(draw: ImageDraw.Draw, xy: tuple, radius: int, fill: tuple):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], fill=fill)
    draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], fill=fill)
    draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], fill=fill)
    draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], fill=fill)


def generate_qr_code(url: str, size: int = 200) -> Image.Image:
    """生成二维码"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="white", back_color="transparent")
    qr_img = qr_img.convert('RGBA')
    qr_img = qr_img.resize((size, size), Image.Resampling.LANCZOS)
    
    return qr_img


def load_ranking_data(json_path: str) -> dict:
    """加载排名数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_top_growth(data: dict, top_n: int = 5) -> list:
    """提取本周Star增长最快的Top N工具"""
    all_projects = []
    
    for category, cat_data in data.get("categories", {}).items():
        for track in ["top_20_pipelines", "top_10_utilities"]:
            projects = cat_data.get(track, [])
            for proj in projects:
                proj["_category"] = category
                proj["_track"] = "Pipeline" if "pipeline" in track else "Utility"
                all_projects.append(proj)
    
    # 按weekly_growth降序排序，去重
    seen = set()
    unique_projects = []
    for proj in sorted(all_projects, key=lambda x: x.get("weekly_growth", 0), reverse=True):
        name = proj.get("full_name", "")
        if name not in seen:
            seen.add(name)
            unique_projects.append(proj)
    
    return unique_projects[:top_n]


def extract_category_champions(data: dict) -> dict:
    """提取各分类冠亚军"""
    champions = {}
    
    for category, cat_data in data.get("categories", {}).items():
        pipelines = cat_data.get("top_20_pipelines", [])[:2]
        utilities = cat_data.get("top_10_utilities", [])[:2]
        
        champions[category] = {
            "pipelines": pipelines,
            "utilities": utilities,
        }
    
    return champions


def generate_weekly_report(json_path: str, output_path: str):
    """生成周报海报"""
    print(f"Loading data from {json_path}...")
    data = load_ranking_data(json_path)
    
    # 提取数据
    top_growth = extract_top_growth(data, top_n=5)
    champions = extract_category_champions(data)
    generated_at = data.get("generated_at", datetime.now().isoformat())
    report_date = datetime.fromisoformat(generated_at).strftime("%Y-%m-%d")
    
    print(f"Creating poster for week of {report_date}...")
    
    # 创建画布
    poster = create_gradient_background(POSTER_WIDTH, POSTER_HEIGHT)
    draw = ImageDraw.Draw(poster)
    
    # 字体
    font_title = get_font(56, bold=True)
    font_subtitle = get_font(28)
    font_section = get_font(36, bold=True)
    font_item = get_font(26, bold=True)
    font_desc = get_font(20)
    font_small = get_font(18)
    font_badge = get_font(16, bold=True)
    
    y_offset = 60
    padding = 50
    
    # ========== Header ==========
    # Logo/品牌
    draw.text((padding, y_offset), "Bio-Rank", fill=hex_to_rgb(COLORS["primary"]), font=font_title)
    draw.text((padding + 260, y_offset + 10), "Gateway", fill=hex_to_rgb(COLORS["text_white"]), font=font_section)
    y_offset += 70
    
    # 标语
    draw.text((padding, y_offset), "Find the best, code the rest.", 
              fill=hex_to_rgb(COLORS["text_muted"]), font=font_subtitle)
    y_offset += 50
    
    # 周报日期
    draw.text((padding, y_offset), f"Weekly Report  {report_date}", 
              fill=hex_to_rgb(COLORS["secondary"]), font=font_subtitle)
    y_offset += 80
    
    # ========== 本周新秀 Top 5 ==========
    # Section 标题
    draw.text((padding, y_offset), "This Week's Rising Stars", 
              fill=hex_to_rgb(COLORS["gold"]), font=font_section)
    y_offset += 55
    
    # Top 5 卡片
    card_height = 100
    for i, proj in enumerate(top_growth):
        # 卡片背景
        card_y = y_offset + i * (card_height + 15)
        draw_rounded_rect(draw, (padding, card_y, POSTER_WIDTH - padding, card_y + card_height), 
                         12, hex_to_rgb(COLORS["card_bg"]))
        
        # 排名徽章
        rank_colors = [COLORS["gold"], COLORS["silver"], COLORS["bronze"], COLORS["primary"], COLORS["primary"]]
        rank_color = hex_to_rgb(rank_colors[i])
        draw.ellipse([padding + 15, card_y + 25, padding + 65, card_y + 75], fill=rank_color)
        rank_text = str(i + 1)
        rank_bbox = draw.textbbox((0, 0), rank_text, font=font_item)
        rank_x = padding + 40 - (rank_bbox[2] - rank_bbox[0]) // 2
        rank_y = card_y + 50 - (rank_bbox[3] - rank_bbox[1]) // 2 - rank_bbox[1]
        draw.text((rank_x, rank_y), rank_text, fill=(0, 0, 0) if i < 3 else (255, 255, 255), font=font_item)
        
        # Logo 或首字母头像
        short_name = proj.get("short_name", proj.get("name", "").split("/")[-1])
        logo_url = proj.get("preview_images", [None])[0] if proj.get("preview_images") else None
        logo = fetch_logo(logo_url, (60, 60)) if logo_url else None
        if logo is None:
            logo = create_initials_avatar(short_name, 60)
        
        logo_x = padding + 85
        logo_y = card_y + 20
        poster.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)
        
        # 项目名称
        name_x = padding + 160
        draw.text((name_x, card_y + 20), short_name[:20], 
                  fill=hex_to_rgb(COLORS["text_white"]), font=font_item)
        
        # 分类标签
        category = proj.get("_category", "")
        cat_bbox = draw.textbbox((0, 0), category, font=font_small)
        cat_width = cat_bbox[2] - cat_bbox[0] + 16
        cat_x = POSTER_WIDTH - padding - cat_width - 10
        draw_rounded_rect(draw, (cat_x, card_y + 20, cat_x + cat_width, card_y + 42), 
                         8, hex_to_rgb(COLORS["secondary"]))
        draw.text((cat_x + 8, card_y + 22), category, 
                  fill=hex_to_rgb(COLORS["text_white"]), font=font_small)
        
        # Star增长
        growth = proj.get("weekly_growth", 0)
        growth_text = f"+{growth} stars/week"
        draw.text((name_x, card_y + 55), growth_text, 
                  fill=hex_to_rgb(COLORS["accent"]), font=font_desc)
        
        # 总Star数
        stars = proj.get("stars", 0)
        stars_text = f"{stars:,} total"
        draw.text((name_x + 180, card_y + 55), stars_text, 
                  fill=hex_to_rgb(COLORS["text_muted"]), font=font_desc)
    
    y_offset += 5 * (card_height + 15) + 40
    
    # ========== 分类冠军 ==========
    draw.text((padding, y_offset), "Category Champions", 
              fill=hex_to_rgb(COLORS["secondary"]), font=font_section)
    y_offset += 55
    
    # 只展示主要分类的冠军 (前4个)
    main_categories = ["Genomics", "Transcriptomics", "Single-cell", "Metagenomics"]
    
    col_width = (POSTER_WIDTH - padding * 2 - 20) // 2
    col_height = 130
    
    for idx, category in enumerate(main_categories):
        if category not in champions:
            continue
        
        col = idx % 2
        row = idx // 2
        
        card_x = padding + col * (col_width + 20)
        card_y = y_offset + row * (col_height + 15)
        
        # 卡片背景
        draw_rounded_rect(draw, (card_x, card_y, card_x + col_width, card_y + col_height), 
                         12, hex_to_rgb(COLORS["card_bg"]))
        
        # 分类名称
        cat_icon = CATEGORY_ICONS.get(category, "?")
        draw.text((card_x + 15, card_y + 12), f"[{cat_icon}] {category}", 
                  fill=hex_to_rgb(COLORS["primary"]), font=font_small)
        
        # Pipeline冠军
        pipelines = champions[category].get("pipelines", [])
        if pipelines:
            top_pipe = pipelines[0]
            pipe_name = top_pipe.get("short_name", "")[:18]
            draw.text((card_x + 15, card_y + 45), f"Pipeline: {pipe_name}", 
                      fill=hex_to_rgb(COLORS["text_white"]), font=font_desc)
        
        # Utility冠军
        utilities = champions[category].get("utilities", [])
        if utilities:
            top_util = utilities[0]
            util_name = top_util.get("short_name", "")[:18]
            draw.text((card_x + 15, card_y + 80), f"Utility: {util_name}", 
                      fill=hex_to_rgb(COLORS["text_muted"]), font=font_desc)
    
    y_offset += 2 * (col_height + 15) + 40
    
    # ========== Footer: 二维码 + 统计 ==========
    # 分隔线
    draw.line([(padding, y_offset), (POSTER_WIDTH - padding, y_offset)], 
              fill=hex_to_rgb(COLORS["text_muted"]), width=1)
    y_offset += 30
    
    # 统计数据
    total_repos = data.get("total_repositories", 0)
    summary = data.get("summary", {})
    total_pipelines = summary.get("total_pipelines", 0)
    total_utilities = summary.get("total_utilities", 0)
    
    stats_text = f"Total: {total_repos} repos  |  {total_pipelines} pipelines  |  {total_utilities} utilities"
    draw.text((padding, y_offset), stats_text, 
              fill=hex_to_rgb(COLORS["text_muted"]), font=font_small)
    y_offset += 40
    
    # 二维码
    qr_size = 180
    qr_img = generate_qr_code(WEBSITE_URL, qr_size)
    qr_x = (POSTER_WIDTH - qr_size) // 2
    qr_y = y_offset
    
    # 二维码白色背景
    draw_rounded_rect(draw, (qr_x - 10, qr_y - 10, qr_x + qr_size + 10, qr_y + qr_size + 10), 
                     12, (255, 255, 255))
    poster.paste(qr_img, (qr_x, qr_y), qr_img)
    
    y_offset += qr_size + 25
    
    # 扫码提示
    scan_text = "Scan to explore full rankings"
    scan_bbox = draw.textbbox((0, 0), scan_text, font=font_desc)
    scan_x = (POSTER_WIDTH - (scan_bbox[2] - scan_bbox[0])) // 2
    draw.text((scan_x, y_offset), scan_text, 
              fill=hex_to_rgb(COLORS["text_white"]), font=font_desc)
    
    y_offset += 50
    
    # 版权
    copyright_text = f"Bio-Rank Gateway v{data.get('version', '13.0')} | Generated automatically"
    copy_bbox = draw.textbbox((0, 0), copyright_text, font=font_small)
    copy_x = (POSTER_WIDTH - (copy_bbox[2] - copy_bbox[0])) // 2
    draw.text((copy_x, y_offset), copyright_text, 
              fill=hex_to_rgb(COLORS["text_muted"]), font=font_small)
    
    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    poster.save(output_path, "PNG", quality=95)
    print(f"Weekly report saved to: {output_path}")
    
    return output_path


def main():
    # 路径配置
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    json_path = project_root / "data" / "ranking_report.json"
    
    # 生成带日期的文件名
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = project_root / "docs" / "reports" / f"weekly_report_{today}.png"
    
    # 同时生成一个latest版本
    latest_path = project_root / "docs" / "reports" / "weekly_report_latest.png"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        sys.exit(1)
    
    # 生成报告
    generate_weekly_report(str(json_path), str(output_path))
    
    # 复制为latest
    import shutil
    shutil.copy(output_path, latest_path)
    print(f"Latest report copied to: {latest_path}")


if __name__ == "__main__":
    main()
