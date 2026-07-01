"""
智能 PDF 元数据提取器

相比简单取首行的做法，增加了：
- 期刊卷期信息识别与跳过
- 论文标题智能定位
- 作者/机构/摘要/关键词提取
- 发表年份提取
"""

import re
from typing import Optional


# ─── 常见的期刊卷期行模式 ───
JOURNAL_HEADER_PATTERNS = [
    re.compile(r"第\s*\d+\s*卷\s*第\s*\d+\s*期"),       # 第 25 卷 第 1 期
    re.compile(r"第\s*\d+\s*期"),                       # 第2 期
    re.compile(r"Vol\.\s*\d+\s*No\.\s*\d+", re.I),     # Vol. 25 No. 1
    re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月"),          # 2026年4月
    re.compile(r"[A-Z][a-z]+\.?\s*\d{4}"),              # Apr. 2026
    re.compile(r"【文章编号】"),                         # 【文章编号】1002-6274
    re.compile(r"ISSN\s*[\d\-]+", re.I),
    re.compile(r"收稿日期"),
    re.compile(r"基金项目"),
    re.compile(r"作者简介"),
    re.compile(r"中图分类号"),
    re.compile(r"文献标识码"),
    re.compile(r"DOI"),
    re.compile(r"^\s*\d+\s*$"),                         # 纯页码
    re.compile(r"^\s*[■□▲△●○◆◇★☆▶▷]+\s*$"),          # 装饰符号行
]


def _is_meta_line(line: str) -> bool:
    """判断是否为元数据行（卷期/编号/摘要标记等），应该跳过"""
    line = line.strip()
    if len(line) < 5:
        return True
    for pat in JOURNAL_HEADER_PATTERNS:
        if pat.search(line):
            return True
    # 摘要/关键词标记行
    if re.match(r"^(摘\s*要[：:]|【内容摘要】|【摘\s*要】|关键词|关键\s*词)", line):
        return True
    return False


def _has_chinese(text: str) -> bool:
    """是否包含中文"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def extract_title(text: str, filename: str = "") -> str:
    """
    从文本中智能提取标题。

    策略：
    1. 文件名如果包含章节标记（第X章/序章等），直接使用文件名
    2. 文件名如果包含中文且足够长（>=4字），优先使用文件名
    3. 扫描文本前 30 行，跳过元数据行，找标题特征行
    4. 回退到文件名
    """
    # 优先：文件名通常就是最准确的标题
    fn_title = _filename_title(filename)

    # 章节标记文件名：直接使用，不管长度
    if fn_title and re.match(r"^(第[一二三四五六七八九十百千\d]+[章节回]|楔子|序章|序言|尾声|番外|后记|Chapter)", fn_title, re.I):
        return fn_title[:200]

    # 中文文件名 >= 4 字：优先使用
    if _has_chinese(fn_title) and len(fn_title) >= 4:
        return fn_title[:200]

    # 回退：从文本提取
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return fn_title or _strip_extension(filename)

    # 先检查第一行是否像标题（短行+中文+非元数据）
    first_line = lines[0]
    if _has_chinese(first_line) and not _is_meta_line(first_line) and len(first_line) <= 80:
        # 如果第一行像章节标题，直接用
        if re.match(r"^(第[一二三四五六七八九十百千\d]+[章节回]|楔子|序章|序言|尾声|番外|后记)", first_line):
            return first_line[:200]
        # 如果文件名有效，优先文件名
        if fn_title and _has_chinese(fn_title):
            return fn_title[:200]
        return first_line[:200]

    candidates = []
    for i, line in enumerate(lines[:30]):
        line = line.strip()
        if _is_meta_line(line):
            continue
        line_len = len(line)

        if _has_chinese(line):
            if 10 <= line_len <= 120:
                bonus = 1.0
                if any(c in line for c in ["：", ":", "——", "？"]):
                    bonus = 2.0
                if re.search(r"[（(]\s*[A-Za-z0-9]+\s*[）)]", line):
                    bonus = 0.5
                candidates.append((line, i, bonus * line_len))

    if candidates:
        candidates.sort(key=lambda x: (-x[2], x[1]))
        best = candidates[0][0][:200]
        if re.match(r"^(摘\s*要[：:]|【内容摘要】)", best):
            return fn_title or _filename_title(filename)
        # 如果有文件名，优先文件名（比从正文猜更准确）
        if fn_title and _has_chinese(fn_title) and len(fn_title) >= 3:
            return fn_title[:200]
        return best

    return fn_title or _filename_title(filename)


# 所有支持的文件扩展名（用于从文件名中清除后缀）
ALL_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".markdown", ".docx", ".epub",
    ".html", ".htm", ".pptx",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    ".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz",
}


def _strip_extension(filename: str) -> str:
    """去除文件名的扩展名（支持双扩展名如 .tar.gz）"""
    name = filename.strip()
    # 先检查双扩展名
    for ext in [".tar.gz", ".tar.bz2", ".tar.xz"]:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    # 单扩展名
    dot_idx = name.rfind(".")
    if dot_idx > 0 and name[dot_idx:].lower() in ALL_EXTENSIONS:
        return name[:dot_idx]
    return name


def _filename_title(filename: str) -> str:
    """从文件名提取标题"""
    if not filename:
        return ""
    name = _strip_extension(filename)
    # 去掉编号前缀（如 "01_" "12." "3-" 等）
    name = re.sub(r"^[\d\s\.\-_]+", "", name)
    # 章节标记检测：如果文件名以"第X章/序章/楔子"开头，不做作者后缀剥离
    is_chapter_file = bool(re.match(r"^(第[一二三四五六七八九十百千\d]+[章节回]|楔子|序章|序言|尾声|番外|后记|Chapter)", name, re.I))
    if not is_chapter_file:
        # 去掉文件名中的作者后缀（如 "标题_作者名"）
        name = re.sub(r"_[\u4e00-\u9fff\w]+$", "", name)
    # 去掉常见的前缀分隔符
    name = name.lstrip("_-— ")
    # 将下划线替换为空格（更好的可读性）
    name = name.replace("_", " ")
    return name[:200] if name else ""


def extract_author(text: str, filename: str = "") -> str:
    """
    智能提取作者。

    策略：
    1. 搜索 "作者简介：XXX" 或 "XXX¹" 格式
    2. 搜索独立的作者行（通常 2-4 个汉字）
    3. 回退到文件名中的作者
    """
    # 策略 1：作者简介
    m = re.search(r"作者简介[：:]\s*(\w+)", text)
    if m:
        return m.group(1)

    # 策略 2：带有上标编号的作者行（如 "张三¹，李四²"）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:15]:
        line = line.strip()
        if re.match(r"^[\u4e00-\u9fff\w]+[\d¹²³⁴⁵⁶⁷⁸⁹,，]+$", line) and 3 <= len(line) <= 80:
            # 提取第一个中文名
            m = re.search(r"([\u4e00-\u9fff]{2,4})", line)
            if m:
                return m.group(1)

    # 策略 3：文件名中的作者
    if filename:
        clean_fn = _strip_extension(filename)
        # 如果文件名是章节格式，不做作者提取（避免把章节名当作者）
        is_chapter_fn = bool(re.match(r"^(第[一二三四五六七八九十百千\d]+[章节回]|楔子|序章|序言|尾声|番外|后记|Chapter)", clean_fn, re.I))
        if not is_chapter_fn:
            m = re.search(r"_([\u4e00-\u9fff]+)$", clean_fn)
            if m:
                return m.group(1)
            # 英文名
            m = re.search(r"_(\w+)$", clean_fn)
            if m:
                return m.group(1)

    return "未知"


def extract_summary(text: str) -> str:
    """
    提取摘要。匹配"摘要"到"关键词"之间的文本。
    """
    patterns = [
        r"摘\s*要[：:]\s*(.+?)(?:\n\s*(?:关键词|关键\s*词))",
        r"摘\s*要\s*\n(.+?)(?:\n\s*(?:关键词|关键\s*词))",
        r"摘\s*要[：:]\s*(.+?)(?:\n\s*(?:一、|二、|引言|前言|正\s*文))",
    ]
    for pat in patterns:
        m = re.search(pat, text[:5000], re.DOTALL)
        if m:
            s = m.group(1).strip()
            s = re.sub(r"\s+", " ", s)
            if len(s) > 20:
                # 清理卷期噪音（摘要段落内可能混入的元数据）
                s = re.sub(r"第\s*\d+\s*卷\s*第\s*\d+\s*期", "", s)
                s = re.sub(r"\d{4}\s*年\s*\d{1,2}\s*月", "", s)
                s = re.sub(r"【文章编号】[\d\-\w]+", "", s)
                s = re.sub(r"No\.\s*\d+", "", s, flags=re.I)
                s = re.sub(r"\s{2,}", " ", s)
                return s.strip()[:500]

    # 回退
    clean = text[:500].strip().replace("\n", " ")
    return clean[:300]


def extract_keywords(text: str) -> list[str]:
    """提取关键词"""
    patterns = [
        r"关键词[：:]\s*(.+?)(?:\n|$)",
        r"关键\s*词[：:]\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text[:5000])
        if m:
            kw = m.group(1).strip()
            # 按常见分隔符拆分
            kws = re.split(r"[；;，,\s]+", kw)
            return [k.strip() for k in kws if k.strip()]
    return []


def extract_journal(text: str) -> Optional[str]:
    """提取期刊名称"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    journal_patterns = [
        re.compile(r"^(.+)杂志$"),
        re.compile(r"^(.+)学报$"),
        re.compile(r"^(.+)论丛$"),
        re.compile(r"^(.+)研究$"),
        re.compile(r"^(.+)评论$"),
    ]
    for line in lines[:10]:
        for pat in journal_patterns:
            m = pat.match(line)
            if m:
                return m.group(0)
    return None


def extract_year(text: str) -> Optional[int]:
    """提取发表年份"""
    patterns = [
        r"(\d{4})\s*年",
        r"20\d{2}",
    ]
    for pat in patterns:
        m = re.search(pat, text[:2000])
        if m:
            year = int(m.group(1)) if m.lastindex else int(m.group(0))
            if 1990 <= year <= 2030:
                return year
    return None


def auto_tag_enhanced(
    text: str,
    title: str,
    keywords: list[str],
    db_tag_names: set[str],
    domain: str = "",
) -> list[str]:
    """
    增强版自动标签：综合标题 + 摘要 + 关键词匹配。

    返回数据库中存在的标签名列表。
    """
    # 法学关键词映射
    LAW_TAG_MAP = {
        "行政法学": ["行政监管", "行政许可", "行政处罚", "行政复议", "行政诉讼"],
        "经济法": ["金融市场", "货币政策", "财政", "反垄断", "税法", "数字货币", "数字人民币"],
        "国际法学": ["国际法", "跨境", "主权", "域外管辖", "条约", "WTO"],
        "知识产权法": ["知识产权", "著作权", "专利权", "商标权", "创造性"],
        "民法学": ["民事", "合同", "侵权", "物权", "人格权"],
        "宪法学": ["宪法", "基本权利", "合宪", "违宪", "立法法"],
        "刑法学": ["刑事", "犯罪", "刑罚", "罪名"],
        "数据法学": ["数据法学", "数据权利", "数据立法"],
        "法理学": ["法理学", "法哲学", "法治理论", "规范分析"],
        "数据跨境": ["数据跨境", "跨境流动", "跨境传输", "跨境流通", "跨境数据"],
        "数据治理": ["数据治理", "数据流通", "数据要素", "数据管理", "数据共享"],
        "个人信息保护": ["个人信息保护", "隐私权", "删除权", "被遗忘权"],
        "数据安全": ["数据安全", "网络安全", "数据泄露"],
        "数据确权": ["数据确权", "数据产权", "数据权属", "创造性劳动"],
        "AI法律": ["人工智能", "算法治理", "自动化决策", "机器学习"],
        "数字货币": ["数字货币", "数字人民币", "虚拟货币", "加密货币"],
    }

    # 写作域关键词映射
    WRITING_TAG_MAP = {
        # 体裁
        "百合": ["百合", "双女主", "女性之间的", "她爱", "她们之间"],
        "悬疑": ["悬疑", "谜团", "谜题", "线索", "真相", "诡计", "不可能犯罪"],
        "推理": ["推理", "逻辑", "演绎", "排除法", "证据链", "密室", "不在场证明"],
        "言情": ["言情", "心动", "暗恋", "表白", "暧昧", "喜欢"],
        # 结构
        "大纲": ["大纲", "主线", "支线", "剧情走向", "故事梗概"],
        "正文": ["正文", "第.*章", "Chapter"],
        "番外": ["番外", "后日谈", "特别篇"],
        # 元类型
        "人物": ["人物", "角色", "主角", "配角", "反派", "性格", "外貌", "人物设定"],
        "场景": ["场景", "courtroom", "法庭", "办公室", "雨夜", "街道", "咖啡", "酒吧"],
        "情节": ["情节", "主线", "支线", "伏笔", "反转", "悬念", "冲突", "高潮"],
        "设定": ["设定", "世界观", "魔法体系", "历史背景", "规则", "体系"],
        "情绪": ["情绪", "心理", "内心", "压抑", "爆发", "温情", "紧张", "悲伤"],
        # 角色
        "主角": ["主角", "女主角", "男主角", "protagonist"],
        "配角": ["配角", "次要角色", "朋友", "同事"],
        "反派": ["反派", "敌人", "对手", "antagonist"],
        # 情绪基调
        "压抑": ["压抑", "沉重", "窒息", "憋闷", "喘不过气"],
        "爆发": ["爆发", "怒吼", "摔", "砸", "崩溃", "失控"],
        "温情": ["温情", "温暖", "柔软", "微笑", "温柔", "心疼"],
        "紧张": ["紧张", "心跳", "冷汗", "屏住呼吸", "握紧"],
        "悲伤": ["悲伤", "哭", "泪", "痛", "心碎", "哽咽"],
        # 场景类型
        "法庭": ["法庭", "庭审", "审判席", "旁听席", "法官", "律师"],
        "办公室": ["办公室", "工位", "会议室", "走廊"],
        "雨夜": ["雨夜", "暴雨", "细雨", "雨水", "雨打"],
        "家": ["家", "客厅", "卧室", "厨房", "沙发", "床上"],
    }

    combined = title + " " + text[:3000] + " " + " ".join(keywords)
    matched = []

    # 根据领域选择标签映射（或两者都用）
    tag_maps = []
    if domain == "law":
        tag_maps.append(LAW_TAG_MAP)
    elif domain == "writing":
        tag_maps.append(WRITING_TAG_MAP)
    else:
        # 未知领域，两个都试
        tag_maps.append(LAW_TAG_MAP)
        tag_maps.append(WRITING_TAG_MAP)

    for tag_map in tag_maps:
        for tag_name, tag_keywords in tag_map.items():
            if tag_name not in db_tag_names:
                continue
            for kw in tag_keywords:
                if kw in combined:
                    matched.append(tag_name)
                    break

    return matched
