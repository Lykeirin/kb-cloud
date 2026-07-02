#!/usr/bin/env python3
"""
知识库 → Obsidian Vault 导出脚本

从 KB API 拉取所有文档，生成 Obsidian Vault：
- 每篇文档一个 .md 文件（含 YAML 元数据 + 全文）
- 基于语义相似度的 [[wikilinks]]
- 自动生成 MOC（Map of Content）索引文件
- 同时生成 Graphviz .dot 文件（供服务端渲染）

用法：
  python export_kb_to_obsidian.py                    # 输出到 ~/kb-obsidian-vault/
  python export_kb_to_obsidian.py --output ~/MyVault # 自定义路径
  python export_kb_to_obsidian.py --api http://localhost:8765  # 自定义 API
  python export_kb_to_obsidian.py --min-sim 0.6      # 相似度阈值
"""

import argparse
import json
import os
import sys
import re
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# ─── 默认配置 ───────────────────────────────────────────────
DEFAULT_API = "http://localhost:8765"
DEFAULT_OUTPUT = os.path.expanduser("~/kb-obsidian-vault")
DEFAULT_MIN_SIM = 0.5

# 域名 → 中文标签映射
DOMAIN_LABELS = {
    "law": "法学",
    "writing": "创作",
}

# ─── 工具函数 ───────────────────────────────────────────────

def safe_filename(title: str, doc_id: str) -> str:
    """生成安全的文件名（不含非法字符）"""
    name = title.strip()
    # 替换非法字符
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    # 截断过长标题
    if len(name) > 60:
        name = name[:57] + "..."
    # 追加 ID 后缀避免冲突
    short_id = doc_id[:8] if doc_id else "unknown"
    return f"{name}_{short_id}.md"


def sanitize_yaml_value(v: str) -> str:
    """清理 YAML 字符串值，转义引号"""
    if not v:
        return ""
    v = v.replace('"', '\\"')
    # 截断过长的摘要
    if len(v) > 300:
        v = v[:297] + "..."
    return v


def fetch_json(url: str) -> dict:
    """从 API 获取 JSON 数据"""
    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  ✗ API 请求失败: {url} — {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON 解析失败: {url} — {e}", file=sys.stderr)
        raise


# ─── 核心导出逻辑 ───────────────────────────────────────────

def export_obsidian(api_url: str, output_dir: str, min_sim: float):
    """主导出函数"""
    vault = Path(output_dir)
    vault.mkdir(parents=True, exist_ok=True)

    print(f"知识库 → Obsidian 导出")
    print(f"  API:    {api_url}")
    print(f"  输出:   {vault}")
    print(f"  阈值:   {min_sim}")
    print()

    # ── Step 1: 获取图谱数据（节点 + 边） ──
    print("[1/5] 获取图谱数据...")
    graph_url = f"{api_url}/api/graph?min_similarity={min_sim}"
    graph = fetch_json(graph_url)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    print(f"  ✓ {len(nodes)} 个节点, {len(edges)} 条边")

    if not nodes:
        print("  ⚠ 没有文档数据，退出")
        return

    # 构建邻接表（source → [target, similarity]）
    adjacency = {}
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        sim = edge["similarity"]
        adjacency.setdefault(src, []).append((tgt, sim))
        adjacency.setdefault(tgt, []).append((src, sim))

    # ── Step 2: 获取每篇文档的详情 ──
    print("[2/5] 获取文档详情...")
    doc_details = {}
    for i, node in enumerate(nodes):
        doc_id = node["id"]
        try:
            detail = fetch_json(f"{api_url}/api/get/{doc_id}")
            doc_details[doc_id] = detail
        except Exception:
            # 降级：用 graph 数据
            doc_details[doc_id] = {
                "id": doc_id,
                "title": node["title"],
                "domain": node["domain"],
                "doc_type": node["doc_type"],
                "author": node["author"],
                "summary": node["summary"],
                "word_count": node["word_count"],
                "tags": [{"name": t, "category": "auto"} for t in node.get("tags", [])],
                "content": node.get("summary", ""),
                "created_at": None,
            }
        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{len(nodes)}")

    print(f"  ✓ 获取完成")

    # ── Step 3: 生成 Markdown 文件 ──
    print("[3/5] 生成文档文件...")
    written = 0
    doc_files = {}  # doc_id → relative filename

    for node in nodes:
        doc_id = node["id"]
        detail = doc_details.get(doc_id, {})
        title = detail.get("title", node["title"])

        # 文件名
        filename = safe_filename(title, doc_id)
        doc_files[doc_id] = filename

        # YAML frontmatter
        tags = [t["name"] for t in detail.get("tags", [])]
        domain = detail.get("domain", node["domain"])
        domain_cn = DOMAIN_LABELS.get(domain, domain)
        author = detail.get("author", node.get("author", ""))
        created = detail.get("created_at", "")
        summary = detail.get("summary", node.get("summary", ""))
        content = detail.get("content", "")

        yaml_lines = [
            "---",
            f'domain: "{domain}"',
            f'domain_cn: "{domain_cn}"',
            f'doc_type: "{detail.get("doc_type", node.get("doc_type", ""))}"',
            f'author: "{sanitize_yaml_value(author)}"',
            f'word_count: {detail.get("word_count", node.get("word_count", 0))}',
            f'tags: {json.dumps(tags, ensure_ascii=False)}',
        ]
        if created:
            yaml_lines.append(f'created: "{created}"')
        if summary:
            yaml_lines.append(f'summary: "{sanitize_yaml_value(summary)}"')

        # 从邻接表生成 wikilinks
        neighbors = adjacency.get(doc_id, [])
        if neighbors:
            neighbor_refs = []
            for nid, sim in neighbors:
                ntitle = ""
                for n in nodes:
                    if n["id"] == nid:
                        ntitle = n["title"]
                        break
                if ntitle:
                    nfname = safe_filename(ntitle, nid)
                    neighbor_refs.append(f"- [[{nfname.replace('.md', '')}|{ntitle}]]（相似度 {sim:.0%}）")
            if neighbor_refs:
                yaml_lines.append(f"related: {json.dumps([n[0] for n in neighbors[:5]], ensure_ascii=False)}")
                related_section = "\n## 相关文档\n\n" + "\n".join(neighbor_refs)
            else:
                related_section = ""
        else:
            related_section = ""

        yaml_lines.append("---")
        yaml_block = "\n".join(yaml_lines)

        # 正文：优先用完整内容，其次摘要
        body = content if content and len(content) > 100 else summary
        if not body:
            body = "（无内容）"

        # 限制正文长度（Obsidian 打开大文件可能卡）
        max_body = 50000
        if len(body) > max_body:
            body = body[:max_body] + f"\n\n---\n*（正文已截断，原文 {len(content):,} 字符）*"

        md_content = f"{yaml_block}\n\n# {title}\n\n{body}\n{related_section}\n"

        filepath = vault / filename
        filepath.write_text(md_content, encoding="utf-8")
        written += 1

    print(f"  ✓ 已写入 {written} 个 .md 文件")

    # ── Step 4: 生成 MOC 索引 ──
    print("[4/5] 生成索引文件...")

    # 按域名分组
    domain_groups = {}
    for node in nodes:
        d = node["domain"]
        domain_groups.setdefault(d, []).append(node)

    moc_lines = [
        "---",
        "tags: [MOC, 索引]",
        f"updated: \"{datetime.now().isoformat()}\"",
        "---",
        "",
        "# 知识库索引",
        "",
        f"> 共 {len(nodes)} 篇文档，{len(edges)} 条语义关联",
        f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for domain, domain_nodes in sorted(domain_groups.items()):
        domain_cn = DOMAIN_LABELS.get(domain, domain)
        moc_lines.append(f"## {domain_cn}（{len(domain_nodes)} 篇）\n")

        # 按类型分组
        type_groups = {}
        for n in domain_nodes:
            dt = n["doc_type"]
            type_groups.setdefault(dt, []).append(n)

        for dtype, dnodes in sorted(type_groups.items()):
            moc_lines.append(f"### {dtype}（{len(dnodes)} 篇）\n")
            for n in sorted(dnodes, key=lambda x: x["title"]):
                fname = doc_files.get(n["id"], safe_filename(n["title"], n["id"]))
                link = fname.replace(".md", "")
                tags_str = ", ".join(n.get("tags", [])[:5])
                moc_lines.append(f"- [[{link}|{n['title']}]]  `{tags_str}`")
            moc_lines.append("")

    # Dataview 查询（可选）
    moc_lines.extend([
        "---",
        "",
        "## Dataview 动态查询",
        "",
        "```dataview",
        "TABLE domain_cn AS \"领域\", doc_type AS \"类型\", author AS \"作者\", word_count AS \"字数\"",
        "FROM \"\"",
        "WHERE domain",
        "SORT domain ASC, doc_type ASC",
        "```",
    ])

    (vault / "00_索引_MOC.md").write_text("\n".join(moc_lines), encoding="utf-8")
    print(f"  ✓ 索引文件: 00_索引_MOC.md")

    # ── Step 5: 生成 Graphviz DOT 文件 ──
    print("[5/5] 生成 Graphviz DOT...")
    dot_lines = [
        "digraph KB {",
        '  rankdir=LR;',
        '  bgcolor="#0f172a";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [color="#475569", arrowsize=0.5];',
        "",
        "  // 颜色定义",
        '  subgraph cluster_law {',
        '    label="法学"; fontcolor="#3b82f6"; color="#3b82f6";',
    ]

    law_nodes = []
    writing_nodes = []
    for node in nodes:
        node_id = node["id"].replace("-", "_")
        title = node["title"][:30]
        tags_str = ", ".join(node.get("tags", [])[:3])
        label = f'{title}\\n{tags_str}'

        if node["domain"] == "law":
            law_nodes.append(f'    "{node_id}" [label="{label}", fillcolor="#1e3a5f", fontcolor="#e2e8f0"];')
        else:
            writing_nodes.append(f'    "{node_id}" [label="{label}", fillcolor="#2d1b4e", fontcolor="#e2e8f0"];')

    dot_lines.extend(law_nodes)
    dot_lines.append("  }")
    dot_lines.append('  subgraph cluster_writing {')
    dot_lines.append('    label="创作"; fontcolor="#a78bfa"; color="#a78bfa";')
    dot_lines.extend(writing_nodes)
    dot_lines.append("  }")

    # 边
    dot_lines.append("")
    for edge in edges:
        src = edge["source"].replace("-", "_")
        tgt = edge["target"].replace("-", "_")
        sim = edge["similarity"]
        # 边颜色随相似度变化
        alpha = int(sim * 255)
        color = f'"#{alpha:02x}{alpha:02x}{alpha:02x}"' if sim >= 0.4 else '"#475569"'
        dot_lines.append(f'  "{src}" -> "{tgt}" [color={color}, penwidth={sim*3:.1f}, weight={sim*100:.0f}];')

    dot_lines.append("}")

    dot_path = vault / "kb_graph.dot"
    dot_path.write_text("\n".join(dot_lines), encoding="utf-8")
    print(f"  ✓ DOT 文件: {dot_path}")

    # ── 完成 ──
    print()
    print("=" * 50)
    print(f"导出完成！")
    print(f"  Obsidian Vault: {vault}")
    print(f"  文档数: {written}")
    print()
    print(f"  用 Obsidian 打开: Open → Open folder as vault → 选择 {vault}")
    print("=" * 50)


# ─── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="知识库 → Obsidian Vault 导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python export_kb_to_obsidian.py
  python export_kb_to_obsidian.py --output ~/MyKnowledgeVault
  python export_kb_to_obsidian.py --api http://47.97.7.165:8765 --min-sim 0.6
        """,
    )
    parser.add_argument(
        "--api", default=DEFAULT_API,
        help=f"KB API 地址（默认: {DEFAULT_API}）"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"输出目录（默认: {DEFAULT_OUTPUT}）"
    )
    parser.add_argument(
        "--min-sim", "-s", type=float, default=DEFAULT_MIN_SIM,
        help=f"语义相似度阈值（默认: {DEFAULT_MIN_SIM}）"
    )
    args = parser.parse_args()

    export_obsidian(args.api, args.output, args.min_sim)


if __name__ == "__main__":
    main()
