"""
云端 MCP Server — HTTP/SSE 传输模式
监听 8765 端口，WorkBuddy 通过 SSE 连接。
启动时预加载 Embedding 模型，确保首次查询秒级响应。
"""
import os
import sys
import json
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# 确保离线模式
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from kb_core import KnowledgeBase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("kb-mcp")

# 启动时预加载模型
log.info("预加载 Embedding 模型...")
from embedder import get_embedder
_embedder = get_embedder()
_embedder._load()
log.info(f"Embedding 模型就绪，维度: {_embedder.dim}")

# 数据库连接
kb = KnowledgeBase()
log.info("数据库连接就绪")

# MCP Server
server = Server("knowledge-base")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="kb_search",
            description="全文检索知识库。用关键词搜索文档标题和正文，支持按领域和类型过滤。适合精确关键词匹配。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "domain": {"type": "string", "enum": ["law", "writing"], "description": "限定领域"},
                    "doc_type": {"type": "string", "description": "限定文档类型"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="kb_semantic_search",
            description="语义搜索知识库。用自然语言描述你想找的内容，系统自动理解语义并返回最相关的文本段落。适合模糊查询和跨文档关联发现。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言查询"},
                    "domain": {"type": "string", "enum": ["law", "writing"], "description": "限定领域"},
                    "doc_type": {"type": "string", "description": "限定文档类型"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="kb_ingest",
            description="新增文档到知识库。传入标题、领域、类型、正文即可录入。",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "文档标题"},
                    "domain": {"type": "string", "enum": ["law", "writing"]},
                    "doc_type": {"type": "string", "description": "文档类型（如 paper, chapter, note）"},
                    "content": {"type": "string", "description": "文档正文"},
                    "source": {"type": "string", "description": "来源"},
                    "source_url": {"type": "string", "description": "来源链接"},
                    "author": {"type": "string", "description": "作者"},
                    "published_at": {"type": "string", "description": "发布日期（ISO 格式 YYYY-MM-DD）"},
                    "summary": {"type": "string", "description": "摘要"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "标签列表"},
                },
                "required": ["title", "domain", "doc_type", "content"],
            },
        ),
        Tool(
            name="kb_index",
            description="对指定文档执行语义索引（分块+向量化）。新录入文档后调用此工具启用语义搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "文档 UUID"},
                    "force": {"type": "boolean", "default": False, "description": "是否强制重建索引"},
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="kb_list",
            description="列出知识库中的文档，支持按领域和类型过滤。",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "enum": ["law", "writing"]},
                    "doc_type": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "offset": {"type": "integer", "default": 0},
                },
            },
        ),
        Tool(
            name="kb_get",
            description="获取单个文档的完整内容、元数据和关联标签。",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "文档 UUID"},
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="kb_tags",
            description="获取知识库的标签列表，可按领域过滤。",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "enum": ["law", "writing"]},
                },
            },
        ),
        Tool(
            name="kb_stats",
            description="获取知识库统计信息：各领域文档数量、标签数、分块数等。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kb_concept_search",
            description="概念检索：搜索知识库中自动抽取的学术概念（从文档中 KeyBERT 提取）。支持按概念名称模糊搜索，返回概念关联的文档数量和相关概念。可用于文献综述时快速找到某个概念在不同文档中的论述。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "概念名称或关键词"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="kb_concept_detail",
            description="获取单个概念的详细信息：关联文档列表、相关概念网络（共现分析）、概念类别。",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {"type": "string", "description": "概念 UUID"},
                },
                "required": ["concept_id"],
            },
        ),
        Tool(
            name="kb_health_check",
            description="知识健康检查：检测未索引文档、孤儿概念、语义邻近未关联、概念命名冲突、重复文档等 8 类问题，返回健康评分和修复建议。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="kb_summary",
            description="获取文档的结构化摘要（7 模块：核心论点/关键发现/研究方法/核心概念/局限性/知识关联/实践价值）。如果尚未生成，会自动生成。",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "文档 UUID"},
                    "generate": {"type": "boolean", "default": True, "description": "如果摘要不存在是否自动生成"},
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="kb_landscape",
            description="跨文献知识图景报告：概念分布地图、核心集群、知识空白、概念共现网络、增长趋势。用于文献综述和知识全景分析。",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "enum": ["law", "writing"], "description": "限定领域（可选）"},
                },
            },
        ),
        Tool(
            name="kb_context",
            description='获取会话上下文记忆：返回指定会话的查询历史和上下文摘要。用于多轮对话时让 AI 记住之前聊过什么。',
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "会话 ID"},
                },
                "required": ["session_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _run_tool(name, arguments))
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        log.error(f"Tool error {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


def _run_tool(name: str, args: dict) -> dict:
    if name == "kb_search":
        return {
            "results": kb.search(
                query=args["query"],
                domain=args.get("domain"),
                doc_type=args.get("doc_type"),
                limit=args.get("limit", 10),
            )
        }
    elif name == "kb_semantic_search":
        return {
            "results": kb.semantic_search_text(
                query=args["query"],
                domain=args.get("domain"),
                doc_type=args.get("doc_type"),
                limit=args.get("limit", 10),
            )
        }
    elif name == "kb_ingest":
        result = kb.ingest(
            title=args["title"],
            domain=args["domain"],
            doc_type=args["doc_type"],
            content=args["content"],
            source=args.get("source"),
            source_url=args.get("source_url"),
            author=args.get("author"),
            published_at=args.get("published_at"),
            summary=args.get("summary"),
            tags=args.get("tags"),
        )
        return {"document": result}
    elif name == "kb_index":
        return kb.index_document(
            doc_id=args["document_id"],
            force=args.get("force", False),
        )
    elif name == "kb_list":
        return {
            "documents": kb.list_documents(
                domain=args.get("domain"),
                doc_type=args.get("doc_type"),
                limit=args.get("limit", 20),
                offset=args.get("offset", 0),
            )
        }
    elif name == "kb_get":
        doc = kb.get_document(args["document_id"])
        if doc is None:
            return {"error": "Document not found"}
        return {"document": doc}
    elif name == "kb_tags":
        return {"tags": kb.get_tags(domain=args.get("domain"))}
    elif name == "kb_stats":
        return {"stats": kb.get_stats()}
    elif name == "kb_concept_search":
        return {
            "concepts": kb.search_concepts(
                query=args["query"],
                limit=args.get("limit", 20),
            )
        }
    elif name == "kb_concept_detail":
        concept = kb.get_concept(args["concept_id"])
        if concept is None:
            return {"error": "Concept not found"}
        # 附带相关概念
        concept["related_concepts"] = kb.get_related_concepts(args["concept_id"], limit=10)
        # 附带证据片段
        concept["evidence"] = kb.get_concept_evidence(args["concept_id"], limit=10)
        return {"concept": concept}
    elif name == "kb_health_check":
        return kb.health_check()
    elif name == "kb_summary":
        doc_id = args["document_id"]
        generate = args.get("generate", True)
        summary = kb.get_summary(doc_id)
        if not summary and generate:
            summary = kb.generate_summary(doc_id)
        return {"summary": summary} if summary else {"error": "Summary not found and generation failed"}
    elif name == "kb_landscape":
        return kb.generate_knowledge_landscape(domain=args.get("domain"))
    elif name == "kb_context":
        return kb.get_session_context(args["session_id"])
    else:
        return {"error": f"Unknown tool: {name}"}


# ============================================================
# HTTP/SSE 传输层
# ============================================================

sse = SseServerTransport("/messages/")


async def handle_sse(request):
    """SSE 端点 — WorkBuddy 通过此端点订阅消息"""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )


async def handle_messages(request):
    """消息端点 — WorkBuddy POST JSON-RPC 消息"""
    await sse.handle_post_message(request.scope, request.receive, request._send)


async def health(request):
    """健康检查"""
    stats = kb.get_stats()
    return JSONResponse({"status": "ok", **stats})


# ============================================================
# REST API — 供 OpenWebUI / 外部应用调用
# ============================================================

async def api_search(request):
    """关键词搜索 API — GET /api/search?q=关键词&domain=law&limit=10"""
    try:
        params = request.query_params
        query = params.get("q", "")
        if not query:
            body = await request.json() if request.method == "POST" else {}
            query = body.get("query", "")
        if not query:
            return JSONResponse({"error": "Missing 'q' or 'query' parameter"}, status_code=400)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: kb.search(
                query=query,
                domain=params.get("domain") or (body.get("domain") if 'body' in dir() else None),
                doc_type=params.get("doc_type"),
                limit=int(params.get("limit", "10")),
            ),
        )
        return JSONResponse({"results": result})
    except Exception as e:
        log.error(f"API search error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_semantic_search(request):
    """语义搜索 API — POST /api/semantic_search {"query": "...", "limit": 5}"""
    try:
        body = await request.json()
        query = body.get("query", "")
        if not query:
            return JSONResponse({"error": "Missing 'query' in body"}, status_code=400)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: kb.semantic_search_text(
                query=query,
                domain=body.get("domain"),
                doc_type=body.get("doc_type"),
                limit=int(body.get("limit", 5)),
            ),
        )
        return JSONResponse({"results": result})
    except Exception as e:
        log.error(f"API semantic_search error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_stats(request):
    """统计信息 API — GET /api/stats"""
    try:
        stats = await asyncio.get_event_loop().run_in_executor(None, kb.get_stats)
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_list(request):
    """文档列表 API — GET /api/list?domain=law&limit=20"""
    try:
        params = request.query_params
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: kb.list_documents(
                domain=params.get("domain"),
                doc_type=params.get("doc_type"),
                limit=int(params.get("limit", "20")),
                offset=int(params.get("offset", "0")),
            ),
        )
        return JSONResponse({"documents": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_get_document(request):
    """获取单篇文档详情 — GET /api/get/{doc_id}（供导出脚本用）"""
    try:
        doc_id = request.path_params.get("doc_id", "") if hasattr(request, "path_params") else ""
        if not doc_id:
            return JSONResponse({"error": "missing doc_id"}, status_code=400)
        doc = await asyncio.get_event_loop().run_in_executor(None, lambda: kb.get_document(doc_id))
        if not doc:
            return JSONResponse({"error": "document not found"}, status_code=404)
        return JSONResponse(doc)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 文档阅读页 — /view/{doc_id}
# ============================================================

VIEW_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ - 知识库</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;background:#0d1117;color:#d0d7de;line-height:1.8}
.container{max-width:860px;margin:0 auto;padding:24px 32px 64px}
header{margin-bottom:32px;padding-bottom:20px;border-bottom:1px solid #21262d}
header .back{display:inline-flex;align-items:center;gap:6px;color:#58a6ff;text-decoration:none;font-size:14px;margin-bottom:12px}
header .back:hover{text-decoration:underline}
header h1{font-size:26px;font-weight:700;color:#e6edf3;line-height:1.4;word-break:break-word;margin-bottom:12px}
.meta{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:#8b949e}
.meta span{display:inline-flex;align-items:center;gap:4px}
.meta .tag{background:#161b22;border:1px solid #30363d;padding:2px 10px;border-radius:12px;font-size:12px;color:#79c0ff}
.content{font-size:17px;color:#d0d7de}
.content p{margin-bottom:1.2em;text-indent:2em;word-wrap:break-word;overflow-wrap:break-word}
.content p:first-child{text-indent:0}
.content h2,.content h3{color:#e6edf3;margin:1.5em 0 .6em;font-weight:600}
.content blockquote{border-left:3px solid #30363d;margin:1em 0;padding:.5em 1em;color:#8b949e;background:#0d1117;border-radius:0 6px 6px 0}
.content code{background:#161b22;padding:2px 6px;border-radius:4px;font-size:15px;color:#79c0ff}
.content pre{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:16px;overflow-x:auto;margin:1em 0}
.content pre code{background:none;padding:0;font-size:14px}
.chunks-section{margin-top:40px;padding-top:20px;border-top:1px solid #21262d}
.chunks-section h2{font-size:18px;color:#8b949e;margin-bottom:16px;font-weight:500}
.chunk-item{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 20px;margin-bottom:10px}
.chunk-item .chunk-index{font-size:11px;color:#484f58;margin-bottom:6px}
.chunk-item p{font-size:15px;color:#c9d1d9;margin:0;text-indent:0;line-height:1.75}
@media(max-width:640px){
.container{padding:16px}
header h1{font-size:20px}
.content{font-size:16px}
}
</style>
</head>
<body>
<div class="container">
<header>
<a class="back" href="/graph">← 返回知识图谱</a>
<h1>__TITLE__</h1>
<div class="meta">
<span>👤 __AUTHOR__</span>
<span>📂 __DOMAIN__</span>
<span>🏷 __TAGS__</span>
<span>📅 __DATE__</span>
</div>
</header>
<div class="content">__CONTENT__</div>
<div class="chunks-section" id="chunksSection">
<h2>📄 文本分块 (__CHUNK_COUNT__ 段)</h2>
<div id="chunksList"></div>
</div>
</div>
<script>
var chunksData=__CHUNKS_JSON__;
if(chunksData&&chunksData.length){var list=document.getElementById("chunksList");chunksData.forEach(function(c,i){var d=document.createElement("div");d.className="chunk-item";d.innerHTML='<div class="chunk-index">#'+(i+1)+' (相似度 '+((c.similarity||0)*100).toFixed(0)+'%)</div><p>'+escHtml(c.text)+'</p>';list.appendChild(d);});}else{document.getElementById("chunksSection").style.display="none";}
function escHtml(s){if(!s)return"";return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
</script>
</body>
</html>"""


def _format_content_html(raw_text: str) -> str:
    """将原始文本转为 HTML：按自然段分 <p>，保留段落缩进"""
    import re
    # 先清理多余空白
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # 按双换行或多个空行分割段落
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    result = []
    for para in paragraphs:
        # 清理单行内的换行和多余空白
        clean = re.sub(r'[ \t]+', ' ', para.replace('\n', ' ')).strip()
        if clean:
            result.append(f'<p>{clean}</p>')
    return '\n'.join(result)


async def view_page(request):
    """文档阅读页面 - GET /view/{doc_id}"""
    try:
        doc_id = request.path_params.get("doc_id", "")
        if not doc_id:
            return HTMLResponse("<html><body><h1>缺少文档ID</h1></body></html>", status_code=400)

        doc = await asyncio.get_event_loop().run_in_executor(
            None, lambda: kb.get_document(doc_id)
        )
        if not doc:
            return HTMLResponse("<html><body><h1>文档不存在</h1></body></html>", status_code=404)

        title = doc.get("title", "未知文档")
        author = doc.get("author", "未知")
        domain_map = {"law": "法学", "writing": "创作"}
        domain = domain_map.get(doc.get("domain", ""), doc.get("domain", ""))
        tags = doc.get("tags", [])
        tags_html = " ".join(
            f'<span class="tag">{t}</span>' for t in tags[:8]
        )
        date_val = doc.get("published_at") or ""
        content_raw = doc.get("content", "")

        # 格式化正文为 HTML 段落
        content_html = _format_content_html(content_raw)

        # 获取分块数据
        chunks = []
        try:
            chunks_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: kb.search(query=title, limit=50, filter_doc_id=doc_id)
            )
            chunks = [
                {"text": c.get("content", ""), "similarity": c.get("score", 0)}
                for c in (chunks_result.get("results") or [])[:30]
                if c.get("document_id") == doc_id or c.get("id") != doc_id
            ]
        except Exception:
            pass

        html = VIEW_PAGE_HTML
        html = html.replace("__TITLE__", _esc_html(title))
        html = html.replace("__AUTHOR__", _esc_html(author))
        html = html.replace("__DOMAIN__", domain or "未分类")
        html = html.replace("__TAGS__", tags_html or "无标签")
        html = html.replace("__DATE__", date_val or "未知")
        html = html.replace("__CONTENT__", content_html)
        html = html.replace("__CHUNK_COUNT__", str(len(chunks)))
        html = html.replace(
            "__CHUNKS_JSON__",
            json.dumps(chunks, ensure_ascii=False) if chunks else "[]",
        )
        return HTMLResponse(html)

    except Exception as e:
        log.error(f"View page error: {e}", exc_info=True)
        return HTMLResponse(
            f"<html><body><h1>错误: {_esc_html(str(e))}</h1></body></html>",
            status_code=500,
        )


def _esc_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ============================================================
# 网页上传界面 + 文本直接入库
# ============================================================

UPLOAD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识库上传</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;padding:20px}
.container{max-width:720px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #30363d}
header h1{font-size:24px;font-weight:600}
#stats{font-size:14px;color:#8b949e}
.tabs{display:flex;gap:8px;margin-bottom:20px}
.tab{padding:8px 20px;background:#161b22;border:1px solid #30363d;border-radius:8px;color:#8b949e;cursor:pointer;font-size:14px;transition:all .2s}
.tab.active{background:#1f6feb;color:#fff;border-color:#1f6feb}
.panel{display:none;background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px}
.panel.active{display:block}
.dropzone{border:2px dashed #30363d;border-radius:12px;padding:40px;text-align:center;cursor:pointer;transition:all .2s}
.dropzone:hover,.dropzone.dragover{border-color:#58a6ff;background:rgba(88,166,255,.05)}
.dropzone p{color:#8b949e;margin-bottom:12px}
.formats{font-size:12px;color:#484f58;margin-top:12px}
.btn{padding:8px 20px;background:#238636;color:#fff;border:none;border-radius:8px;font-size:14px;cursor:pointer;transition:all .2s}
.btn:hover{background:#2ea043}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-secondary{background:#21262d;border:1px solid #30363d}
.btn-secondary:hover{background:#30363d}
.file-list{margin-top:16px}
.file-item{display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;margin-bottom:4px;font-size:14px}
.file-item:hover{background:#21262d}
.file-item .icon{font-size:18px}
.file-item .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-item .size{color:#8b949e;font-size:12px}
.file-item .status{font-size:12px;color:#8b949e}
.file-item .status.success{color:#3fb950}
.file-item .status.error{color:#f85149}
.upload-actions{margin-top:16px;display:flex;gap:8px}
.form-group{margin-bottom:16px}
.form-group label{display:block;font-size:14px;color:#8b949e;margin-bottom:6px}
.form-group input,.form-group textarea,.form-group select{width:100%;padding:10px 12px;background:#0d1117;border:1px solid #30363d;border-radius:8px;color:#e6edf3;font-size:14px;font-family:inherit}
.form-group input:focus,.form-group textarea:focus,.form-group select:focus{outline:none;border-color:#58a6ff}
.form-group textarea{min-height:200px;resize:vertical}
.form-row{display:flex;gap:12px}
.form-row .form-group{flex:1}
#statusLog{margin-top:20px}
.status-msg{padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:8px;animation:fadeIn .3s;white-space:pre-line}
.status-msg.success{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.3);color:#3fb950}
.status-msg.error{background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);color:#f85149}
.status-msg.info{background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.3);color:#58a6ff}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<div class="container">
<header><h1>\U0001f4da 知识库</h1><div id="stats">加载中...</div></header>
<div class="tabs">
<button class="tab active" data-tab="files" onclick="switchTab('files')">文件上传</button>
<button class="tab" data-tab="text" onclick="switchTab('text')">文本粘贴</button>
</div>
<div id="files-panel" class="panel active">
<div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
<p>\U0001f4c1 拖拽文件/文件夹到此处，或点击选择</p>
<div class="formats">支持: PDF \u00b7 TXT \u00b7 MD \u00b7 DOCX \u00b7 EPUB \u00b7 HTML \u00b7 PPTX \u00b7 JPG \u00b7 PNG \u00b7 ZIP \u00b7 TAR.GZ</div>
</div>
<div class="meta-section" style="margin-top:16px;padding:16px;background:#0d1117;border:1px solid #30363d;border-radius:8px">
<p style="margin:0 0 12px;color:#8b949e;font-size:13px">\u270f\ufe0f 文档元数据（批量上传时所有文件共用，可留空由系统自动提取）</p>
<div class="form-row">
<div class="form-group" style="margin-bottom:0"><label>标题</label><input type="text" id="fileTitle" placeholder="留空则使用文件名"></div>
<div class="form-group" style="margin-bottom:0"><label>作者</label><input type="text" id="fileAuthor" placeholder="作者（可选）"></div>
</div>
<div class="form-row">
<div class="form-group" style="margin-bottom:0"><label>领域</label><select id="fileDomain"><option value="">自动检测</option><option value="law">法学</option><option value="writing">写作</option></select></div>
<div class="form-group" style="margin-bottom:0"><label>标签（逗号分隔）</label><input type="text" id="fileTags" placeholder="如：民法,物权法,比较法（可选）"></div>
</div>
</div>
<div class="upload-actions" style="margin-top:12px;gap:8px">
<button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">选择文件</button>
<button class="btn btn-secondary" onclick="document.getElementById('folderInput').click()">选择文件夹</button>
<button class="btn" id="uploadBtn" onclick="uploadAll()">上传全部</button>
<button class="btn btn-secondary" onclick="clearList()">清空列表</button>
</div>
<input type="file" id="fileInput" multiple hidden>
<input type="file" id="folderInput" webkitdirectory directory multiple hidden>
<div class="file-list" id="fileList"></div>
</div>
<div id="text-panel" class="panel">
<div class="form-group"><label>标题 *</label><input type="text" id="title" placeholder="输入文档标题"></div>
<div class="form-row">
<div class="form-group"><label>作者</label><input type="text" id="author" placeholder="作者（可选）"></div>
<div class="form-group"><label>领域</label><select id="domain"><option value="law">法学</option><option value="writing">写作</option></select></div>
</div>
<div class="form-group"><label>来源链接</label><input type="text" id="sourceUrl" placeholder="URL（可选）"></div>
<div class="form-group"><label>正文内容 *</label><textarea id="content" placeholder="粘贴文本内容...&#10;&#10;可直接粘贴公众号文章、笔记、论文摘要等"></textarea></div>
<button class="btn" id="ingestBtn" onclick="ingestText()">提交入库</button>
</div>
<div id="statusLog"></div>
</div>
<script>
const MAX_SIZE=50*1024*1024;
const IMAGE_EXTS=['.jpg','.jpeg','.png','.bmp','.tiff','.webp'];
let pendingFiles=[];
async function loadStats(){try{const r=await fetch('/api/stats');const d=await r.json();document.getElementById('stats').textContent=d.total_documents+' 篇文档 \u00b7 '+d.total_chunks+' 个分块 \u00b7 '+d.total_tags+' 个标签';}catch(e){document.getElementById('stats').textContent='统计加载失败';}}
function switchTab(t){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));document.querySelector('[data-tab="'+t+'"]').classList.add('active');document.getElementById(t+'-panel').classList.add('active');}
document.getElementById('fileInput').addEventListener('change',function(e){handleFiles(e.target.files);e.target.value='';});
document.getElementById('folderInput').addEventListener('change',function(e){handleFiles(e.target.files);e.target.value='';});
var dz=document.getElementById('dropzone');
dz.addEventListener('dragover',function(e){e.preventDefault();dz.classList.add('dragover');});
dz.addEventListener('dragleave',function(){dz.classList.remove('dragover');});
dz.addEventListener('drop',function(e){e.preventDefault();dz.classList.remove('dragover');handleFiles(e.dataTransfer.files);});
var VALID_EXTS=['.pdf','.txt','.md','.markdown','.docx','.epub','.html','.htm','.pptx','.jpg','.jpeg','.png','.bmp','.tiff','.webp','.zip','.tar.gz','.tgz','.tar.bz2','.tar.xz'];
function handleFiles(files){var skipped=0;for(var i=0;i<files.length;i++){var fn=files[i].name.toLowerCase();var ok=false;for(var j=0;j<VALID_EXTS.length;j++){if(fn.endsWith(VALID_EXTS[j])){ok=true;break;}}if(ok){var relPath=files[i].webkitRelativePath||files[i].name;pendingFiles.push({file:files[i],status:'pending',message:'',relPath:relPath});}else{skipped++;}}if(skipped>0){showStatus('info','已跳过 '+skipped+' 个不支持的文件');}renderFileList();}
function renderFileList(){var list=document.getElementById('fileList');list.innerHTML='';pendingFiles.forEach(function(item){var ext='.'+item.file.name.split('.').pop().toLowerCase();var icon=IMAGE_EXTS.indexOf(ext)>=0?'\U0001f5bc\ufe0f':'\U0001f4c4';var si={pending:'\u23f3',uploading:'\U0001f4e4',success:'\u2705',error:'\u274c'};var d=document.createElement('div');d.className='file-item';var displayName=item.relPath&&item.relPath!==item.file.name?item.relPath:item.file.name;d.innerHTML='<span class="icon">'+icon+'</span><span class="name">'+displayName+'</span><span class="size">'+formatSize(item.file.size)+'</span><span class="status '+item.status+'">'+(si[item.status]||'')+' '+item.message+'</span>';list.appendChild(d);});}
function formatSize(b){if(b<1024)return b+' B';if(b<1048576)return(b/1024).toFixed(1)+' KB';return(b/1048576).toFixed(1)+' MB';}
function fileToBase64(f){return new Promise(function(resolve,reject){var r=new FileReader();r.onload=function(e){resolve(e.target.result.split(',')[1]);};r.onerror=reject;r.readAsDataURL(f);});}
async function uploadAll(){var btn=document.getElementById('uploadBtn');btn.disabled=true;btn.textContent='上传中...';var ok=0,fail=0;var meta={title:document.getElementById('fileTitle').value.trim(),author:document.getElementById('fileAuthor').value.trim(),domain:document.getElementById('fileDomain').value,tags:document.getElementById('fileTags').value.trim()};for(var i=0;i<pendingFiles.length;i++){if(pendingFiles[i].status!=='pending')continue;if(pendingFiles[i].file.size>MAX_SIZE){pendingFiles[i].status='error';pendingFiles[i].message='文件过大(>50MB)';fail++;continue;}pendingFiles[i].status='uploading';pendingFiles[i].message='上传中...';renderFileList();try{var b64=await fileToBase64(pendingFiles[i].file);var payload={filename:pendingFiles[i].file.name,data:b64,relative_path:pendingFiles[i].relPath||pendingFiles[i].file.name};if(meta.title||meta.author||meta.domain||meta.tags)payload.metadata=meta;var resp=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});var result=await resp.json();if(result.status==='ok'){pendingFiles[i].status='success';pendingFiles[i].message='已上传,自动处理中';ok++;}else{pendingFiles[i].status='error';pendingFiles[i].message=result.error||'失败';fail++;}}catch(err){pendingFiles[i].status='error';pendingFiles[i].message=err.message;fail++;}renderFileList();}btn.disabled=false;btn.textContent='上传全部';if(ok>0){showStatus('success',ok+' 个文件上传成功，系统将在约15秒内自动处理（解析+向量化+入库）');loadStats();}if(fail>0){showStatus('error',fail+' 个文件上传失败');}}
function clearList(){pendingFiles=[];renderFileList();}
async function ingestText(){var title=document.getElementById('title').value.trim();var content=document.getElementById('content').value.trim();var author=document.getElementById('author').value.trim();var domain=document.getElementById('domain').value;var url=document.getElementById('sourceUrl').value.trim();if(!title){showStatus('error','请输入标题');return;}if(!content){showStatus('error','请输入正文内容');return;}var btn=document.getElementById('ingestBtn');btn.disabled=true;btn.textContent='入库中...';try{var resp=await fetch('/api/ingest_text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:title,content:content,author:author||'未知',domain:domain,source_url:url||null})});var result=await resp.json();if(result.status==='ok'){var tags=result.tags&&result.tags.length>0?result.tags.join(' \u00b7 '):'无标签';showStatus('success','入库成功！\\n文档ID: '+result.document.id.substring(0,8)+'...\\n标签: '+tags);document.getElementById('title').value='';document.getElementById('content').value='';document.getElementById('sourceUrl').value='';loadStats();}else{showStatus('error',result.error||'入库失败');}}catch(err){showStatus('error',err.message);}btn.disabled=false;btn.textContent='提交入库';}
function showStatus(type,msg){var log=document.getElementById('statusLog');var d=document.createElement('div');d.className='status-msg '+type;d.textContent=msg;log.insertBefore(d,log.firstChild);setTimeout(function(){d.style.opacity='0';setTimeout(function(){d.remove();},300);},10000);}
loadStats();
</script>
</body>
</html>"""


async def upload_page(request):
    """上传页面 - GET /upload"""
    return HTMLResponse(UPLOAD_HTML)


async def api_upload(request):
    """文件上传 API - POST /api/upload
    Body: {"filename": "...", "data": "base64...", "relative_path": "...", "metadata": {...}}
    relative_path 用于保留文件夹结构（如 "小说名/第一章.txt"）
    metadata 可选: {title, author, domain, tags} → 保存为 .meta.json 侧车文件供 watcher 使用
    """
    try:
        body = await request.json()
        filename = body.get("filename", "")
        data_b64 = body.get("data", "")
        relative_path = body.get("relative_path", "")
        metadata = body.get("metadata")  # 可选的元数据字典

        if not filename or not data_b64:
            return JSONResponse({"error": "Missing filename or data"}, status_code=400)

        safe_name = os.path.basename(filename)
        if not safe_name:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inbox_dir = Path("/root/kb-inbox")
        inbox_dir.mkdir(parents=True, exist_ok=True)

        # 如果有 relative_path（文件夹上传），保留子目录结构
        if relative_path and relative_path != filename:
            # 安全处理路径：只保留相对路径部分，防止路径遍历
            clean_path = relative_path.replace("\\", "/").lstrip("/")
            # 去掉 .. 等危险路径组件
            parts = [p for p in clean_path.split("/") if p and p != ".." and p != "."]
            if parts:
                # 用第一级目录名 + 时间戳前缀避免冲突
                subdir = inbox_dir / parts[0]
                subdir.mkdir(parents=True, exist_ok=True)
                dest_name = f"{timestamp}_{safe_name}"
                filepath = subdir / dest_name
            else:
                dest_name = f"{timestamp}_{safe_name}"
                filepath = inbox_dir / dest_name
        else:
            dest_name = f"{timestamp}_{safe_name}"
            filepath = inbox_dir / dest_name

        file_data = base64.b64decode(data_b64)
        filepath.write_bytes(file_data)

        # 如果有元数据，保存为侧车文件供 watcher 使用
        meta_file = None
        if metadata and isinstance(metadata, dict):
            meta_filepath = filepath.with_suffix(filepath.suffix + ".meta.json")
            # 只保留有效字段
            clean_meta = {k: v for k, v in metadata.items() if v and k in ("title", "author", "domain", "tags")}
            if clean_meta:
                meta_filepath.write_text(json.dumps(clean_meta, ensure_ascii=False), encoding="utf-8")
                meta_file = str(meta_filepath.relative_to(inbox_dir))

        log.info(f"文件上传: {safe_name} ({len(file_data)} bytes) -> {filepath.relative_to(inbox_dir)}")

        return JSONResponse({
            "status": "ok",
            "filename": safe_name,
            "saved_as": str(filepath.relative_to(inbox_dir)),
            "size": len(file_data),
            "meta_file": meta_file,
            "message": "文件已上传，watcher 将在约15秒内自动处理",
        })
    except Exception as e:
        log.error(f"API upload error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_ingest_text(request):
    """文本直接入库 API - POST /api/ingest_text
    Body: {"title": "...", "content": "...", "domain": "law", "author": "...", ...}
    """
    try:
        body = await request.json()
        title = body.get("title", "").strip()
        content = body.get("content", "").strip()

        if not title or not content:
            return JSONResponse({"error": "Missing title or content"}, status_code=400)

        domain = body.get("domain", "law")
        author = body.get("author", "未知")
        source_url = body.get("source_url")

        from pdf_extractor import extract_summary, extract_keywords, extract_year, auto_tag_enhanced

        summary = extract_summary(content)
        keywords = extract_keywords(content)
        year = extract_year(content)

        with kb.conn.cursor() as cur:
            cur.execute("SELECT name FROM tags WHERE domain = %s", (domain,))
            db_tags = {r[0] for r in cur.fetchall()}

        tags = auto_tag_enhanced(content, title, keywords, db_tags)
        published_at = f"{year}-01-01" if year else None

        doc_type = "note"
        if any(k in content[:2000] for k in ["摘要", "关键词", "参考文献"]):
            doc_type = "paper_thematic"

        result = kb.ingest(
            title=title,
            domain=domain,
            doc_type=doc_type,
            content=content,
            source="网页直接录入",
            source_url=source_url,
            author=author,
            published_at=published_at,
            summary=summary,
            metadata={"keywords": keywords, "ingested_by": "web_text", "char_count": len(content)},
            tags=tags,
        )

        log.info(f"文本入库: {title[:40]} -> {result.get('id', '?')[:8]}... tags={tags}")

        return JSONResponse({
            "status": "ok",
            "document": result,
            "tags": tags,
            "summary": summary[:200] if summary else "",
        })
    except Exception as e:
        log.error(f"API ingest_text error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# REST API — 概念系统
# ============================================================

async def api_concepts_list(request):
    """列出概念 — GET /api/concepts?category=法学概念&sort=doc_count&limit=50"""
    try:
        category = request.query_params.get("category")
        sort_by = request.query_params.get("sort", "doc_count")
        limit = min(int(request.query_params.get("limit", "50")), 200)
        concepts = kb.list_concepts(category=category, sort_by=sort_by, limit=limit)
        return JSONResponse({"concepts": concepts, "total": len(concepts)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_concepts_search(request):
    """搜索概念 — GET /api/concepts/search?q=物权行为&limit=20"""
    try:
        query = request.query_params.get("q", "")
        if not query:
            return JSONResponse({"error": "Missing query parameter 'q'"}, status_code=400)
        limit = min(int(request.query_params.get("limit", "20")), 100)
        concepts = kb.search_concepts(query=query, limit=limit)
        return JSONResponse({"concepts": concepts, "total": len(concepts)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_concept_detail(request):
    """获取概念详情 — GET /api/concept/{concept_id}"""
    try:
        concept_id = request.path_params.get("concept_id", "")
        include_related = request.query_params.get("related", "1") == "1"
        include_evidence = request.query_params.get("evidence", "1") == "1"
        concept = kb.get_concept(concept_id)
        if not concept:
            return JSONResponse({"error": "Concept not found"}, status_code=404)
        if include_related:
            concept["related_concepts"] = kb.get_related_concepts(concept_id, limit=10)
        if include_evidence:
            concept["evidence"] = kb.get_concept_evidence(concept_id, limit=10)
        return JSONResponse({"concept": concept})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_concept_extract(request):
    """为已有文档批量抽取概念 — POST /api/concept/extract
    Body: {"doc_id": "uuid", "max_docs": 0}
    如果不传 doc_id，则为所有未抽取过的文档批量处理。
    """
    try:
        body = {}
        if request.method == "POST":
            body = await request.json()
        doc_id = body.get("doc_id")
        if doc_id:
            concepts = kb.extract_concepts(doc_id, top_n=10)
            return JSONResponse({"status": "ok", "doc_id": doc_id, "concepts": concepts})
        else:
            max_docs = body.get("max_docs", 0)
            result = kb.extract_concepts_for_existing(max_docs=max_docs)
            return JSONResponse({"status": "ok", **result})
    except Exception as e:
        log.error(f"API concept_extract error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_concept_stats(request):
    """概念统计 — GET /api/concept/stats"""
    try:
        stats = kb.get_concept_stats()
        return JSONResponse({"stats": stats})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# REST API — 知识复利层（P0-P3）
# ============================================================

async def api_health_check(request):
    """知识健康检查 — GET /api/health_check"""
    try:
        report = await asyncio.get_event_loop().run_in_executor(None, kb.health_check)
        return JSONResponse(report)
    except Exception as e:
        log.error(f"API health_check error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_summary(request):
    """文档结构化摘要 — GET /api/summary/{doc_id} 或 POST /api/summary"""
    try:
        if request.method == "POST":
            body = await request.json()
            doc_id = body.get("document_id", "")
            force = body.get("force", False)
        else:
            doc_id = request.path_params.get("doc_id", "")
            force = request.query_params.get("force", "0") == "1"

        if not doc_id:
            return JSONResponse({"error": "Missing document_id"}, status_code=400)

        def _get_or_generate():
            existing = kb.get_summary(doc_id)
            if existing and not force:
                return existing
            return kb.generate_summary(doc_id)

        summary = await asyncio.get_event_loop().run_in_executor(None, _get_or_generate)
        if not summary:
            return JSONResponse({"error": "Failed to generate summary"}, status_code=500)
        return JSONResponse({"summary": summary})
    except Exception as e:
        log.error(f"API summary error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_knowledge_landscape(request):
    """跨文献知识图景 — GET /api/knowledge_landscape?domain=law"""
    try:
        domain = request.query_params.get("domain")
        landscape = await asyncio.get_event_loop().run_in_executor(
            None, lambda: kb.generate_knowledge_landscape(domain=domain)
        )
        return JSONResponse(landscape)
    except Exception as e:
        log.error(f"API landscape error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_conflicts(request):
    """矛盾检测/列表 — GET /api/conflicts?detect=1 或 GET /api/conflicts"""
    try:
        detect = request.query_params.get("detect", "0") == "1"
        resolved = request.query_params.get("resolved", "0") == "1"
        limit = min(int(request.query_params.get("limit", "30")), 100)

        if detect:
            conflicts = await asyncio.get_event_loop().run_in_executor(
                None, lambda: kb.detect_conflicts(max_pairs=limit)
            )
            return JSONResponse({"conflicts": conflicts, "total": len(conflicts), "action": "detected"})
        else:
            conflicts = kb.get_conflicts(resolved=resolved, limit=limit)
            return JSONResponse({"conflicts": conflicts, "total": len(conflicts)})
    except Exception as e:
        log.error(f"API conflicts error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_concept_evidence(request):
    """概念证据积累 — GET /api/concept/{concept_id}/evidence"""
    try:
        concept_id = request.path_params.get("concept_id", "")
        limit = min(int(request.query_params.get("limit", "20")), 50)
        evidence = kb.get_concept_evidence(concept_id, limit=limit)
        return JSONResponse({"concept_id": concept_id, "evidence": evidence, "total": len(evidence)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_operations_log(request):
    """操作日志 — GET /api/operations?type=ingest&limit=50"""
    try:
        op_type = request.query_params.get("type")
        entity_id = request.query_params.get("entity_id")
        limit = min(int(request.query_params.get("limit", "50")), 200)
        offset = int(request.query_params.get("offset", "0"))
        logs = kb.get_operations_log(operation_type=op_type, entity_id=entity_id, limit=limit, offset=offset)
        return JSONResponse({"logs": logs, "total": len(logs)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_query_history(request):
    """查询历史/会话记忆 — GET /api/query_history?session_id=xxx"""
    try:
        session_id = request.query_params.get("session_id", "")
        if not session_id:
            return JSONResponse({"error": "Missing session_id"}, status_code=400)
        limit = min(int(request.query_params.get("limit", "10")), 50)
        context = kb.get_session_context(session_id, limit=limit)
        return JSONResponse(context)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_multiperspective(request):
    """多视角概念提取 — POST /api/multiperspective {"doc_id": "...", "perspectives": ["legal","social"]}"""
    try:
        body = await request.json()
        doc_id = body.get("doc_id", "")
        perspectives = body.get("perspectives", ["legal", "social", "technical"])
        if not doc_id:
            return JSONResponse({"error": "Missing doc_id"}, status_code=400)
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: kb.extract_concepts_multiperspective(doc_id, perspectives)
        )
        return JSONResponse(result)
    except Exception as e:
        log.error(f"API multiperspective error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 网页页面 — 概念浏览
# ============================================================

CONCEPTS_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>概念检索 - KB-Cloud</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{padding:12px 20px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:16px}
.header h1{font-size:16px;font-weight:600;white-space:nowrap}
.header a{color:#60a5fa;text-decoration:none;font-size:13px}
.header .nav{gap:12px;display:flex}
.search-box{flex:1;display:flex;gap:8px;margin:0 20px}
.search-box input{flex:1;padding:6px 12px;border:1px solid #334155;border-radius:6px;background:#0f172a;color:#e2e8f0;font-size:14px}
.search-box input:focus{outline:none;border-color:#3b82f6}
.search-box button{padding:6px 16px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px}
.container{max-width:960px;margin:20px auto;padding:0 20px}
.stats-bar{display:flex;gap:24px;margin-bottom:20px;padding:12px 16px;background:#1e293b;border-radius:8px;font-size:13px;color:#94a3b8}
.stats-bar strong{color:#e2e8f0;font-size:16px}
.concept-list{display:flex;flex-direction:column;gap:8px}
.concept-card{background:#1e293b;border-radius:8px;padding:14px 18px;display:flex;align-items:center;gap:16px;transition:background .15s;text-decoration:none;color:inherit}
.concept-card:hover{background:#273548}
.concept-name{font-size:15px;font-weight:600;color:#e2e8f0;flex:1}
.concept-category{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500;white-space:nowrap}
.cat-law{background:#1e3a5f;color:#60a5fa}
.cat-theory{background:#1e3a5f;color:#a78bfa}
.cat-method{background:#1e3a5f;color:#34d399}
.cat-case{background:#292524;color:#fbbf24}
.cat-general{background:#1e293b;color:#94a3b8}
.cat-article{background:#292524;color:#f87171}
.concept-count{font-size:12px;color:#64748b;min-width:40px;text-align:center}
.concept-count strong{display:block;font-size:18px;color:#60a5fa}
.tabs{display:flex;gap:4px;margin-bottom:16px}
.tabs button{padding:6px 14px;border:1px solid #334155;border-radius:6px;background:transparent;color:#94a3b8;cursor:pointer;font-size:13px}
.tabs button.active{background:#3b82f6;color:#fff;border-color:#3b82f6}
.empty{text-align:center;padding:60px 20px;color:#64748b}
.concept-detail{background:#1e293b;border-radius:8px;padding:20px}
.concept-detail h2{font-size:20px;margin-bottom:8px}
.concept-detail .meta{font-size:13px;color:#94a3b8;margin-bottom:16px}
.concept-detail .related{margin-top:16px}
.concept-detail .related h3{font-size:14px;margin-bottom:8px;color:#94a3b8}
.related-tag{display:inline-block;padding:4px 10px;background:#1e3a5f;color:#60a5fa;border-radius:12px;margin:2px 4px;font-size:12px;text-decoration:none}
.related-tag:hover{background:#2d5080}
.doc-link{display:block;padding:8px 12px;border-left:2px solid #3b82f6;margin:4px 0;font-size:13px;color:#94a3b8;text-decoration:none}
.doc-link:hover{color:#e2e8f0;background:#1e293b}
</style>
</head>
<body>
<div class="header">
  <h1>KB-Cloud 概念检索</h1>
  <div class="nav">
    <a href="/graph">知识图谱</a>
    <a href="/upload">上传文档</a>
  </div>
  <div class="search-box">
    <input type="text" id="searchInput" placeholder="搜索概念..." onkeydown="if(event.key==='Enter')searchConcepts()">
    <button onclick="searchConcepts()">搜索</button>
  </div>
</div>
<div class="container">
  <div class="stats-bar" id="statsBar">加载中...</div>
  <div class="tabs">
    <button class="active" onclick="switchTab('all')">全部</button>
    <button onclick="switchTab('法学概念')">法学概念</button>
    <button onclick="switchTab('学术理论')">学术理论</button>
    <button onclick="switchTab('方法论')">方法论</button>
    <button onclick="switchTab('案例引用')">案例引用</button>
  </div>
  <div id="content">
    <div class="concept-list" id="conceptList">加载中...</div>
  </div>
</div>
<script>
var currentTab = 'all';
var allConcepts = [];

async function loadStats(){
  try{var r=await fetch('/api/concept/stats');var d=await r.json();
  document.getElementById('statsBar').innerHTML='<span>总概念: <strong>'+d.stats.total_concepts+'</strong></span><span>总关联: <strong>'+d.stats.total_links+'</strong></span>';
  }catch(e){}
}

async function loadConcepts(category){
  currentTab = category || 'all';
  var tabs = document.querySelectorAll('.tabs button');
  tabs.forEach(function(b){b.classList.toggle('active', b.textContent === (category||'全部'));});
  
  var url = '/api/concepts?sort=doc_count&limit=100';
  if(category && category !== 'all') url += '&category=' + encodeURIComponent(category);
  
  try{
    var r = await fetch(url);
    var d = await r.json();
    allConcepts = d.concepts || [];
    renderConcepts(allConcepts);
  }catch(e){
    document.getElementById('conceptList').innerHTML = '<div class="empty">加载失败</div>';
  }
}

function renderConcepts(concepts){
  if(!concepts.length){
    document.getElementById('conceptList').innerHTML = '<div class="empty">暂无概念数据<br>上传文档后系统将自动抽取概念</div>';
    return;
  }
  var html = '';
  for(var i=0;i<concepts.length;i++){
    var c = concepts[i];
    var catClass = 'cat-general';
    if(c.category === '法学概念') catClass='cat-law';
    else if(c.category === '学术理论') catClass='cat-theory';
    else if(c.category === '方法论') catClass='cat-method';
    else if(c.category === '案例引用') catClass='cat-case';
    else if(c.category === '法条引用') catClass='cat-article';
    html += '<a class="concept-card" href="javascript:showDetail(\''+c.id+'\')">';
    html += '<span class="concept-name">'+c.name+'</span>';
    html += '<span class="concept-category '+catClass+'">'+c.category+'</span>';
    html += '<span class="concept-count"><strong>'+c.doc_count+'</strong>篇</span>';
    html += '</a>';
  }
  document.getElementById('conceptList').innerHTML = html;
}

async function searchConcepts(){
  var q = document.getElementById('searchInput').value.trim();
  if(!q){loadConcepts(currentTab);return;}
  try{
    var r = await fetch('/api/concepts/search?q='+encodeURIComponent(q)+'&limit=50');
    var d = await r.json();
    allConcepts = d.concepts || [];
    renderConcepts(d.concepts||[]);
  }catch(e){}
}

function switchTab(cat){loadConcepts(cat);}

async function showDetail(id){
  try{
    var r = await fetch('/api/concept/'+id+'?related=1');
    var d = await r.json();
    var c = d.concept;
    var html = '<div class="concept-detail">';
    html += '<h2>'+c.name+'</h2>';
    html += '<div class="meta">类别: '+c.category+' | 关联文档: '+c.doc_count+'篇 | 更新: '+(c.updated_at||'').substr(0,10)+'</div>';
    html += '<button onclick="loadConcepts(currentTab)" style="padding:4px 12px;background:#3b82f6;color:#fff;border:none;border-radius:4px;cursor:pointer;margin-bottom:16px">← 返回列表</button>';
    
    if(c.related_concepts && c.related_concepts.length){
      html += '<div class="related"><h3>相关概念</h3>';
      for(var i=0;i<c.related_concepts.length;i++){
        var rc = c.related_concepts[i];
        html += '<a class="related-tag" href="javascript:showDetail(\''+rc.id+'\')">'+rc.name+' ('+rc.co_occurrence+')</a>';
      }
      html += '</div>';
    }
    
    if(c.documents && c.documents.length){
      html += '<div class="related" style="margin-top:20px"><h3>关联文档</h3>';
      for(var i=0;i<c.documents.length;i++){
        var doc = c.documents[i];
        html += '<a class="doc-link" href="/view/'+doc.id+'">'+doc.title+' <span style="color:#64748b">('+doc.author+')</span></a>';
      }
      html += '</div>';
    }
    html += '</div>';
    document.getElementById('conceptList').innerHTML = html;
  }catch(e){
    document.getElementById('conceptList').innerHTML = '<div class="empty">加载失败: '+e.message+'</div>';
  }
}

loadStats();
loadConcepts('all');
</script>
</body>
</html>
"""


async def concepts_page(request):
    """概念浏览页面 — GET /concepts"""
    return HTMLResponse(CONCEPTS_PAGE_HTML)


async def graph_page(request):
    """知识图谱可视化页面 - GET /graph（Graphviz 服务端渲染，零前端依赖）"""
    min_sim = request.query_params.get("min_similarity", "0.65")
    try:
        sim_val = float(min_sim)
        sim_val = max(0.2, min(0.95, sim_val))
        min_sim = str(sim_val)
    except ValueError:
        min_sim = "0.65"
        sim_val = 0.65

    html = GRAPH_PAGE_HTML
    html = html.replace("__MIN_SIM__", min_sim)
    html = html.replace("__THRESHOLD_VAL__", str(int(sim_val * 100)))
    html = html.replace("__THRESHOLD_DISPLAY__", f"{sim_val:.2f}")
    return HTMLResponse(html)


# ============================================================
# Web UI — 知识健康检查页面 (/health_lint)
# ============================================================

HEALTH_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识健康检查 - KB-Cloud</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{padding:12px 20px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:16px}
.header h1{font-size:16px;font-weight:600}
.header a{color:#60a5fa;text-decoration:none;font-size:13px}
.header .nav{gap:12px;display:flex}
.container{max-width:960px;margin:20px auto;padding:0 20px}
.score-card{background:#1e293b;border-radius:12px;padding:24px;text-align:center;margin-bottom:20px}
.score-card .score{font-size:48px;font-weight:700;line-height:1}
.score-card .label{font-size:14px;color:#94a3b8;margin-top:8px}
.score-high{color:#3fb950}
.score-medium{color:#fbbf24}
.score-low{color:#f85149}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.metric-card{background:#1e293b;border-radius:8px;padding:14px;text-align:center}
.metric-card .value{font-size:24px;font-weight:700;color:#60a5fa}
.metric-card .label{font-size:11px;color:#94a3b8;margin-top:4px}
.issue-section{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
.issue-section h2{font-size:15px;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.issue-badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500}
.badge-high{background:rgba(248,81,73,.15);color:#f85149}
.badge-medium{background:rgba(251,191,36,.15);color:#fbbf24}
.badge-low{background:rgba(100,116,139,.15);color:#94a3b8}
.issue-message{font-size:13px;color:#cbd5e1;margin-bottom:12px}
.issue-items{display:flex;flex-direction:column;gap:6px}
.issue-item{padding:8px 12px;background:#0f172a;border-radius:6px;font-size:12px;color:#94a3b8;display:flex;align-items:center;gap:8px}
.issue-item a{color:#60a5fa;text-decoration:none}
.issue-item a:hover{text-decoration:underline}
.recommendations{background:#1e3a5f;border-radius:12px;padding:20px;margin-bottom:20px}
.recommendations h2{font-size:15px;margin-bottom:12px;color:#60a5fa}
.rec-item{font-size:13px;color:#cbd5e1;margin-bottom:8px;padding-left:20px;position:relative}
.rec-item:before{content:"\2192";position:absolute;left:0;color:#60a5fa}
.empty{text-align:center;padding:60px 20px;color:#64748b}
.loading{text-align:center;padding:40px;color:#64748b}
button.refresh{padding:6px 16px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px}
button.refresh:hover{background:#2563eb}
</style>
</head>
<body>
<div class="header">
  <h1>KB-Cloud 知识健康检查</h1>
  <div class="nav">
    <a href="/graph">知识图谱</a>
    <a href="/concepts">概念检索</a>
    <a href="/landscape">知识图景</a>
    <a href="/upload">上传文档</a>
  </div>
  <button class="refresh" onclick="loadHealth()" style="margin-left:auto">重新检查</button>
</div>
<div class="container">
  <div id="content"><div class="loading">正在执行知识健康检查...</div></div>
</div>
<script>
var severityClass={high:'badge-high',medium:'badge-medium',low:'badge-low'};
async function loadHealth(){
  var c=document.getElementById('content');
  c.innerHTML='<div class="loading">正在执行知识健康检查...</div>';
  try{
    var r=await fetch('/api/health_check');
    var d=await r.json();
    renderReport(d);
  }catch(e){
    c.innerHTML='<div class="empty">检查失败: '+e.message+'</div>';
  }
}
function renderReport(d){
  var scoreClass=d.overall_score>=80?'score-high':d.overall_score>=60?'score-medium':'score-low';
  var m=d.metrics||{};
  var html='<div class="score-card"><div class="score '+scoreClass+'">'+d.overall_score+'</div><div class="label">知识健康评分</div></div>';
  html+='<div class="metrics-grid">';
  html+=metricCard(m.total_docs||0,'文档总数');
  html+=metricCard(m.total_chunks||0,'文本分块');
  html+=metricCard(m.total_concepts||0,'概念总数');
  html+=metricCard(m.total_concept_links||0,'概念关联');
  html+=metricCard(m.cross_doc_concepts||0,'跨文档概念');
  html+=metricCard(m.structured_summaries||0,'结构化摘要');
  html+=metricCard(m.unresolved_conflicts||0,'未解决矛盾');
  html+='</div>';
  if(d.recommendations&&d.recommendations.length){
    html+='<div class="recommendations"><h2>修复建议</h2>';
    for(var i=0;i<d.recommendations.length;i++){
      html+='<div class="rec-item">'+d.recommendations[i]+'</div>';
    }
    html+='</div>';
  }
  if(d.issues&&d.issues.length){
    for(var i=0;i<d.issues.length;i++){
      var iss=d.issues[i];
      var sc=severityClass[iss.severity]||'badge-low';
      html+='<div class="issue-section">';
      html+='<h2>'+iss.type.replace(/_/g,' ')+' <span class="issue-badge '+sc+'">'+iss.severity.toUpperCase()+'</span></h2>';
      html+='<div class="issue-message">'+iss.message+'</div>';
      if(iss.items&&iss.items.length){
        html+='<div class="issue-items">';
        for(var j=0;j<Math.min(iss.items.length,5);j++){
          var item=iss.items[j];
          if(item.id&&item.title){
            html+='<div class="issue-item"><a href="/view/'+item.id+'">'+item.title+'</a></div>';
          }else if(item.name){
            html+='<div class="issue-item">'+item.name+(item.doc_count?' ('+item.doc_count+'篇)':'')+'</div>';
          }else if(item.doc_a&&item.doc_b){
            html+='<div class="issue-item">'+item.doc_a.title+' <-> '+item.doc_b.title+(item.similarity?' ('+(item.similarity*100).toFixed(0)+'%)':'')+'</div>';
          }else if(item.names){
            html+='<div class="issue-item">'+item.names.join(' / ')+'</div>';
          }
        }
        if(iss.items.length>5){html+='<div class="issue-item" style="color:#64748b">... 还有 '+(iss.items.length-5)+' 项</div>';}
        html+='</div>';
      }
      html+='</div>';
    }
  }else{
    html+='<div class="issue-section" style="text-align:center;color:#3fb950;padding:40px">所有检查项通过，知识库状态良好</div>';
  }
  document.getElementById('content').innerHTML=html;
}
function metricCard(val,label){
  return '<div class="metric-card"><div class="value">'+val+'</div><div class="label">'+label+'</div></div>';
}
loadHealth();
</script>
</body>
</html>
"""

async def health_lint_page(request):
    """知识健康检查页面 — GET /health_lint"""
    return HTMLResponse(HEALTH_PAGE_HTML)


# ============================================================
# Web UI — 知识图景页面 (/landscape)
# ============================================================

LANDSCAPE_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识图景 - KB-Cloud</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.header{padding:12px 20px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:16px}
.header h1{font-size:16px;font-weight:600}
.header a{color:#60a5fa;text-decoration:none;font-size:13px}
.header .nav{gap:12px;display:flex}
.header select{padding:4px 10px;border:1px solid #334155;border-radius:6px;background:#0f172a;color:#e2e8f0;font-size:13px}
.container{max-width:960px;margin:20px auto;padding:0 20px}
.section{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
.section h2{font-size:15px;margin-bottom:12px;color:#60a5fa}
.insights{background:#1e3a5f;border-radius:12px;padding:20px;margin-bottom:16px}
.insights h2{font-size:15px;margin-bottom:12px;color:#60a5fa}
.insight-item{font-size:13px;color:#cbd5e1;margin-bottom:8px;padding-left:20px;position:relative}
.insight-item:before{content:"\2217";position:absolute;left:0;color:#60a5fa}
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.cluster-card{padding:12px;background:#0f172a;border-radius:8px;border-left:3px solid #3b82f6}
.cluster-card .name{font-size:14px;font-weight:600;color:#e2e8f0}
.cluster-card .meta{font-size:11px;color:#94a3b8;margin-top:4px}
.gap-list{display:flex;flex-wrap:wrap;gap:6px}
.gap-tag{padding:4px 10px;background:#292524;border-radius:12px;font-size:12px;color:#fbbf24}
.co-occurrence-list{display:flex;flex-direction:column;gap:6px}
.co-item{padding:8px 12px;background:#0f172a;border-radius:6px;font-size:13px;color:#94a3b8;display:flex;align-items:center;gap:8px}
.co-item .pair{color:#e2e8f0;font-weight:500}
.co-item .count{background:#1e3a5f;color:#60a5fa;padding:2px 8px;border-radius:10px;font-size:11px}
.trend-table{width:100%;border-collapse:collapse}
.trend-table th,.trend-table td{padding:8px 12px;text-align:left;border-bottom:1px solid #334155;font-size:13px}
.trend-table th{color:#94a3b8;font-weight:500}
.trend-table td{color:#e2e8f0}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-bottom:16px}
.summary-card{background:#1e293b;border-radius:8px;padding:14px;text-align:center}
.summary-card .val{font-size:24px;font-weight:700;color:#60a5fa}
.summary-card .lbl{font-size:11px;color:#94a3b8;margin-top:4px}
.loading{text-align:center;padding:40px;color:#64748b}
.empty{text-align:center;padding:40px;color:#64748b}
.dist-row{display:flex;align-items:center;gap:12px;padding:6px 0;font-size:13px}
.dist-bar{flex:1;height:8px;background:#0f172a;border-radius:4px;overflow:hidden}
.dist-fill{height:100%;background:#3b82f6;border-radius:4px}
.dist-label{min-width:120px;color:#94a3b8}
.dist-count{min-width:40px;text-align:right;color:#60a5fa;font-weight:600}
</style>
</head>
<body>
<div class="header">
  <h1>KB-Cloud 知识图景</h1>
  <div class="nav">
    <a href="/graph">知识图谱</a>
    <a href="/concepts">概念检索</a>
    <a href="/health_lint">健康检查</a>
    <a href="/upload">上传文档</a>
  </div>
  <select id="domainFilter" onchange="loadLandscape()">
    <option value="">全部领域</option>
    <option value="law">法学</option>
    <option value="writing">创作</option>
  </select>
</div>
<div class="container">
  <div id="content"><div class="loading">正在生成知识图景报告...</div></div>
</div>
<script>
async function loadLandscape(){
  var c=document.getElementById('content');
  var domain=document.getElementById('domainFilter').value;
  c.innerHTML='<div class="loading">正在生成知识图景报告...</div>';
  try{
    var url='/api/knowledge_landscape';
    if(domain)url+='?domain='+domain;
    var r=await fetch(url);
    var d=await r.json();
    renderLandscape(d);
  }catch(e){
    c.innerHTML='<div class="empty">加载失败: '+e.message+'</div>';
  }
}
function renderLandscape(d){
  var s=d.summary||{};
  var html='<div class="summary-grid">';
  html+=sumCard(s.total_concepts||0,'概念总数');
  html+=sumCard(s.core_clusters||0,'核心集群');
  html+=sumCard(s.knowledge_gaps||0,'知识空白');
  html+=sumCard(s.co_occurrence_pairs||0,'概念共现对');
  html+='</div>';
  if(d.insights&&d.insights.length){
    html+='<div class="insights"><h2>AI 洞察</h2>';
    for(var i=0;i<d.insights.length;i++){html+='<div class="insight-item">'+d.insights[i]+'</div>';}
    html+='</div>';
  }
  if(d.knowledge_map&&d.knowledge_map.core_clusters&&d.knowledge_map.core_clusters.length){
    html+='<div class="section"><h2>核心概念集群</h2><div class="cluster-grid">';
    for(var i=0;i<d.knowledge_map.core_clusters.length;i++){
      var cc=d.knowledge_map.core_clusters[i];
      html+='<div class="cluster-card"><div class="name">'+cc.name+'</div><div class="meta">'+cc.category+' | '+cc.doc_count+'篇 | '+cc.related_concepts+'个关联</div></div>';
    }
    html+='</div></div>';
  }
  if(d.knowledge_gaps&&d.knowledge_gaps.length){
    html+='<div class="section"><h2>知识空白（仅 1 篇文档支撑）</h2><div class="gap-list">';
    for(var i=0;i<Math.min(d.knowledge_gaps.length,30);i++){
      html+='<span class="gap-tag">'+d.knowledge_gaps[i].name+'</span>';
    }
    html+='</div></div>';
  }
  if(d.discipline_distribution&&d.discipline_distribution.length){
    var maxCnt=0;
    for(var i=0;i<d.discipline_distribution.length;i++){if(d.discipline_distribution[i].count>maxCnt)maxCnt=d.discipline_distribution[i].count;}
    html+='<div class="section"><h2>学科分布</h2>';
    for(var i=0;i<d.discipline_distribution.length;i++){
      var dd=d.discipline_distribution[i];
      var pct=maxCnt>0?(dd.count/maxCnt*100):0;
      html+='<div class="dist-row"><span class="dist-label">'+dd.domain+' / '+dd.doc_type+'</span><div class="dist-bar"><div class="dist-fill" style="width:'+pct+'%"></div></div><span class="dist-count">'+dd.count+'</span></div>';
    }
    html+='</div>';
  }
  if(d.concept_co_occurrences&&d.concept_co_occurrences.length){
    html+='<div class="section"><h2>概念共现网络 (Top 20)</h2><div class="co-occurrence-list">';
    for(var i=0;i<d.concept_co_occurrences.length;i++){
      var co=d.concept_co_occurrences[i];
      html+='<div class="co-item"><span class="pair">'+co.concept_a+' <-> '+co.concept_b+'</span><span class="count">'+co.co_occurrence+'</span></div>';
    }
    html+='</div></div>';
  }
  if(d.growth_trends&&d.growth_trends.length){
    html+='<div class="section"><h2>增长趋势</h2><table class="trend-table"><tr><th>年份</th><th>文档数</th><th>概念数</th></tr>';
    for(var i=0;i<d.growth_trends.length;i++){
      var t=d.growth_trends[i];
      html+='<tr><td>'+(t.year||'未知')+'</td><td>'+t.docs+'</td><td>'+t.concepts+'</td></tr>';
    }
    html+='</table></div>';
  }
  document.getElementById('content').innerHTML=html;
}
function sumCard(val,label){return '<div class="summary-card"><div class="val">'+val+'</div><div class="lbl">'+label+'</div></div>';}
loadLandscape();
</script>
</body>
</html>
"""

async def landscape_page(request):
    """知识图景页面 — GET /landscape"""
    return HTMLResponse(LANDSCAPE_PAGE_HTML)


async def api_graph(request):
    """图谱数据 API - GET /api/graph?min_similarity=0.5"""
    try:
        min_sim = float(request.query_params.get("min_similarity", "0.5"))
        min_sim = max(0.2, min(0.95, min_sim))
        data = kb.get_graph_data(min_similarity=min_sim)
        return JSONResponse(data)
    except Exception as e:
        log.error(f"API graph error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_graph_svg(request):
    """Graphviz SVG 图谱 — GET /api/graph_svg?min_similarity=0.65（服务端渲染，零 JS）"""
    try:
        import graphviz as gv

        min_sim = float(request.query_params.get("min_similarity", "0.65"))
        min_sim = max(0.2, min(0.95, min_sim))
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: kb.get_graph_data(min_similarity=min_sim)
        )

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not nodes:
            return HTMLResponse(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100" width="100%" height="300">'
                '<rect width="400" height="100" fill="#0f172a"/>'
                '<text x="200" y="50" text-anchor="middle" fill="#94a3b8" font-size="14">暂无文档数据</text>'
                '</svg>',
                media_type="image/svg+xml",
            )

        # 边太多时截取 top-N（按相似度降序），避免渲染超时
        MAX_EDGES = 200
        if len(edges) > MAX_EDGES:
            edges = sorted(edges, key=lambda e: e["similarity"], reverse=True)[:MAX_EDGES]

        # 只保留有边连接的节点（减少孤立节点）
        connected_ids = set()
        for e in edges:
            connected_ids.add(e["source"])
            connected_ids.add(e["target"])
        # 保留所有节点（包括孤立的），但如果太多就只保留有连接的
        if len(nodes) > 60:
            nodes = [n for n in nodes if n["id"] in connected_ids]

        # 根据节点数量选择引擎
        # sfdp: 超快的大图力导向布局;  neato: 中等图;  dot: 小图层次布局
        engine = "sfdp" if len(nodes) > 20 else "dot"

        dot = gv.Digraph("KB", format="svg", engine=engine)
        dot.attr(
            bgcolor="#0f172a",
            fontname="Helvetica",
            overlap="false",
            splines="true" if len(nodes) <= 30 else "false",
            K="2.0",
        )
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
                 fontsize="10", margin="0.08,0.04")
        dot.attr("edge", fontname="Helvetica", fontsize="8")

        LAW_FILL = "#1e3a5f"
        LAW_BORDER = "#3b82f6"
        WRITING_FILL = "#2d1b4e"
        WRITING_BORDER = "#a78bfa"

        for node in nodes:
            nid = node["id"].replace("-", "_")[:20]
            title = node["title"][:24]
            tags = ", ".join(node.get("tags", [])[:2])
            label = f"{title}\\n{tags}" if tags else title
            view_url = f"/view/{node['id']}"

            if node["domain"] == "law":
                dot.node(nid, label=label, fillcolor=LAW_FILL, fontcolor="#e2e8f0",
                         color=LAW_BORDER, penwidth="1.0", URL=view_url, target="_self")
            elif node["domain"] == "writing":
                dot.node(nid, label=label, fillcolor=WRITING_FILL, fontcolor="#e2e8f0",
                         color=WRITING_BORDER, penwidth="1.0", URL=view_url, target="_self")
            else:
                dot.node(nid, label=label, fillcolor="#1e293b", fontcolor="#e2e8f0",
                         URL=view_url, target="_self")

        # 边太多时不显示百分比标签（渲染慢）
        show_labels = len(edges) <= 50
        for edge in edges:
            src = edge["source"].replace("-", "_")[:20]
            tgt = edge["target"].replace("-", "_")[:20]
            sim = edge["similarity"]
            alpha = int(min(sim, 1.0) * 120 + 60)
            color = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
            if show_labels:
                dot.edge(src, tgt, color=color, penwidth=str(max(0.5, sim * 2.0)),
                        label=f" {sim:.0%}", fontcolor="#64748b")
            else:
                dot.edge(src, tgt, color=color, penwidth=str(max(0.3, sim * 1.5)))

        svg_bytes = dot.pipe()
        svg_str = svg_bytes.decode("utf-8")

        svg_start = svg_str.find("<svg")
        if svg_start >= 0:
            svg_str = svg_str[svg_start:]

        return HTMLResponse(svg_str, media_type="image/svg+xml")

    except ImportError:
        log.error("graphviz Python 库未安装，回退到 JSON 数据")
        return JSONResponse({"error": "graphviz not installed on server"}, status_code=500)
    except Exception as e:
        log.error(f"graph_svg error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


GRAPH_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识图谱 - KB-Cloud</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
.header{padding:12px 20px;background:#1e293b;border-bottom:1px solid #334155;display:flex;align-items:center;gap:16px;flex-shrink:0}
.header h1{font-size:16px;font-weight:600}
.header a{color:#60a5fa;text-decoration:none;font-size:13px}
.controls{display:flex;align-items:center;gap:12px;margin-left:auto}
.controls label{font-size:13px;color:#94a3b8}
.controls input[type=range]{width:120px;accent-color:#3b82f6}
.controls span{font-size:13px;color:#60a5fa;min-width:32px;font-weight:600}
.graph-area{flex:1;overflow:auto;display:flex;align-items:flex-start;justify-content:center;padding:8px}
.graph-area svg{max-width:100%;height:auto;background:#0f172a}
.legend{display:flex;gap:16px;margin-left:20px}
.legend span{font-size:12px;display:inline-flex;align-items:center;gap:4px}
.legend .dot{width:10px;height:10px;border-radius:2px;display:inline-block}
.error{color:#ef4444;text-align:center;padding:40px;font-size:14px}
.hint{color:#64748b;font-size:12px;text-align:center;padding:4px 0}
</style>
</head>
<body>
<div class="header">
  <h1>KB-Cloud 知识图谱</h1>
  <a href="/upload">上传文档</a>
  <div class="legend">
    <span><span class="dot" style="background:#3b82f6"></span>法学</span>
    <span><span class="dot" style="background:#a78bfa"></span>创作</span>
  </div>
  <form class="controls" method="get" action="/graph">
    <label for="threshold">相似度阈值</label>
    <input type="range" id="threshold" name="min_similarity" min="20" max="95" value="__THRESHOLD_VAL__" step="5"
           oninput="document.getElementById('val').textContent=(this.value/100).toFixed(2)"
           onchange="this.form.submit()">
    <span id="val">__THRESHOLD_DISPLAY__</span>
  </form>
</div>
<div class="hint">💡 点击图谱节点可查看文档详情 · 拖动滑块调整相似度阈值</div>
<div class="graph-area" id="graphArea">
  <p style="color:#64748b">加载中...</p>
</div>
<script>
async function loadGraph(){
  var area = document.getElementById('graphArea');
  area.innerHTML = '<p style="color:#64748b">渲染中...（文档较多，请耐心等待）</p>';
  try {
    var controller = new AbortController();
    var timeoutId = setTimeout(function(){controller.abort();}, 90000);
    var resp = await fetch('/api/graph_svg?min_similarity=__MIN_SIM__', {signal: controller.signal});
    clearTimeout(timeoutId);
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    var svgText = await resp.text();
    area.innerHTML = svgText;
    area.querySelectorAll('a').forEach(function(a){
      a.setAttribute('target','_self');
    });
  } catch(e) {
    area.innerHTML = '<div class="error">图谱加载失败: '+e.message+'<br>建议调高相似度阈值减少边数量<br><a href="/api/graph" style="color:#60a5fa">查看原始数据</a></div>';
  }
}
loadGraph();
</script>
</body>
</html>
"""




# 静态资源（vis-network.js 等）— 带 fallback 防止 NoneType 错误
import os as _os
_static_dir = "/app/static"
_static_app = StaticFiles(directory=_static_dir) if _os.path.isdir(_static_dir) else None

async def static_fallback(scope, receive, send):
    """静态资源 ASGI 应用 — 带 fallback 防止 TypeError"""
    if _static_app is None:
        response = JSONResponse({"error": "static dir not found"}, status_code=404)
        await response(scope, receive, send)
        return
    try:
        await _static_app(scope, receive, send)
    except Exception:
        response = JSONResponse({"error": "static file not found"}, status_code=404)
        await response(scope, receive, send)


app = Starlette(
    routes=[
        # MCP SSE 传输
        Route("/sse", handle_sse),
        Route("/messages/", handle_messages, methods=["POST"]),
        # 健康检查
        Route("/health", health),
        # 文档阅读页
        Route("/view/{doc_id:str}", view_page),
        # 上传页面
        Route("/upload", upload_page),
        # 知识图谱页面
        Route("/graph", graph_page),
        # 概念浏览页面
        Route("/concepts", concepts_page),
        # 知识健康检查页面
        Route("/health_lint", health_lint_page),
        # 知识图景页面
        Route("/landscape", landscape_page),
        # REST API（供 OpenWebUI 等外部应用）
        Route("/api/search", api_search, methods=["GET", "POST"]),
        Route("/api/semantic_search", api_semantic_search, methods=["POST"]),
        Route("/api/stats", api_stats, methods=["GET"]),
        Route("/api/list", api_list, methods=["GET"]),
        Route("/api/graph", api_graph, methods=["GET"]),
        Route("/api/graph_svg", api_graph_svg, methods=["GET"]),
        Route("/api/get/{doc_id:str}", api_get_document, methods=["GET"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/ingest_text", api_ingest_text, methods=["POST"]),
        # 概念系统 API
        Route("/api/concepts", api_concepts_list, methods=["GET"]),
        Route("/api/concepts/search", api_concepts_search, methods=["GET"]),
        Route("/api/concept/stats", api_concept_stats, methods=["GET"]),
        Route("/api/concept/extract", api_concept_extract, methods=["POST"]),
        Route("/api/concept/{concept_id:str}/evidence", api_concept_evidence, methods=["GET"]),
        Route("/api/concept/{concept_id:str}", api_concept_detail, methods=["GET"]),
        # 知识复利层 API（P0-P3）
        Route("/api/health_check", api_health_check, methods=["GET"]),
        Route("/api/summary/{doc_id:str}", api_summary, methods=["GET"]),
        Route("/api/summary", api_summary, methods=["POST"]),
        Route("/api/knowledge_landscape", api_knowledge_landscape, methods=["GET"]),
        Route("/api/conflicts", api_conflicts, methods=["GET"]),
        Route("/api/operations", api_operations_log, methods=["GET"]),
        Route("/api/query_history", api_query_history, methods=["GET"]),
        Route("/api/multiperspective", api_multiperspective, methods=["POST"]),
        # 静态资源（带 fallback）
        Mount("/static", app=static_fallback, name="static"),
    ],
)


if __name__ == "__main__":
    port = int(os.getenv("KB_MCP_PORT", "8765"))
    log.info(f"🚀 MCP Server 启动 (SSE 模式): http://0.0.0.0:{port}/sse")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
