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
from starlette.routing import Route
from starlette.responses import JSONResponse, HTMLResponse
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
async function uploadAll(){var btn=document.getElementById('uploadBtn');btn.disabled=true;btn.textContent='上传中...';var ok=0,fail=0;for(var i=0;i<pendingFiles.length;i++){if(pendingFiles[i].status!=='pending')continue;if(pendingFiles[i].file.size>MAX_SIZE){pendingFiles[i].status='error';pendingFiles[i].message='文件过大(>50MB)';fail++;continue;}pendingFiles[i].status='uploading';pendingFiles[i].message='上传中...';renderFileList();try{var b64=await fileToBase64(pendingFiles[i].file);var resp=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:pendingFiles[i].file.name,data:b64,relative_path:pendingFiles[i].relPath||pendingFiles[i].file.name})});var result=await resp.json();if(result.status==='ok'){pendingFiles[i].status='success';pendingFiles[i].message='已上传,自动处理中';ok++;}else{pendingFiles[i].status='error';pendingFiles[i].message=result.error||'失败';fail++;}}catch(err){pendingFiles[i].status='error';pendingFiles[i].message=err.message;fail++;}renderFileList();}btn.disabled=false;btn.textContent='上传全部';if(ok>0){showStatus('success',ok+' 个文件上传成功，系统将在约15秒内自动处理（解析+向量化+入库）');loadStats();}if(fail>0){showStatus('error',fail+' 个文件上传失败');}}
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
    Body: {"filename": "...", "data": "base64...", "relative_path": "..."}
    relative_path 用于保留文件夹结构（如 "小说名/第一章.txt"）
    """
    try:
        body = await request.json()
        filename = body.get("filename", "")
        data_b64 = body.get("data", "")
        relative_path = body.get("relative_path", "")

        if not filename or not data_b64:
            return JSONResponse({"error": "Missing filename or data"}, status_code=400)

        safe_name = os.path.basename(filename)
        if not safe_name:
            return JSONResponse({"error": "Invalid filename"}, status_code=400)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inbox_dir = Path(os.getenv("KB_INBOX_DIR", "/root/kb-inbox"))
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

        log.info(f"文件上传: {safe_name} ({len(file_data)} bytes) -> {filepath.relative_to(inbox_dir)}")

        return JSONResponse({
            "status": "ok",
            "filename": safe_name,
            "saved_as": str(filepath.relative_to(inbox_dir)),
            "size": len(file_data),
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


app = Starlette(
    routes=[
        # MCP SSE 传输
        Route("/sse", handle_sse),
        Route("/messages/", handle_messages, methods=["POST"]),
        # 健康检查
        Route("/health", health),
        # 上传页面
        Route("/upload", upload_page),
        # REST API（供 OpenWebUI 等外部应用）
        Route("/api/search", api_search, methods=["GET", "POST"]),
        Route("/api/semantic_search", api_semantic_search, methods=["POST"]),
        Route("/api/stats", api_stats, methods=["GET"]),
        Route("/api/list", api_list, methods=["GET"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/ingest_text", api_ingest_text, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    port = int(os.getenv("KB_MCP_PORT", "8765"))
    log.info(f"🚀 MCP Server 启动 (SSE 模式): http://0.0.0.0:{port}/sse")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
