# KB-Cloud: 自托管 AI 知识库

> 基于 PostgreSQL + pgvector + 本地 Embedding 的私有知识库系统，支持多格式文档自动入库、语义检索、MCP 协议接入。

## 概述

KB-Cloud 是一个完全自托管的私有知识库系统，具备以下核心能力：

- **零外部依赖**：Embedding 模型本地运行（BAAI/bge-large-zh-v1.5），不消耗任何外部 API token
- **自动入库**：拖入文件即自动解析、提取元数据、打标签、分块、向量化
- **语义检索**：支持全文检索 + 向量语义检索双通道
- **MCP 协议**：标准 MCP (Model Context Protocol) 接入，支持 WorkBuddy 等 AI 工具
- **REST API**：提供搜索、上传、统计等 HTTP 接口
- **网页上传**：内置拖拽上传界面，支持文件、文件夹、文本直接粘贴
- **双领域支持**：法学（law）+ 创作域（writing），各自独立标签体系
- **多格式解析**：PDF / TXT / Markdown / DOCX / EPUB / HTML / PPTX / 图片(OCR) / 压缩包

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户 / AI 客户端                       │
│         (WorkBuddy / OpenWebUI / REST API)              │
└──────────────┬──────────────────────┬───────────────────┘
               │ MCP/SSE              │ REST API
               ▼                      ▼
┌──────────────────────┐   ┌──────────────────────┐
│   MCP Server (:8765) │   │   网页上传界面        │
│  - 8 个 MCP 工具      │   │  - 文件拖拽上传       │
│  - SSE 传输           │   │  - 文件夹选择         │
│  - REST API 端点      │   │  - 文本粘贴入库       │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL 17 + pgvector                    │
│  documents │ chunks(1024维) │ tags │ document_tags       │
└─────────────────────────────────────────────────────────┘
           ▲
           │
┌──────────┴──────────────────────────────────────────────┐
│              Watcher 文件监控守护进程                     │
│  监控 ~/kb-inbox/ → 解析 → 元数据提取 → 打标签 →          │
│  分块 → 向量化 → 入库 → 归档到 ~/kb-archive/             │
└─────────────────────────────────────────────────────────┘
```

## 三容器架构

| 容器 | 作用 | 端口 |
|------|------|------|
| `kb-postgres` | PostgreSQL 17 + pgvector 向量数据库 | 5433 (仅本地) |
| `kb-mcp-server` | MCP Server + REST API + 网页上传 | 8765 (仅本地) |
| `kb-watcher` | 文件监控守护进程，自动入库 | - |

## 快速开始

### 前提条件

- Docker + Docker Compose
- 至少 2GB 可用内存（Embedding 模型需要约 1.5GB）
- 磁盘空间约 3GB（Docker 镜像 + 模型文件）

### 一键部署

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/kb-cloud.git
cd kb-cloud

# 2. 创建环境变量文件
cp .env.example .env
# 编辑 .env，设置数据库密码
vi .env

# 3. 创建入库目录
mkdir -p ~/kb-inbox ~/kb-archive

# 4. 启动所有服务（首次构建约需 5-10 分钟）
docker compose up -d --build

# 5. 等待服务就绪
docker compose logs -f mcp-server
# 看到 "Connected to PostgreSQL" 和 "Uvicorn running" 后按 Ctrl+C 退出

# 6. 健康检查
curl http://localhost:8765/health
```

### 中国大陆用户加速构建

```bash
docker compose build \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --build-arg PIP_TRUSTED_HOST=mirrors.aliyun.com \
  --build-arg HF_ENDPOINT=https://hf-mirror.com
```

## 使用方式

### 1. 文件自动入库

将文件放入 `~/kb-inbox/` 目录，Watcher 会在 15 秒内自动处理：

```bash
cp 论文.pdf ~/kb-inbox/
cp 笔记.txt ~/kb-inbox/
cp 小说章节.docx ~/kb-inbox/

# 查看处理日志
docker logs kb-watcher -f
```

**支持的文件格式：**

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PDF | .pdf | 自动提取标题、作者、摘要 |
| 纯文本 | .txt | - |
| Markdown | .md .markdown | - |
| Word | .docx | - |
| EPUB | .epub | 电子书 |
| HTML | .html .htm | 网页 |
| PPT | .pptx | 幻灯片 |
| 图片 | .jpg .png .bmp .tiff .webp | OCR 识别（需 tesseract） |
| 压缩包 | .zip .tar.gz .tar.bz2 .tar.xz | 自动解压，保留目录结构 |

### 2. 网页上传

浏览器访问 `http://localhost:8765/upload`，支持：
- **文件拖拽上传**：拖入文件即上传
- **文件夹选择**：保留目录结构（如小说章节文件夹）
- **文本直接粘贴**：输入文本即可入库，自动打标签

### 3. MCP 协议接入

在 WorkBuddy 的 MCP 配置中添加：

```json
{
  "knowledge-base": {
    "url": "http://localhost:8765/sse"
  }
}
```

**可用的 MCP 工具：**

| 工具 | 功能 |
|------|------|
| `kb_search` | 全文关键词检索 |
| `kb_semantic_search` | 向量语义检索 |
| `kb_ingest` | 直接文本入库 |
| `kb_index` | 对文档建立向量索引 |
| `kb_list` | 列出所有文档 |
| `kb_get` | 获取文档详情 |
| `kb_tags` | 列出所有标签 |
| `kb_stats` | 知识库统计信息 |

### 4. REST API

```bash
# 语义搜索
curl -X POST http://localhost:8765/api/semantic_search \
  -H "Content-Type: application/json" \
  -d '{"query": "个人信息保护法", "limit": 5}'

# 全文检索
curl "http://localhost:8765/api/search?q=删除权&limit=5"

# 知识库统计
curl http://localhost:8765/api/stats

# 文档列表
curl http://localhost:8765/api/list

# 文本直接入库
curl -X POST http://localhost:8765/api/ingest_text \
  -H "Content-Type: application/json" \
  -d '{"title": "我的笔记", "content": "这是一段笔记内容...", "domain": "writing"}'

# 文件上传
curl -X POST http://localhost:8765/api/upload \
  -F "file=@论文.pdf"
```

### 5. 远程上传脚本

```bash
# 设置远程服务器
export KB_REMOTE_HOST=your-server.com
export KB_REMOTE_USER=root
export KB_REMOTE_PORT=22

# 上传文件
./scripts/upload_kb.sh 论文.pdf 笔记.txt
```

## OpenWebUI 集成

将 `integrations/openwebui_kb_function.py` 的内容作为 Function 添加到 OpenWebUI：

1. 打开 OpenWebUI → Admin Panel → Functions
2. 创建新 Function，类型选择 **Filter**
3. 粘贴 `openwebui_kb_function.py` 的内容
4. 配置 Valves：
   - `KB_API_URL`: 知识库 API 地址（默认 `http://kb-mcp-server:8765`）
   - `TOP_K`: 检索结果数量（默认 5）

用户在对话中提出问题时，Function 会自动检索知识库并将相关内容注入为上下文。

## 数据库结构

### 表结构

| 表 | 说明 |
|----|------|
| `documents` | 文档主表（标题、领域、类型、内容、元数据等） |
| `chunks` | 文本分块表（1024 维向量嵌入） |
| `tags` | 标签表（支持层级标签树） |
| `document_tags` | 文档-标签关联表（含置信度） |

### 双领域设计

| 领域 | domain | 文档类型 | 标签分类 |
|------|--------|----------|----------|
| 法学 | `law` | paper / case / statute / note | 部门法、争议焦点 |
| 创作 | `writing` | novel / chapter / character / worldbuilding | 元类型、角色、情绪基调、场景类型 |

预置 53 个标签（29 法学 + 24 创作），可在 `db-init/01-schema.sql` 中修改。

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KB_DB_PASSWORD` | （必填） | PostgreSQL 数据库密码 |
| `KB_DB_HOST` | localhost | 数据库地址 |
| `KB_DB_PORT` | 5433 | 数据库端口 |
| `KB_DB_NAME` | knowledge_base | 数据库名 |
| `KB_DB_USER` | kb_admin | 数据库用户 |
| `KB_MCP_PORT` | 8765 | MCP Server 端口 |
| `KB_INBOX_DIR` | /root/kb-inbox | 文件入库监控目录 |
| `KB_ARCHIVE_DIR` | /root/kb-archive | 文件归档目录 |
| `KB_EMBEDDING_DEVICE` | cpu | Embedding 设备（cpu / mps / cuda） |

### Docker Compose 卷路径

在 `.env` 文件中可自定义宿主机目录：

```env
KB_INBOX_DIR=/home/user/kb-inbox
KB_ARCHIVE_DIR=/home/user/kb-archive
```

## 远程部署

### SSH 隧道访问

MCP Server 仅监听 127.0.0.1，通过 SSH 隧道安全访问：

```bash
# 建立 SSH 隧道
ssh -L 8765:localhost:8765 user@your-server.com -N

# 本地即可访问
curl http://localhost:8765/health
```

### Nginx 反向代理（可选）

```nginx
server {
    listen 443 ssl;
    server_name kb.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # SSE 支持
        proxy_buffering off;             # SSE 需要
        proxy_read_timeout 86400;        # SSE 长连接
    }
}
```

## 自定义

### 添加标签

在 `db-init/01-schema.sql` 中添加，或运行时通过 SQL 插入：

```sql
INSERT INTO tags (name, domain, category, parent_id)
VALUES ('新标签', 'law', '自定义分类', NULL);
```

### 修改 Embedding 模型

1. 修改 `embedder.py` 中的模型名称
2. 如果维度不是 1024，需同步修改：
   - `db-init/01-schema.sql` 中的 `vector(1024)`
   - `kb_core.py` 中的向量维度
3. 修改 `Dockerfile.mcp` 中的模型下载命令
4. 重建镜像：`docker compose build && docker compose up -d`

### 添加新的文件格式解析器

在 `kb_watcher.py` 的 `extract_text` 函数中添加新的格式处理逻辑。

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 数据库 | PostgreSQL + pgvector | 17 |
| Embedding | BAAI/bge-large-zh-v1.5 | 1024 维 |
| MCP 协议 | mcp (Python SDK) | SSE 传输 |
| Web 框架 | Starlette + Uvicorn | - |
| 文件监控 | watchdog | - |
| PDF 解析 | PyMuPDF (fitz) | - |
| OCR | tesseract + chi_sim | - |

## License

MIT License - 详见 [LICENSE](LICENSE)
