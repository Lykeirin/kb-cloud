import os
import json
import uuid
import logging
from datetime import date, datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kb-mcp")


class KnowledgeBase:
    """知识库核心操作类"""

    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        """连接数据库"""
        db_config = {
            "host": os.getenv("KB_DB_HOST", "localhost"),
            "port": int(os.getenv("KB_DB_PORT", "5433")),
            "dbname": os.getenv("KB_DB_NAME", "knowledge_base"),
            "user": os.getenv("KB_DB_USER", "kb_admin"),
            "password": os.getenv("KB_DB_PASSWORD", "change_me_please"),
        }
        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = False
        register_vector(self.conn)
        logger.info(f"Connected to PostgreSQL at {db_config['host']}:{db_config['port']}")

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """全文检索（中文友好：ILIKE 匹配 + 标题优先加权）"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = ["d.status = 'ready'"]
            params = []

            if domain:
                conditions.append("d.domain = %s")
                params.append(domain)
            if doc_type:
                conditions.append("d.doc_type = %s")
                params.append(doc_type)

            where = " AND ".join(conditions)
            search_pattern = f"%{query}%"
            params_full = [search_pattern, search_pattern] + params + [limit]

            sql = f"""
                SELECT
                    d.id,
                    d.title,
                    d.domain,
                    d.doc_type,
                    d.summary,
                    d.author,
                    d.metadata,
                    d.created_at,
                    CASE
                        WHEN d.title ILIKE %s THEN 1.0
                        ELSE 0.5
                    END AS rank
                FROM documents d
                WHERE {where}
                  AND (d.title ILIKE %s OR d.content ILIKE %s)
                ORDER BY rank DESC, d.created_at DESC
                LIMIT %s
            """
            params_full = [search_pattern] + params + [search_pattern, search_pattern, limit]
            cur.execute(sql, params_full)
            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "domain": row["domain"],
                    "doc_type": row["doc_type"],
                    "summary": row["summary"],
                    "author": row["author"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "score": float(row["rank"]),
                })
            return results

    def semantic_search(
        self,
        query_embedding: list[float],
        domain: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """向量语义检索"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = ["d.status = 'ready'"]
            params = [query_embedding]

            if domain:
                conditions.append("d.domain = %s")
                params.append(domain)
            if doc_type:
                conditions.append("d.doc_type = %s")
                params.append(doc_type)

            where = " AND ".join(conditions)
            # embedding 在 SELECT 和 ORDER BY 中各用一次
            params.append(query_embedding)
            params.append(limit)

            sql = f"""
                SELECT
                    c.id AS chunk_id,
                    d.id AS document_id,
                    d.title,
                    d.author,
                    d.domain,
                    d.doc_type,
                    c.content,
                    (1.0 - (c.embedding <=> %s::vector)) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE {where}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "title": row["title"],
                    "author": row["author"] or "",
                    "domain": row["domain"],
                    "doc_type": row["doc_type"],
                    "content": row["content"],
                    "score": float(row["score"]),
                })
            return results

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> list[str]:
        """
        将文本切分为重叠分块。

        中文友好：优先按段落切分（换行符），超长段落再按句号切分。
        """
        if not text or not text.strip():
            return []

        # 第一步：按段落切分
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += ("\n" if current_chunk else "") + para
            else:
                # 当前 chunk 满了，保存
                if current_chunk:
                    chunks.append(current_chunk)
                # 新段落太长，按句号切分
                if len(para) > chunk_size:
                    sub_chunks = self._split_long_paragraph(para, chunk_size, overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # 合并过短的 chunk（<100 字）
        merged = []
        buffer = ""
        for c in chunks:
            if len(buffer) + len(c) < chunk_size:
                buffer += ("\n" if buffer else "") + c
            else:
                if buffer:
                    merged.append(buffer)
                buffer = c
        if buffer:
            merged.append(buffer)

        logger.info(f"文本分块完成: {len(merged)} 个块 (原文 {len(text)} 字)")
        return merged

    def _split_long_paragraph(
        self,
        para: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """对超长段落按句号/分号切分"""
        sentences = []
        for sep in ["。", "；", "！", "？"]:
            parts = para.split(sep)
            if len(parts) > 1:
                sentences = [p.strip() + sep for p in parts if p.strip()]
                break
        if not sentences:
            # 无标点，硬切
            sentences = [
                para[i : i + chunk_size]
                for i in range(0, len(para), chunk_size - overlap)
            ]

        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) <= chunk_size:
                current += s
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return chunks

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata_list: Optional[list[dict]] = None,
    ) -> int:
        """
        批量插入文本分块及其向量。

        返回插入的 chunk 数量。
        """
        if not chunks or not embeddings:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) 与 embeddings ({len(embeddings)}) 数量不匹配")

        with self.conn.cursor() as cur:
            count = 0
            for i, (content, emb) in enumerate(zip(chunks, embeddings)):
                meta = json.dumps(metadata_list[i] if metadata_list else {})
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, token_count, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET content = EXCLUDED.content,
                        token_count = EXCLUDED.token_count,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        doc_id,
                        i,
                        content,
                        len(content),
                        emb,
                        meta,
                    ),
                )
                count += 1
            self.conn.commit()

        logger.info(f"chunk 入库完成: doc_id={doc_id[:8]}..., chunks={count}")
        return count

    def index_document(
        self,
        doc_id: str,
        force: bool = False,
    ) -> dict:
        """
        对文档做分块 + 向量化 + 入库。

        返回 {"chunks": N, "status": "ok"}
        """
        from embedder import get_embedder

        # 检查是否已索引
        if not force:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s", (doc_id,))
                existing = cur.fetchone()[0]
                if existing > 0:
                    logger.info(f"文档 {doc_id[:8]} 已有 {existing} 个 chunk，跳过（force=False）")
                    return {"chunks": existing, "status": "already_indexed"}

        # 获取文档内容
        with self.conn.cursor() as cur:
            cur.execute("SELECT content, word_count FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                return {"chunks": 0, "status": "no_content"}
            content = row[0]

        # 分块
        chunks = self.chunk_text(content)
        if not chunks:
            return {"chunks": 0, "status": "empty_content"}

        # 向量化
        embedder = get_embedder()
        embeddings = embedder.encode_batch(chunks)

        # 入库
        count = self.add_chunks(doc_id, chunks, embeddings)
        return {"chunks": count, "status": "ok"}

    def semantic_search_text(
        self,
        query: str,
        domain: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        自然语言语义搜索。

        自动将查询文本编码为向量，然后做向量相似度检索。
        返回与语义最相关的文本块。
        """
        from embedder import get_embedder

        embedder = get_embedder()
        query_embedding = embedder.encode_query(query)

        return self.semantic_search(query_embedding, domain, doc_type, limit)

    def ingest(
        self,
        title: str,
        domain: str,
        doc_type: str,
        content: str,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        author: Optional[str] = None,
        published_at: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """新增文档"""
        doc_id = uuid.uuid4()

        # 摘要：优先用传入的，否则取正文前 200 字
        auto_summary = (
            summary
            if summary
            else (content[:200] + "..." if content and len(content) > 200 else content)
        )

        pub_date = None
        if published_at:
            try:
                pub_date = date.fromisoformat(published_at)
            except (ValueError, TypeError):
                pass

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, title, domain, doc_type, content, source,
                    source_url, author, published_at, word_count, char_count,
                    summary, metadata, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ready')
                """,
                (
                    str(doc_id),
                    title,
                    domain,
                    doc_type,
                    content,
                    source,
                    source_url,
                    author,
                    pub_date,
                    len(content.split()) if content else 0,
                    len(content) if content else 0,
                    auto_summary,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

            # 处理标签
            if tags:
                for tag_name in tags:
                    cur.execute(
                        "SELECT id FROM tags WHERE name = %s",
                        (tag_name,),
                    )
                    tag_row = cur.fetchone()
                    if tag_row:
                        cur.execute(
                            "INSERT INTO document_tags (document_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (str(doc_id), tag_row[0]),
                        )

            self.conn.commit()

        # 自动触发语义索引（异步不阻塞）
        try:
            index_result = self.index_document(str(doc_id), force=False)
            logger.info(f"自动索引完成: {index_result['chunks']} chunks")
        except Exception as e:
            logger.warning(f"自动索引失败（不影响录入）: {e}")

        return {
            "id": str(doc_id),
            "title": title,
            "domain": domain,
            "doc_type": doc_type,
            "status": "ready",
        }

    def list_documents(
        self,
        domain: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """列出文档"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = []
            params = []

            if domain:
                conditions.append("domain = %s")
                params.append(domain)
            if doc_type:
                conditions.append("doc_type = %s")
                params.append(doc_type)

            where = " AND ".join(conditions) if conditions else "TRUE"

            sql = f"""
                SELECT id, title, domain, doc_type, author, summary,
                       word_count, status, created_at
                FROM documents
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            cur.execute(sql, params)
            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "domain": row["domain"],
                    "doc_type": row["doc_type"],
                    "author": row["author"],
                    "summary": row["summary"],
                    "word_count": row["word_count"],
                    "status": row["status"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                })
            return results

    def get_document(self, doc_id: str) -> Optional[dict]:
        """获取单个文档详情"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            # 获取关联标签
            cur.execute(
                """
                SELECT t.name, t.category
                FROM tags t
                JOIN document_tags dt ON t.id = dt.tag_id
                WHERE dt.document_id = %s
                """,
                (doc_id,),
            )
            tags = [{"name": t["name"], "category": t["category"]} for t in cur.fetchall()]

            return {
                "id": str(row["id"]),
                "title": row["title"],
                "domain": row["domain"],
                "doc_type": row["doc_type"],
                "source": row["source"],
                "source_url": row["source_url"],
                "content": row["content"],
                "summary": row["summary"],
                "author": row["author"],
                "published_at": row["published_at"].isoformat() if row["published_at"] else None,
                "word_count": row["word_count"],
                "status": row["status"],
                "metadata": row["metadata"],
                "tags": tags,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }

    def get_tags(self, domain: Optional[str] = None) -> list[dict]:
        """获取标签列表"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if domain:
                cur.execute(
                    "SELECT id, name, domain, category, parent_id FROM tags WHERE domain = %s ORDER BY category, name",
                    (domain,),
                )
            else:
                cur.execute(
                    "SELECT id, name, domain, category, parent_id FROM tags ORDER BY domain, category, name"
                )
            rows = cur.fetchall()
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "domain": row["domain"],
                    "category": row["category"],
                    "parent_id": str(row["parent_id"]) if row["parent_id"] else None,
                }
                for row in rows
            ]

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT domain, doc_type, COUNT(*) FROM documents WHERE status = 'ready' GROUP BY domain, doc_type ORDER BY domain, doc_type")
            breakdown = {}
            for row in cur.fetchall():
                domain, doc_type, count = row
                if domain not in breakdown:
                    breakdown[domain] = {}
                breakdown[domain][doc_type] = count

            cur.execute("SELECT COUNT(*) FROM documents WHERE status = 'ready'")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM tags")
            tag_count = cur.fetchone()[0]

        return {
            "total_documents": total,
            "total_chunks": chunk_count,
            "total_tags": tag_count,
            "breakdown": breakdown,
        }

    def get_graph_data(self, min_similarity: float = 0.5) -> dict:
        """
        构建知识图谱数据：节点（文档）+ 边（文档间语义相似度）

        节点包含文档元数据和标签，边权重 = 文档间最大 chunk 余弦相似度。
        边仅返回相似度 >= min_similarity 的文档对。
        """
        # rollback any aborted transaction before we start
        try:
            self.conn.rollback()
        except Exception:
            pass

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Step 1: 获取所有已索引文档（节点）
            cur.execute("""
                SELECT d.id, d.title, d.domain, d.author, d.doc_type,
                       d.word_count, d.summary, d.created_at
                FROM documents d
                WHERE d.status = 'ready'
                ORDER BY d.created_at DESC
            """)
            doc_rows = cur.fetchall()

            nodes = []
            doc_ids = set()
            for row in doc_rows:
                doc_ids.add(str(row["id"]))
                # 获取文档标签
                cur.execute(
                    "SELECT t.name, t.category FROM tags t "
                    "JOIN document_tags dt ON t.id = dt.tag_id "
                    "WHERE dt.document_id = %s",
                    (row["id"],),
                )
                tag_rows = cur.fetchall()

                nodes.append({
                    "id": str(row["id"]),
                    "title": row["title"],
                    "domain": row["domain"],
                    "author": row["author"] or "",
                    "doc_type": row["doc_type"],
                    "word_count": row["word_count"],
                    "summary": row["summary"] or "",
                    "tags": [t["name"] for t in tag_rows],
                    "primary_tag": tag_rows[0]["name"] if tag_rows else row["domain"],
                })

            # Step 2: 批量计算文档间语义相似度（取最大 chunk 相似度作为边权重）
            if len(doc_ids) >= 2:
                cur.execute("""
                    SELECT
                        d1.id AS doc_a,
                        d2.id AS doc_b,
                        MAX(1.0 - (c1.embedding <=> c2.embedding)) AS similarity
                    FROM chunks c1
                    JOIN documents d1 ON c1.document_id = d1.id
                    JOIN chunks c2 ON c2.document_id != c1.document_id
                    JOIN documents d2 ON c2.document_id = d2.id
                    WHERE d1.id < d2.id
                    GROUP BY d1.id, d2.id
                    HAVING MAX(1.0 - (c1.embedding <=> c2.embedding)) >= %s
                    ORDER BY similarity DESC
                """, (min_similarity,))
                edge_rows = cur.fetchall()
            else:
                edge_rows = []

            edges = []
            for row in edge_rows:
                edges.append({
                    "source": str(row["doc_a"]),
                    "target": str(row["doc_b"]),
                    "similarity": round(float(row["similarity"]), 4),
                })

            logger.info(
                "图谱数据: %d 节点, %d 边 (min_similarity=%.2f)",
                len(nodes), len(edges), min_similarity,
            )

            return {"nodes": nodes, "edges": edges}
