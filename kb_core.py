import os
import json
import re
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

    # ============================================================
    # 概念系统（KeyBERT 自动抽取 + 知识复利）
    # ============================================================

    @staticmethod
    def _normalize_concept(name: str) -> str:
        """归一化概念名称，用于去重"""
        n = name.strip().lower()
        n = re.sub(r'[\s\u3000\-_·•、，。；：！？""''（）【】《》]', '', n)
        return n

    @staticmethod
    def _classify_concept(name: str) -> str:
        """根据名称启发式分类"""
        law_keywords = ['法', '权', '诉', '判', '罪', '刑', '合同', '侵权', '物权',
                        '债权', '不当得利', '无因管理', '善意取得', '公示公信']
        theory_keywords = ['理论', '学说', '主义', '学派', '思想', '体系']
        method_keywords = ['数据', '算法', '模型', '框架', '路径', '机制', '构建']
        article_keywords = ['第', '条', '款', '项']
        case_keywords = ['案例', '判决', '法院', '裁判']

        if any(k in name for k in case_keywords):
            return '案例引用'
        if any(k in name for k in article_keywords):
            return '法条引用'
        if any(k in name for k in law_keywords):
            return '法学概念'
        if any(k in name for k in theory_keywords):
            return '学术理论'
        if any(k in name for k in method_keywords):
            return '方法论'
        return '通用概念'

    def extract_concepts(self, doc_id: str, top_n: int = 10) -> list[dict]:
        """
        从文档中提取关键概念（KeyBERT + MMR 多样性保证）。

        复用已有的 bge-large-zh-v1.5 embedding 模型，不额外加载模型。
        概念去重后存入 concepts 表，建立 document_concepts 关联。
        """
        from embedder import get_embedder

        # 获取文档正文
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content, title, domain FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                logger.warning(f"extract_concepts: 文档 {doc_id[:8]} 无内容")
                return []
            content = row[0]
            domain = row[2] or "law"

        # 文本预处理（取前 8000 字足矣）
        extract_text = content[:8000]

        # 概念提取：优先 jieba TF-IDF（轻量、离线、无模型加载开销）
        # 如 jieba 不可用，回退到 KeyBERT（需要 embedder 模型）
        try:
            keyphrases = self._extract_concepts_tfidf(extract_text, top_n)
            if not keyphrases:
                raise RuntimeError("jieba returned empty")
        except Exception as e:
            logger.warning(f"jieba 提取失败，尝试 KeyBERT: {e}")
            try:
                from keybert import KeyBERT
                embedder = get_embedder()
                kw_model = KeyBERT(model=embedder._model)
                keyphrases = kw_model.extract_keywords(
                    extract_text,
                    keyphrase_ngram_range=(1, 4),
                    stop_words=None,
                    top_n=top_n,
                    use_mmr=True,
                    diversity=0.7,
                )
            except ImportError:
                logger.warning("keybert 未安装")
                keyphrases = []
            except Exception as e2:
                logger.warning(f"KeyBERT 提取失败: {e2}")
                keyphrases = []

        if not keyphrases:
            return []

        # 批量检查已有概念 + 生成 embedding
        unique_phrases = []
        seen_norm = set()
        for phrase, score in keyphrases:
            normalized = self._normalize_concept(phrase)
            if not normalized or len(normalized) < 2 or normalized in seen_norm:
                continue
            seen_norm.add(normalized)
            unique_phrases.append((phrase, score))

        # 查询已存在的概念
        normalized_list = [self._normalize_concept(p[0]) for p in unique_phrases]
        existing_map = {}
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, normalized FROM concepts WHERE normalized = ANY(%s)",
                (normalized_list,),
            )
            for row in cur.fetchall():
                existing_map[row[1]] = row[0]

        # 生成概念 embedding（用于未来语义搜索）
        try:
            embedder = get_embedder()
        except Exception:
            embedder = None

        # 插入新概念 + 建立关联
        concepts_result = []
        with self.conn.cursor() as cur:
            for phrase, score in unique_phrases:
                normalized = self._normalize_concept(phrase)
                category = self._classify_concept(phrase)

                if normalized in existing_map:
                    concept_id = existing_map[normalized]
                    cur.execute(
                        "UPDATE concepts SET doc_count = doc_count + 1, updated_at = NOW() WHERE id = %s",
                        (concept_id,),
                    )
                else:
                    concept_id = str(uuid.uuid4())
                    concept_emb = None
                    if embedder:
                        try:
                            concept_emb = embedder.encode(phrase)
                        except Exception:
                            pass

                    cur.execute(
                        """INSERT INTO concepts (id, name, normalized, category, doc_count, embedding)
                        VALUES (%s, %s, %s, %s, 1, %s)""",
                        (concept_id, phrase, normalized, category, concept_emb),
                    )

                # 建立文档-概念关联
                cur.execute(
                    """INSERT INTO document_concepts (document_id, concept_id, relevance)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (document_id, concept_id) DO UPDATE
                    SET relevance = EXCLUDED.relevance""",
                    (doc_id, concept_id, round(float(score) * 10, 4)),
                )

                concepts_result.append({
                    "id": concept_id,
                    "name": phrase,
                    "category": category,
                    "relevance": round(float(score), 4),
                })

        self.conn.commit()
        logger.info(
            "概念抽取完成: doc=%s..., 抽取 %d 个概念 (新增 %d)",
            doc_id[:8], len(concepts_result),
            len(concepts_result) - len(existing_map),
        )
        return concepts_result

    def _extract_concepts_tfidf(self, text: str, top_n: int = 10) -> list[tuple]:
        """
        TF-IDF 关键词提取（KeyBERT 不可用时的回退方案）。
        使用 jieba 分词 + TF-IDF 权重，过滤学术通用停用词。
        """
        # 学术论文通用停用词（这些词太泛，不应作为"概念"）
        _STOP_WORDS = {
            '研究', '分析', '理论', '思想', '发展', '问题', '影响', '作用',
            '关系', '意义', '价值', '特征', '特点', '过程', '形成', '方面',
            '方式', '方法', '内容', '结构', '体系', '制度', '社会', '国家',
            '本文', '认为', '提出', '进行', '具有', '存在', '可以', '需要',
            '这一', '一个', '一种', '不同', '主要', '重要', '相关', '基本',
        }

        try:
            import jieba.analyse

            # 使用 jieba TF-IDF
            tags = jieba.analyse.extract_tags(text, topK=top_n * 3, withWeight=True)

            # 过滤单字和停用词
            result = [(word, weight) for word, weight in tags
                      if len(word) >= 2 and word not in _STOP_WORDS]

            # 补充 TextRank（提取短语，权重减半）
            try:
                tr_tags = jieba.analyse.textrank(text, topK=top_n * 2, withWeight=True)
                tr_filtered = [(w, wgt * 0.5) for w, wgt in tr_tags
                               if len(w) >= 2 and w not in _STOP_WORDS]
                result.extend(tr_filtered)
            except Exception:
                pass

            # 去重（按名称），取 top_n
            seen = set()
            deduped = []
            for word, weight in sorted(result, key=lambda x: x[1], reverse=True):
                if word not in seen and len(word) >= 2:
                    seen.add(word)
                    deduped.append((word, weight))
                if len(deduped) >= top_n:
                    break

            return deduped
        except ImportError:
            logger.warning("jieba 未安装，概念提取不可用")
            return []

    def search_concepts(self, query: str, limit: int = 20) -> list[dict]:
        """按名称搜索概念（支持模糊匹配）"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            search_pattern = f"%{query}%"
            cur.execute(
                """SELECT id, name, category, doc_count, created_at
                FROM concepts
                WHERE name ILIKE %s OR normalized ILIKE %s
                ORDER BY doc_count DESC, created_at DESC
                LIMIT %s""",
                (search_pattern, search_pattern, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "category": r["category"],
                    "doc_count": r["doc_count"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

    def get_concept(self, concept_id: str) -> Optional[dict]:
        """获取概念详情及其关联文档"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 概念基本信息
            cur.execute(
                """SELECT id, name, category, summary, doc_count, created_at, updated_at
                FROM concepts WHERE id = %s""",
                (concept_id,),
            )
            concept = cur.fetchone()
            if not concept:
                return None

            # 关联文档
            cur.execute(
                """SELECT d.id, d.title, d.author, d.domain, dc.relevance
                FROM document_concepts dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.concept_id = %s AND d.status = 'ready'
                ORDER BY dc.relevance DESC
                LIMIT 30""",
                (concept_id,),
            )
            docs = [
                {
                    "id": str(d["id"]),
                    "title": d["title"],
                    "author": d["author"],
                    "domain": d["domain"],
                    "relevance": float(d["relevance"]),
                }
                for d in cur.fetchall()
            ]

            return {
                "id": str(concept["id"]),
                "name": concept["name"],
                "category": concept["category"],
                "summary": concept["summary"],
                "doc_count": concept["doc_count"],
                "documents": docs,
                "created_at": concept["created_at"].isoformat() if concept["created_at"] else None,
                "updated_at": concept["updated_at"].isoformat() if concept["updated_at"] else None,
            }

    def list_concepts(
        self,
        category: Optional[str] = None,
        sort_by: str = "doc_count",
        limit: int = 50,
    ) -> list[dict]:
        """列出概念（可按分类筛选、按文档数/时间排序）"""
        valid_sorts = {"doc_count": "doc_count DESC", "recent": "created_at DESC", "name": "name ASC"}
        order = valid_sorts.get(sort_by, "doc_count DESC")

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if category:
                cur.execute(
                    f"""SELECT id, name, category, doc_count, created_at
                    FROM concepts WHERE category = %s
                    ORDER BY {order} LIMIT %s""",
                    (category, limit),
                )
            else:
                cur.execute(
                    f"""SELECT id, name, category, doc_count, created_at
                    FROM concepts ORDER BY {order} LIMIT %s""",
                    (limit,),
                )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "category": r["category"],
                    "doc_count": r["doc_count"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

    def get_related_concepts(self, concept_id: str, limit: int = 10) -> list[dict]:
        """
        获取相关概念：在同一文档中共现的概念（按共现次数排序）。
        这是"知识复利"的关键——概念之间的语义网络随文档增多自动增强。
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT c2.id, c2.name, c2.category, c2.doc_count,
                       COUNT(DISTINCT dc1.document_id) AS co_occurrence
                FROM document_concepts dc1
                JOIN document_concepts dc2 ON dc1.document_id = dc2.document_id
                    AND dc2.concept_id != %s
                JOIN concepts c2 ON dc2.concept_id = c2.id
                WHERE dc1.concept_id = %s
                GROUP BY c2.id, c2.name, c2.category, c2.doc_count
                ORDER BY co_occurrence DESC, c2.doc_count DESC
                LIMIT %s""",
                (concept_id, concept_id, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "category": r["category"],
                    "doc_count": r["doc_count"],
                    "co_occurrence": r["co_occurrence"],
                }
                for r in rows
            ]

    def get_concept_stats(self) -> dict:
        """获取概念统计信息"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total FROM concepts")
            total = cur.fetchone()["total"]

            cur.execute(
                """SELECT category, COUNT(*) AS cnt
                FROM concepts GROUP BY category ORDER BY cnt DESC"""
            )
            breakdown = {r["category"]: r["cnt"] for r in cur.fetchall()}

            cur.execute("SELECT COUNT(*) AS total FROM document_concepts")
            links = cur.fetchone()["total"]

            cur.execute(
                """SELECT name, doc_count FROM concepts
                ORDER BY doc_count DESC LIMIT 10"""
            )
            top_concepts = [
                {"name": r["name"], "doc_count": r["doc_count"]}
                for r in cur.fetchall()
            ]

        return {
            "total_concepts": total,
            "total_links": links,
            "breakdown": breakdown,
            "top_concepts": top_concepts,
        }

    def extract_concepts_for_existing(self, max_docs: int = 0) -> dict:
        """
        为已有文档批量抽取概念（用于首次迁移）。
        
        Args:
            max_docs: 最多处理多少篇文档（0=全部）
        
        Returns:
            {"processed": N, "total_concepts": N}
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id FROM documents WHERE status = 'ready'
                AND id NOT IN (SELECT DISTINCT document_id FROM document_concepts)
                ORDER BY created_at DESC"""
            )
            if max_docs > 0:
                rows = cur.fetchmany(max_docs)
            else:
                rows = cur.fetchall()

        total_concepts = 0
        for (doc_id,) in rows:
            try:
                result = self.extract_concepts(doc_id)
                total_concepts += len(result)
            except Exception as e:
                logger.error(f"概念抽取失败 doc={doc_id[:8]}: {e}")

        return {"processed": len(rows), "total_concepts": total_concepts}

    # ============================================================
    # P0: 操作审计日志 + 知识健康检查
    # ============================================================

    def log_operation(
        self,
        operation_type: str,
        entity_type: str = None,
        entity_id: str = None,
        entity_title: str = None,
        details: dict = None,
        operator: str = "system",
    ) -> int:
        """记录操作日志（append-only，不可修改）"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO kb_operations_log
                    (operation_type, entity_type, entity_id, entity_title, details, operator)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        operation_type,
                        entity_type,
                        entity_id,
                        entity_title,
                        json.dumps(details or {}, ensure_ascii=False),
                        operator,
                    ),
                )
                log_id = cur.fetchone()[0]
                self.conn.commit()
                return log_id
        except Exception as e:
            logger.warning(f"log_operation 失败（不影响主流程）: {e}")
            self.conn.rollback()
            return 0

    def get_operations_log(
        self,
        operation_type: str = None,
        entity_id: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """查询操作日志"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = []
            params = []
            if operation_type:
                conditions.append("operation_type = %s")
                params.append(operation_type)
            if entity_id:
                conditions.append("entity_id = %s")
                params.append(entity_id)
            where = " AND ".join(conditions) if conditions else "TRUE"
            params.extend([limit, offset])
            cur.execute(
                f"""SELECT id, operation_type, entity_type, entity_id, entity_title,
                          details, operator, created_at
                FROM kb_operations_log
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s""",
                params,
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "operation_type": r["operation_type"],
                    "entity_type": r["entity_type"],
                    "entity_id": str(r["entity_id"]) if r["entity_id"] else None,
                    "entity_title": r["entity_title"],
                    "details": r["details"],
                    "operator": r["operator"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

    def health_check(self) -> dict:
        """
        知识健康检查（纯 SQL，零 LLM）。

        检查 8 类问题：
        1. 未索引文档（有文档但无 chunks）
        2. 未提取概念的文档
        3. 无摘要文档
        4. 孤儿概念（仅出现在 1 篇文档中）
        5. 语义邻近但未关联（向量相似度高但无共同概念）
        6. 概念命名冲突（normalized 相同但 name 不同）
        7. 标签覆盖率（有多少文档没有任何标签）
        8. 重复/近似文档检测
        """
        report = {
            "overall_score": 100,
            "issues": [],
            "metrics": {},
            "recommendations": [],
        }

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 基础指标
            cur.execute("SELECT * FROM health_metrics")
            metrics = cur.fetchone()
            report["metrics"] = dict(metrics) if metrics else {}

            # 1. 未索引文档
            cur.execute("""
                SELECT d.id, d.title, d.domain, d.created_at
                FROM documents d
                WHERE d.status = 'ready'
                AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
                ORDER BY d.created_at DESC LIMIT 20
            """)
            unindexed = cur.fetchall()
            if unindexed:
                report["issues"].append({
                    "type": "unindexed_docs",
                    "severity": "high",
                    "count": metrics.get("unindexed_docs", len(unindexed)),
                    "items": [{"id": str(r["id"]), "title": r["title"], "domain": r["domain"]} for r in unindexed],
                    "message": f"{metrics.get('unindexed_docs', len(unindexed))} 篇文档未建立语义索引，无法被语义搜索命中",
                })
                report["overall_score"] -= 15

            # 2. 未提取概念的文档
            cur.execute("""
                SELECT d.id, d.title, d.domain
                FROM documents d
                WHERE d.status = 'ready'
                AND NOT EXISTS (SELECT 1 FROM document_concepts dc WHERE dc.document_id = d.id)
                ORDER BY d.created_at DESC LIMIT 20
            """)
            no_concepts = cur.fetchall()
            if no_concepts:
                report["issues"].append({
                    "type": "docs_without_concepts",
                    "severity": "medium",
                    "count": metrics.get("docs_without_concepts", len(no_concepts)),
                    "items": [{"id": str(r["id"]), "title": r["title"]} for r in no_concepts],
                    "message": f"{metrics.get('docs_without_concepts', len(no_concepts))} 篇文档未提取概念，无法参与知识复利",
                })
                report["overall_score"] -= 10

            # 3. 无摘要文档
            cur.execute("""
                SELECT d.id, d.title
                FROM documents d
                WHERE d.status = 'ready'
                AND (d.summary IS NULL OR d.summary = '')
                LIMIT 20
            """)
            no_summary = cur.fetchall()
            if no_summary:
                report["issues"].append({
                    "type": "docs_without_summary",
                    "severity": "low",
                    "count": metrics.get("docs_without_summary", len(no_summary)),
                    "items": [{"id": str(r["id"]), "title": r["title"]} for r in no_summary],
                    "message": f"{metrics.get('docs_without_summary', len(no_summary))} 篇文档无摘要",
                })
                report["overall_score"] -= 5

            # 4. 孤儿概念（doc_count = 1）
            cur.execute("""
                SELECT id, name, category
                FROM concepts WHERE doc_count = 1
                ORDER BY created_at DESC LIMIT 30
            """)
            orphans = cur.fetchall()
            if orphans:
                report["issues"].append({
                    "type": "orphan_concepts",
                    "severity": "low",
                    "count": metrics.get("orphan_concepts", len(orphans)),
                    "items": [{"id": str(r["id"]), "name": r["name"], "category": r["category"]} for r in orphans[:10]],
                    "message": f"{metrics.get('orphan_concepts', len(orphans))} 个概念仅出现在 1 篇文档中（孤儿概念），知识复利价值低",
                })
                report["overall_score"] -= 3

            # 5. 语义邻近但未关联
            cur.execute("""
                SELECT d1.id AS doc_a_id, d1.title AS doc_a_title,
                       d2.id AS doc_b_id, d2.title AS doc_b_title,
                       MAX(1.0 - (c1.embedding <=> c2.embedding)) AS similarity
                FROM chunks c1
                JOIN documents d1 ON c1.document_id = d1.id
                JOIN chunks c2 ON c2.document_id != c1.document_id
                JOIN documents d2 ON c2.document_id = d2.id
                WHERE d1.id < d2.id
                GROUP BY d1.id, d1.title, d2.id, d2.title
                HAVING MAX(1.0 - (c1.embedding <=> c2.embedding)) >= 0.75
                AND NOT EXISTS (
                    SELECT 1 FROM document_concepts dc1
                    JOIN document_concepts dc2 ON dc1.concept_id = dc2.concept_id
                    WHERE dc1.document_id = d1.id AND dc2.document_id = d2.id
                )
                ORDER BY similarity DESC LIMIT 10
            """)
            unlinked = cur.fetchall()
            if unlinked:
                report["issues"].append({
                    "type": "semantic_neighbors_unlinked",
                    "severity": "medium",
                    "count": len(unlinked),
                    "items": [
                        {
                            "doc_a": {"id": str(r["doc_a_id"]), "title": r["doc_a_title"]},
                            "doc_b": {"id": str(r["doc_b_id"]), "title": r["doc_b_title"]},
                            "similarity": round(float(r["similarity"]), 3),
                        }
                        for r in unlinked
                    ],
                    "message": f"{len(unlinked)} 对文档语义相似度 >0.75 但无共同概念，可能需要补充概念提取",
                })
                report["overall_score"] -= 8

            # 6. 概念命名冲突（normalized 相同但 name 不同）
            cur.execute("""
                SELECT normalized, array_agg(name) AS names, array_agg(id) AS ids
                FROM concepts
                GROUP BY normalized
                HAVING COUNT(DISTINCT name) > 1
                LIMIT 10
            """)
            conflicts = cur.fetchall()
            if conflicts:
                report["issues"].append({
                    "type": "concept_naming_conflict",
                    "severity": "medium",
                    "count": len(conflicts),
                    "items": [
                        {"normalized": r["normalized"], "names": r["names"], "ids": [str(i) for i in r["ids"]]}
                        for r in conflicts
                    ],
                    "message": f"{len(conflicts)} 组概念归一化后相同但显示名不同，可能导致重复",
                })
                report["overall_score"] -= 5

            # 7. 标签覆盖率
            cur.execute("""
                SELECT COUNT(*) FROM documents d
                WHERE d.status = 'ready'
                AND NOT EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id)
            """)
            untagged_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) FROM documents WHERE status = 'ready'")
            total_docs = cur.fetchone()["count"]
            if total_docs > 0 and untagged_count > 0:
                coverage = round((1 - untagged_count / total_docs) * 100, 1)
                report["issues"].append({
                    "type": "low_tag_coverage",
                    "severity": "low",
                    "count": untagged_count,
                    "coverage_percent": coverage,
                    "message": f"{untagged_count} 篇文档无任何标签（覆盖率 {coverage}%）",
                })
                if coverage < 80:
                    report["overall_score"] -= 5

            # 8. 重复/近似文档检测（标题相似度 > 80%）
            cur.execute("""
                SELECT d1.id AS id_a, d1.title AS title_a,
                       d2.id AS id_b, d2.title AS title_b,
                       similarity(d1.title, d2.title) AS title_sim
                FROM documents d1
                JOIN documents d2 ON d1.id < d2.id
                WHERE d1.status = 'ready' AND d2.status = 'ready'
                AND similarity(d1.title, d2.title) > 0.6
                ORDER BY title_sim DESC LIMIT 10
            """)
            duplicates = cur.fetchall()
            if duplicates:
                report["issues"].append({
                    "type": "potential_duplicates",
                    "severity": "medium",
                    "count": len(duplicates),
                    "items": [
                        {
                            "doc_a": {"id": str(r["id_a"]), "title": r["title_a"]},
                            "doc_b": {"id": str(r["id_b"]), "title": r["title_b"]},
                            "title_similarity": round(float(r["title_sim"]), 3),
                        }
                        for r in duplicates
                    ],
                    "message": f"{len(duplicates)} 对文档标题高度相似，可能存在重复入库",
                })
                report["overall_score"] -= 5

        # 生成建议
        report["overall_score"] = max(0, report["overall_score"])
        if metrics.get("unindexed_docs", 0) > 0:
            report["recommendations"].append("对未索引文档执行 kb_index 建立语义索引")
        if metrics.get("docs_without_concepts", 0) > 0:
            report["recommendations"].append("对未提取概念的文档调用 /api/concept/extract")
        if metrics.get("orphan_concepts", 0) > metrics.get("total_concepts", 1) * 0.5:
            report["recommendations"].append("孤儿概念过多，建议增加同主题文档入库以激活知识复利")
        if metrics.get("docs_without_summary", 0) > 0:
            report["recommendations"].append("对无摘要文档生成结构化摘要")

        # 记录健康检查操作
        self.log_operation(
            operation_type="health_check",
            entity_type="system",
            details={
                "score": report["overall_score"],
                "issue_count": len(report["issues"]),
            },
            operator="api",
        )

        return report

    # ============================================================
    # P1: 单文档结构化摘要
    # ============================================================

    def generate_summary(self, doc_id: str) -> dict:
        """
        为文档生成 7-module 结构化摘要。

        借鉴 NotebookLM 的低幻觉设计：
        - 所有内容严格基于原文，不做推断
        - 每个模块都标注来源位置
        - 未覆盖的内容明确标注"原文未涉及"

        7 个模块：
        1. core_argument — 核心论点（一句话）
        2. key_findings — 关键发现（3-5 条）
        3. methodology — 研究方法/论证路径
        4. key_concepts — 核心概念列表
        5. limitations — 局限性与不足
        6. connections — 与既有知识的关联
        7. practical_value — 实践价值
        """
        # 获取文档内容
        with self.conn.cursor() as cur:
            cur.execute("SELECT content, title, domain, doc_type FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            if not row or not row[0]:
                return {"error": "document not found or empty"}
            content = row[0]
            title = row[1]
            domain = row[2]
            doc_type = row[3]

        text = content[:6000]  # 取前 6000 字做摘要

        # 获取文档已有概念
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT c.name, c.category FROM concepts c
                JOIN document_concepts dc ON c.id = dc.concept_id
                WHERE dc.document_id = %s ORDER BY dc.relevance DESC LIMIT 10""",
                (doc_id,),
            )
            doc_concepts = [{"name": r[0], "category": r[1]} for r in cur.fetchall()]

        # 基于文本统计特征的结构化摘要（纯规则，零 LLM）
        # 1. 核心论点：取摘要字段或正文前 200 字
        with self.conn.cursor() as cur:
            cur.execute("SELECT summary FROM documents WHERE id = %s", (doc_id,))
            existing_summary = cur.fetchone()[0] or ""

        core_argument = existing_summary[:200] if existing_summary else text[:200].replace("\n", " ").strip()

        # 2. 关键发现：从概念和文本中提取
        key_findings = []
        # 取文档中最相关的概念作为关键发现
        for c in doc_concepts[:5]:
            key_findings.append(f"涉及{c['category']}：{c['name']}")

        # 从文本中提取带序号的要点
        import re as _re
        numbered = _re.findall(r'[（(][\d一二三四五六七八九十]+[)）]\s*([^\n]{10,80})', text[:3000])
        for item in numbered[:5]:
            if item not in key_findings:
                key_findings.append(item.strip())

        if not key_findings:
            # 取前 3 个段落的首句
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
            for p in paragraphs[:3]:
                first_sentence = _re.split(r'[。！？]', p)[0]
                if first_sentence and len(first_sentence) > 10:
                    key_findings.append(first_sentence.strip() + "。")

        # 3. 研究方法：检测方法论关键词
        method_keywords = {
            "实证研究": ["实证", "数据", "样本", "问卷", "统计", "回归", "量化"],
            "比较法研究": ["比较", "域外", "各国", "对比", "英美", "大陆法"],
            "案例分析": ["案例", "判决", "裁判", "法院", "案件"],
            "规范分析": ["规范", "应然", "价值", "理念", "原则"],
            "文献研究": ["文献", "综述", "学说", "理论"],
        }
        detected_methods = []
        for method, keywords in method_keywords.items():
            if any(kw in text[:3000] for kw in keywords):
                detected_methods.append(method)

        methodology = "、".join(detected_methods) if detected_methods else "原文未明确标注研究方法"

        # 4. 核心概念
        key_concepts = [c["name"] for c in doc_concepts[:8]]

        # 5. 局限性：检测局限性表述
        limitation_patterns = [
            r"不足之处[，,：:](.{10,100})",
            r"局限性[，,：:](.{10,100})",
            r"存在.*?问题[，,：:](.{10,80})",
            r"尚待.*?研究",
            r"未来.*?方向",
        ]
        limitations = "原文未明确讨论局限性"
        for pat in limitation_patterns:
            match = _re.search(pat, text)
            if match:
                limitations = match.group(0)[:200]
                break

        # 6. 与既有知识的关联：查找同域高相似度文档
        connections = ""
        try:
            cur.execute("""
                SELECT d2.title, MAX(1.0 - (c1.embedding <=> c2.embedding)) AS sim
                FROM chunks c1
                JOIN documents d1 ON c1.document_id = d1.id
                JOIN chunks c2 ON c2.document_id != d1.id
                JOIN documents d2 ON c2.document_id = d2.id
                WHERE d1.id = %s AND d2.status = 'ready' AND d2.domain = %s
                GROUP BY d2.title
                ORDER BY sim DESC LIMIT 3
            """, (doc_id, domain))
            related = cur.fetchall()
            if related:
                connections = "；".join([f"《{r[0]}》(相似度 {r[1]:.1%})" for r in related])
        except Exception:
            pass
        if not connections:
            connections = "未找到高相似度关联文档"

        # 7. 实践价值
        practical_keywords = {
            "立法建议": ["立法", "制度完善", "法律修改", "条文"],
            "司法适用": ["司法", "裁判", "适用", "审判"],
            "学术贡献": ["理论创新", "学说", "新视角", "新框架"],
            "社会实践": ["实践", "应用", "社会", "治理"],
        }
        practical_value = []
        for value_type, keywords in practical_keywords.items():
            if any(kw in text[:4000] for kw in keywords):
                practical_value.append(value_type)
        practical_value = "、".join(practical_value) if practical_value else "原文未明确讨论实践价值"

        # 组装完整 Markdown 摘要
        raw_summary = f"""## 文档摘要：{title}

### 核心论点
{core_argument}

### 关键发现
{chr(10).join(f'- {f}' for f in key_findings)}

### 研究方法
{methodology}

### 核心概念
{', '.join(key_concepts) if key_concepts else '无'}

### 局限性
{limitations}

### 知识关联
{connections}

### 实践价值
{practical_value}
"""

        # 存入数据库
        summary_data = {
            "core_argument": core_argument,
            "key_findings": key_findings,
            "methodology": methodology,
            "key_concepts": key_concepts,
            "limitations": limitations,
            "connections": connections,
            "practical_value": practical_value,
            "raw_summary": raw_summary,
            "model_used": "rule-based-v1",
        }

        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO document_summaries
                (document_id, summary_type, core_argument, key_findings, methodology,
                 key_concepts, limitations, connections, practical_value, raw_summary, model_used)
                VALUES (%s, 'structured', %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, summary_type) DO UPDATE
                SET core_argument = EXCLUDED.core_argument,
                    key_findings = EXCLUDED.key_findings,
                    methodology = EXCLUDED.methodology,
                    key_concepts = EXCLUDED.key_concepts,
                    limitations = EXCLUDED.limitations,
                    connections = EXCLUDED.connections,
                    practical_value = EXCLUDED.practical_value,
                    raw_summary = EXCLUDED.raw_summary,
                    model_used = EXCLUDED.model_used,
                    updated_at = NOW()
                RETURNING id""",
                (
                    doc_id,
                    summary_data["core_argument"],
                    summary_data["key_findings"],
                    summary_data["methodology"],
                    summary_data["key_concepts"],
                    summary_data["limitations"],
                    summary_data["connections"],
                    summary_data["practical_value"],
                    summary_data["raw_summary"],
                    summary_data["model_used"],
                ),
            )
            row = cur.fetchone()
            self.conn.commit()

        self.log_operation(
            operation_type="summary",
            entity_type="document",
            entity_id=doc_id,
            entity_title=title,
            details={"summary_type": "structured"},
            operator="api",
        )

        logger.info(f"结构化摘要已生成: doc={doc_id[:8]}..., title={title[:30]}")
        return {"id": str(row[0]), "document_id": doc_id, **summary_data}

    def get_summary(self, doc_id: str) -> Optional[dict]:
        """获取文档的结构化摘要"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM document_summaries WHERE document_id = %s AND summary_type = 'structured'""",
                (doc_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "document_id": str(row["document_id"]),
                "core_argument": row["core_argument"],
                "key_findings": row["key_findings"],
                "methodology": row["methodology"],
                "key_concepts": row["key_concepts"],
                "limitations": row["limitations"],
                "connections": row["connections"],
                "practical_value": row["practical_value"],
                "raw_summary": row["raw_summary"],
                "model_used": row["model_used"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }

    def get_concept_evidence(self, concept_id: str, limit: int = 20) -> list[dict]:
        """
        获取概念在各文档中的证据片段（概念证据积累视图）。

        从 concept_evidence 视图读取，返回每个文档中包含该概念的上下文片段。
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM concept_evidence WHERE concept_id = %s LIMIT %s""",
                (concept_id, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "document_id": str(r["document_id"]),
                    "document_title": r["document_title"],
                    "author": r["author"],
                    "domain": r["domain"],
                    "relevance": float(r["relevance"]),
                    "perspective": r["perspective"],
                    "evidence_snippet": r["evidence_snippet"],
                }
                for r in rows
            ]

    # ============================================================
    # P2: 跨文献知识图景 + 矛盾检测
    # ============================================================

    def detect_conflicts(self, max_pairs: int = 20) -> list[dict]:
        """
        矛盾检测：通过语义比较发现同一概念在不同文档中的冲突表述。

        策略：
        1. 找出 doc_count >= 2 的概念（跨文档概念）
        2. 对每对文档，提取包含该概念的上下文片段
        3. 用向量相似度比较上下文片段，相似度低但讨论同一概念 = 潜在矛盾
        4. 结果存入 concept_conflicts 表
        """
        conflicts_found = []

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 获取跨文档概念
            cur.execute("""
                SELECT c.id, c.name, c.category, c.doc_count
                FROM concepts c
                WHERE c.doc_count >= 2
                ORDER BY c.doc_count DESC
                LIMIT 50
            """)
            cross_doc_concepts = cur.fetchall()

            for concept in cross_doc_concepts[:30]:  # 限制处理量
                # 获取该概念在各文档中的证据片段
                cur.execute(
                    """SELECT * FROM concept_evidence WHERE concept_id = %s LIMIT 10""",
                    (concept["id"],),
                )
                evidences = cur.fetchall()

                if len(evidences) < 2:
                    continue

                # 对比每对证据片段
                for i in range(len(evidences)):
                    for j in range(i + 1, len(evidences)):
                        ev_a = evidences[i]
                        ev_b = evidences[j]

                        snippet_a = ev_a["evidence_snippet"] or ""
                        snippet_b = ev_b["evidence_snippet"] or ""

                        if len(snippet_a) < 20 or len(snippet_b) < 20:
                            continue

                        # 用概念 embedding 做相似度比较（简化版）
                        # 如果两段证据讨论同一概念但表述差异大，可能是矛盾
                        # 这里用简单的文本重叠度作为近似
                        words_a = set(snippet_a)
                        words_b = set(snippet_b)
                        if len(words_a) == 0 or len(words_b) == 0:
                            continue
                        overlap = len(words_a & words_b) / min(len(words_a), len(words_b))

                        # 重叠度低 = 可能矛盾
                        if overlap < 0.3:
                            conflict_id = str(uuid.uuid4())
                            cur.execute(
                                """INSERT INTO concept_conflicts
                                (id, concept_id, doc_a_id, doc_b_id, conflict_type,
                                 description, evidence_a, evidence_b, severity, detected_by)
                                VALUES (%s, %s, %s, %s, 'definition',
                                        %s, %s, %s, 'medium', 'semantic')
                                ON CONFLICT DO NOTHING""",
                                (
                                    conflict_id,
                                    concept["id"],
                                    ev_a["document_id"],
                                    ev_b["document_id"],
                                    f"概念「{concept['name']}」在两篇文档中的表述可能存在差异",
                                    snippet_a[:500],
                                    snippet_b[:500],
                                ),
                            )
                            conflicts_found.append({
                                "concept_id": str(concept["id"]),
                                "concept_name": concept["name"],
                                "doc_a": {"id": str(ev_a["document_id"]), "title": ev_a["document_title"]},
                                "doc_b": {"id": str(ev_b["document_id"]), "title": ev_b["document_title"]},
                                "evidence_a": snippet_a[:200],
                                "evidence_b": snippet_b[:200],
                                "overlap_score": round(overlap, 3),
                            })

                            if len(conflicts_found) >= max_pairs:
                                break
                    if len(conflicts_found) >= max_pairs:
                        break
                if len(conflicts_found) >= max_pairs:
                    break

            self.conn.commit()

        self.log_operation(
            operation_type="conflict_detect",
            entity_type="system",
            details={"conflicts_found": len(conflicts_found)},
            operator="api",
        )

        logger.info(f"矛盾检测完成：发现 {len(conflicts_found)} 个潜在矛盾")
        return conflicts_found

    def get_conflicts(self, resolved: bool = False, limit: int = 30) -> list[dict]:
        """获取矛盾列表"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT cf.*, c.name AS concept_name,
                          d1.title AS doc_a_title, d2.title AS doc_b_title
                FROM concept_conflicts cf
                LEFT JOIN concepts c ON cf.concept_id = c.id
                LEFT JOIN documents d1 ON cf.doc_a_id = d1.id
                LEFT JOIN documents d2 ON cf.doc_b_id = d2.id
                WHERE cf.resolved = %s
                ORDER BY cf.created_at DESC LIMIT %s""",
                (resolved, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "concept_id": str(r["concept_id"]) if r["concept_id"] else None,
                    "concept_name": r["concept_name"],
                    "doc_a": {"id": str(r["doc_a_id"]), "title": r["doc_a_title"]},
                    "doc_b": {"id": str(r["doc_b_id"]), "title": r["doc_b_title"]},
                    "conflict_type": r["conflict_type"],
                    "description": r["description"],
                    "evidence_a": r["evidence_a"],
                    "evidence_b": r["evidence_b"],
                    "severity": r["severity"],
                    "resolved": r["resolved"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

    def generate_knowledge_landscape(self, domain: str = None) -> dict:
        """
        跨文献知识图景报告（信息升维）。

        借鉴 NotebookLM+Codex 的"信息升维"设计：
        - 在单文档摘要之上做跨文档综合
        - 6 个模块的知识全景分析
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. 知识地图：概念分布与集群
            domain_filter = "WHERE d.domain = %s" if domain else ""
            params = [domain] if domain else []

            cur.execute(
                f"""SELECT c.id, c.name, c.category, c.doc_count,
                    COUNT(dc2.concept_id) AS related_count
                FROM concepts c
                JOIN document_concepts dc ON c.id = dc.concept_id
                JOIN documents d ON dc.document_id = d.id {domain_filter}
                LEFT JOIN document_concepts dc2 ON dc2.document_id = dc.document_id AND dc2.concept_id != c.id
                WHERE c.doc_count >= 1
                GROUP BY c.id, c.name, c.category, c.doc_count
                ORDER BY c.doc_count DESC
                LIMIT 50""",
                params,
            )
            concept_map = cur.fetchall()

            # 2. 核心概念集群（doc_count >= 3）
            core_clusters = [
                {"id": str(r["id"]), "name": r["name"], "category": r["category"], "doc_count": r["doc_count"],
                 "related_concepts": r["related_count"]}
                for r in concept_map if r["doc_count"] >= 3
            ]

            # 3. 知识空白：仅有 1 篇文档支撑的概念
            knowledge_gaps = [
                {"id": str(r["id"]), "name": r["name"], "category": r["category"]}
                for r in concept_map if r["doc_count"] == 1
            ]

            # 4. 学科分布
            cur.execute(
                f"""SELECT d.domain, d.doc_type, COUNT(*) AS cnt
                FROM documents d
                WHERE d.status = 'ready'
                {'AND d.domain = %s' if domain else ''}
                GROUP BY d.domain, d.doc_type
                ORDER BY cnt DESC""",
                params,
            )
            distribution = cur.fetchall()

            # 5. 概念共现网络（top pairs）
            cur.execute(
                f"""SELECT c1.name AS concept_a, c2.name AS concept_b,
                    COUNT(DISTINCT dc1.document_id) AS co_occurrence
                FROM document_concepts dc1
                JOIN document_concepts dc2 ON dc1.document_id = dc2.document_id AND dc1.concept_id < dc2.concept_id
                JOIN concepts c1 ON dc1.concept_id = c1.id
                JOIN concepts c2 ON dc2.concept_id = c2.id
                JOIN documents d ON dc1.document_id = d.id {domain_filter}
                WHERE d.status = 'ready'
                GROUP BY c1.name, c2.name
                ORDER BY co_occurrence DESC LIMIT 20""",
                params,
            )
            co_occurrences = cur.fetchall()

            # 6. 趋势分析：按年统计文档和概念增长
            cur.execute(
                f"""SELECT
                    COALESCE(EXTRACT(YEAR FROM d.published_at)::INT, EXTRACT(YEAR FROM d.created_at)::INT) AS year,
                    COUNT(DISTINCT d.id) AS docs,
                    COUNT(DISTINCT dc.concept_id) AS concepts
                FROM documents d
                LEFT JOIN document_concepts dc ON d.id = dc.document_id
                WHERE d.status = 'ready'
                {'AND d.domain = %s' if domain else ''}
                GROUP BY 1 ORDER BY year DESC LIMIT 10""",
                params,
            )
            trends = cur.fetchall()

        landscape = {
            "generated_at": datetime.now().isoformat(),
            "domain_filter": domain,
            "summary": {
                "total_concepts": len(concept_map),
                "core_clusters": len(core_clusters),
                "knowledge_gaps": len(knowledge_gaps),
                "co_occurrence_pairs": len(co_occurrences),
            },
            "knowledge_map": {
                "core_clusters": core_clusters[:15],
                "concept_distribution": {
                    r["category"]: r["doc_count"] for r in concept_map[:30]
                },
            },
            "knowledge_gaps": knowledge_gaps[:20],
            "discipline_distribution": [
                {"domain": r["domain"], "doc_type": r["doc_type"], "count": r["cnt"]}
                for r in distribution
            ],
            "concept_co_occurrences": [
                {"concept_a": r["concept_a"], "concept_b": r["concept_b"],
                 "co_occurrence": r["co_occurrence"]}
                for r in co_occurrences
            ],
            "growth_trends": [
                {"year": int(r["year"]) if r["year"] else None,
                 "docs": r["docs"], "concepts": r["concepts"]}
                for r in trends
            ],
            "insights": [],
        }

        # 自动生成洞察
        if core_clusters:
            top_concept = core_clusters[0]
            landscape["insights"].append(
                f"核心知识集群围绕「{top_concept['name']}」展开，覆盖 {top_concept['doc_count']} 篇文档"
            )
        if knowledge_gaps:
            landscape["insights"].append(
                f"发现 {len(knowledge_gaps)} 个知识空白点（仅 1 篇文档支撑），建议补充相关文献"
            )
        if co_occurrences:
            top_pair = co_occurrences[0]
            landscape["insights"].append(
                f"最强概念关联：「{top_pair['concept_a']}」与「{top_pair['concept_b']}」共现 {top_pair['co_occurrence']} 次"
            )

        self.log_operation(
            operation_type="landscape",
            entity_type="system",
            details={"domain": domain, "concepts_mapped": len(concept_map)},
            operator="api",
        )

        return landscape

    # ============================================================
    # P3: 对话记忆 + 多视角概念
    # ============================================================

    def record_query(
        self,
        session_id: str,
        query_text: str,
        query_type: str = "search",
        result_count: int = 0,
        result_doc_ids: list = None,
        operator: str = "mcp",
    ) -> str:
        """记录查询历史（对话记忆）"""
        query_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO query_history
                (id, session_id, query_text, query_type, result_count, result_doc_ids)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    query_id,
                    session_id,
                    query_text,
                    query_type,
                    result_count,
                    result_doc_ids or [],
                ),
            )
            self.conn.commit()
        return query_id

    def get_session_context(self, session_id: str, limit: int = 10) -> dict:
        """
        获取会话上下文（hot.md 等效）。

        返回该会话的查询历史 + 上下文摘要，
        供 AI 在后续查询时参考"之前聊过什么"。
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT id, query_text, query_type, result_count, result_doc_ids,
                          created_at, context_summary
                FROM query_history
                WHERE session_id = %s
                ORDER BY created_at DESC LIMIT %s""",
                (session_id, limit),
            )
            rows = cur.fetchall()

            if not rows:
                return {"session_id": session_id, "queries": [], "context_summary": ""}

            queries = [
                {
                    "query": r["query_text"],
                    "type": r["query_type"],
                    "result_count": r["result_count"],
                    "result_doc_ids": [str(d) for d in r["result_doc_ids"]] if r["result_doc_ids"] else [],
                    "timestamp": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]

            # 生成上下文摘要（纯规则版）
            topics = list(set(q["query"] for q in queries[:5]))
            context_summary = f"本会话已讨论 {len(rows)} 个问题，主要涉及：{', '.join(topics[:5])}"

            return {
                "session_id": session_id,
                "query_count": len(rows),
                "queries": queries,
                "context_summary": context_summary,
            }

    def extract_concepts_multiperspective(self, doc_id: str, perspectives: list = None) -> dict:
        """
        同源异构多视角概念提取。

        对同一文档从不同视角（如法学/社会学/经济学）提取概念，
        每个视角的概念标记 perspective 字段。

        借鉴 claude-obsidian 的"同源异构"设计：
        同一来源材料，不同视角的解读会产生不同的概念网络。
        """
        if perspectives is None:
            # 默认视角
            perspectives = ["legal", "social", "technical"]

        perspective_keywords = {
            "legal": ["法", "权", "义务", "责任", "规制", "合规", "法律", "条款", "立法", "司法"],
            "social": ["社会", "文化", "群体", "影响", "变迁", "结构", "关系", "行为", "观念"],
            "technical": ["技术", "算法", "数据", "模型", "系统", "架构", "实现", "方法", "工具"],
            "economic": ["经济", "市场", "成本", "效益", "产业", "商业模式", "竞争", "价格"],
            "ethical": ["伦理", "道德", "公平", "正义", "权利", "尊严", "自由", "善"],
        }

        all_results = {}

        for perspective in perspectives:
            keywords = perspective_keywords.get(perspective, [])
            if not keywords:
                continue

            # 获取文档内容
            with self.conn.cursor() as cur:
                cur.execute("SELECT content FROM documents WHERE id = %s", (doc_id,))
                row = cur.fetchone()
                if not row or not row[0]:
                    continue
                content = row[0][:8000]

            # 使用 jieba 提取关键词，然后按视角过滤
            try:
                import jieba.analyse
                all_keywords = jieba.analyse.extract_tags(content, topK=30, withWeight=True)
                # 过滤出属于当前视角的关键词
                perspective_concepts = [
                    (word, weight) for word, weight in all_keywords
                    if any(kw in word for kw in keywords) and len(word) >= 2
                ][:10]

                if not perspective_concepts:
                    all_results[perspective] = []
                    continue

                # 存入 document_concepts 表，标记 perspective
                for word, weight in perspective_concepts:
                    normalized = self._normalize_concept(word)
                    category = self._classify_concept(word)

                    # 查找或创建概念
                    cur.execute("SELECT id FROM concepts WHERE normalized = %s", (normalized,))
                    existing = cur.fetchone()
                    if existing:
                        concept_id = existing[0]
                        cur.execute(
                            "UPDATE concepts SET doc_count = doc_count + 1 WHERE id = %s",
                            (concept_id,),
                        )
                    else:
                        concept_id = str(uuid.uuid4())
                        cur.execute(
                            """INSERT INTO concepts (id, name, normalized, category, doc_count)
                            VALUES (%s, %s, %s, %s, 1)""",
                            (concept_id, word, normalized, category),
                        )

                    # 建立 perspective 关联
                    cur.execute(
                        """INSERT INTO document_concepts (document_id, concept_id, relevance, perspective)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id, concept_id) DO UPDATE
                        SET relevance = EXCLUDED.relevance, perspective = EXCLUDED.perspective""",
                        (doc_id, concept_id, round(float(weight) * 10, 4), perspective),
                    )

                all_results[perspective] = [
                    {"name": w, "relevance": round(float(wt), 4)}
                    for w, wt in perspective_concepts
                ]

            except ImportError:
                logger.warning("jieba 未安装，多视角提取不可用")
                all_results[perspective] = []

        self.conn.commit()

        self.log_operation(
            operation_type="extract_concepts",
            entity_type="document",
            entity_id=doc_id,
            details={"perspectives": perspectives, "total": sum(len(v) for v in all_results.values())},
            operator="api",
        )

        return {
            "doc_id": doc_id,
            "perspectives": all_results,
            "total_concepts": sum(len(v) for v in all_results.values()),
        }
