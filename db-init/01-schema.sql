-- ============================================================
-- 知识库 Schema v1.0
-- 单库双域设计：domain 字段分离法学(law)与创作(writing)
-- ============================================================

-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 模糊搜索加速
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. 文档主表
-- ============================================================
CREATE TYPE domain_type AS ENUM ('law', 'writing');
CREATE TYPE doc_status AS ENUM ('pending', 'processing', 'ready', 'failed');

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT NOT NULL,
    domain          domain_type NOT NULL,
    doc_type        TEXT NOT NULL,          -- law: paper/case/statute/note; writing: novel/chapter/character/worldbuilding
    source          TEXT,                    -- 来源：知网/北大法宝/微信读书/本地文件
    source_url      TEXT,                    -- 原始链接
    file_path       TEXT,                    -- 本地文件路径
    content         TEXT,                    -- 纯文本全文
    content_md      TEXT,                    -- Markdown 格式全文
    summary         TEXT,                    -- AI 生成的摘要
    metadata        JSONB DEFAULT '{}',      -- 域特定元数据（见下方说明）
    author          TEXT,
    published_at    DATE,
    word_count      INTEGER,
    char_count      INTEGER,
    status          doc_status DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- metadata JSONB 结构约定：
-- law.paper:   {"journal": "法学研究", "volume": "2021(4)", "keywords": [...], "citations": [...]}
-- law.case:    {"court": "北京互联网法院", "case_no": "(2022)京0491民初xxx", "parties": [...], "statutes": ["个人信息保护法第47条"]}
-- law.statute: {"law_name": "个人信息保护法", "article_no": "第47条", "chapter": "第四章"}
-- writing.novel:     {"novel_id": "uuid", "genre": "百合", "status": "连载中"}
-- writing.chapter:   {"novel_id": "uuid", "chapter_no": 43, "pov": "林晚晴", "word_count": 3500}
-- writing.character: {"novel_id": "uuid", "aliases": [...], "traits": [...], "first_appearance": "第1章"}

-- ============================================================
-- 2. 文本分块表（RAG 向量检索核心）
-- ============================================================
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,        -- 在原文档中的序号
    content         TEXT NOT NULL,            -- 分块文本
    token_count     INTEGER,                 -- token 数量
    embedding       vector(1024),            -- bge-large-zh-v1.5 输出 1024 维
    metadata        JSONB DEFAULT '{}',       -- {"section": "引言", "page": 3, ...}
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- ============================================================
-- 3. 标签系统
-- ============================================================
CREATE TABLE tags (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL UNIQUE,
    domain          domain_type NOT NULL,
    category        TEXT NOT NULL,            -- 标签分类（法学：部门法/争议焦点；创作：人物/场景/情绪）
    parent_id       UUID REFERENCES tags(id), -- 支持层级标签树
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_tags (
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id          UUID NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    confidence      REAL DEFAULT 1.0,         -- 自动打标的置信度
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (document_id, tag_id)
);

-- ============================================================
-- 3b. 概念系统（KeyBERT 自动抽取 + 知识复利）
-- ============================================================
CREATE TABLE concepts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,            -- 概念名称（如"物权行为无因性"）
    normalized      TEXT NOT NULL UNIQUE,      -- 归一化名称（去空格/标点，用于去重）
    category        TEXT,                     -- 概念类别：法学争议/学术概念/方法论/案例引用
    summary         TEXT,                     -- 所有来源文档中关于该概念的论述聚合
    doc_count       INTEGER DEFAULT 0,        -- 关联文档数（知识复利指标）
    embedding       vector(1024),             -- 概念名 embedding（用于概念语义搜索）
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_concepts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    concept_id      UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relevance       REAL DEFAULT 1.0,        -- 该概念在此文档中的相关度
    context         TEXT,                     -- 概念出现的原文上下文（约100字）
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, concept_id)
);

-- ============================================================
-- 4. 索引
-- ============================================================

-- 文档表索引
CREATE INDEX idx_documents_domain ON documents(domain);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created ON documents(created_at DESC);
CREATE INDEX idx_documents_title_trgm ON documents USING gin (title gin_trgm_ops);

-- 全文搜索索引（中文需要 zhparser 或简单 gin 索引）
CREATE INDEX idx_documents_content_fts ON documents USING gin (to_tsvector('simple', content));

-- 元数据 JSONB 索引（按 novel_id 等常用查询）
CREATE INDEX idx_documents_metadata ON documents USING gin (metadata);

-- 分块表索引
CREATE INDEX idx_chunks_document ON chunks(document_id);
-- IVFFlat 向量索引（数据量 > 1000 时创建）
-- CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 标签索引
CREATE INDEX idx_tags_domain ON tags(domain);
CREATE INDEX idx_tags_category ON tags(category);
CREATE INDEX idx_document_tags_tag ON document_tags(tag_id);

-- 概念索引
CREATE INDEX idx_concepts_normalized ON concepts(normalized);
CREATE INDEX idx_concepts_category ON concepts(category);
CREATE INDEX idx_concepts_doc_count ON concepts(doc_count DESC);
CREATE INDEX idx_document_concepts_document ON document_concepts(document_id);
CREATE INDEX idx_document_concepts_concept ON document_concepts(concept_id);

-- ============================================================
-- 5. 函数：混合检索（关键词 + 向量）
-- ============================================================
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text      TEXT,
    query_embedding vector(1024),
    target_domain   domain_type DEFAULT NULL,
    target_type     TEXT DEFAULT NULL,
    match_limit     INTEGER DEFAULT 10,
    keyword_weight  REAL DEFAULT 0.3,
    vector_weight   REAL DEFAULT 0.7
)
RETURNS TABLE (
    chunk_id        UUID,
    document_id     UUID,
    title           TEXT,
    domain          domain_type,
    doc_type        TEXT,
    content         TEXT,
    score           REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        d.id,
        d.title,
        d.domain,
        d.doc_type,
        c.content,
        (
            keyword_weight * ts_rank(
                to_tsvector('simple', c.content),
                plainto_tsquery('simple', query_text)
            ) +
            vector_weight * (1.0 - (c.embedding <=> query_embedding))
        ) AS score
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.status = 'ready'
      AND (target_domain IS NULL OR d.domain = target_domain)
      AND (target_type IS NULL OR d.doc_type = target_type)
    ORDER BY score DESC
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 6. 预置标签：法学域
-- ============================================================
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('宪法学', 'law', '部门法', NULL),
    ('民法学', 'law', '部门法', NULL),
    ('刑法学', 'law', '部门法', NULL),
    ('行政法学', 'law', '部门法', NULL),
    ('国际法学', 'law', '部门法', NULL),
    ('知识产权法', 'law', '部门法', NULL),
    ('数据法学', 'law', '部门法', NULL),
    ('个人信息保护法', 'law', '部门法', NULL),
    ('法社会学', 'law', '部门法', NULL),
    ('外国法制史', 'law', '部门法', NULL);

-- 二级标签示例
WITH constitutional AS (SELECT id FROM tags WHERE name = '宪法学')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('基本权利', 'law', '争议焦点', (SELECT id FROM constitutional)),
    ('合宪性审查', 'law', '争议焦点', (SELECT id FROM constitutional)),
    ('国家机构', 'law', '争议焦点', (SELECT id FROM constitutional));

WITH civil AS (SELECT id FROM tags WHERE name = '民法学')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('合同纠纷', 'law', '争议焦点', (SELECT id FROM civil)),
    ('侵权责任', 'law', '争议焦点', (SELECT id FROM civil)),
    ('物权保护', 'law', '争议焦点', (SELECT id FROM civil));

WITH ip AS (SELECT id FROM tags WHERE name = '知识产权法')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('著作权', 'law', '争议焦点', (SELECT id FROM ip)),
    ('AI生成物', 'law', '争议焦点', (SELECT id FROM ip)),
    ('专利权', 'law', '争议焦点', (SELECT id FROM ip)),
    ('商标权', 'law', '争议焦点', (SELECT id FROM ip));

WITH data AS (SELECT id FROM tags WHERE name = '数据法学')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('个人信息保护', 'law', '争议焦点', (SELECT id FROM data)),
    ('数据确权', 'law', '争议焦点', (SELECT id FROM data)),
    ('算法治理', 'law', '争议焦点', (SELECT id FROM data)),
    ('删除权', 'law', '争议焦点', (SELECT id FROM data));

-- ============================================================
-- 7. 预置标签：创作域
-- ============================================================
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('人物', 'writing', '元类型', NULL),
    ('场景', 'writing', '元类型', NULL),
    ('情节', 'writing', '元类型', NULL),
    ('设定', 'writing', '元类型', NULL),
    ('情绪', 'writing', '元类型', NULL);

WITH character AS (SELECT id FROM tags WHERE name = '人物')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('主角', 'writing', '角色', (SELECT id FROM character)),
    ('配角', 'writing', '角色', (SELECT id FROM character)),
    ('反派', 'writing', '角色', (SELECT id FROM character));

WITH emotion AS (SELECT id FROM tags WHERE name = '情绪')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('压抑', 'writing', '情绪基调', (SELECT id FROM emotion)),
    ('爆发', 'writing', '情绪基调', (SELECT id FROM emotion)),
    ('温情', 'writing', '情绪基调', (SELECT id FROM emotion)),
    ('紧张', 'writing', '情绪基调', (SELECT id FROM emotion)),
    ('悲伤', 'writing', '情绪基调', (SELECT id FROM emotion));

WITH scene AS (SELECT id FROM tags WHERE name = '场景')
INSERT INTO tags (name, domain, category, parent_id) VALUES
    ('法庭', 'writing', '场景类型', (SELECT id FROM scene)),
    ('办公室', 'writing', '场景类型', (SELECT id FROM scene)),
    ('雨夜', 'writing', '场景类型', (SELECT id FROM scene)),
    ('家', 'writing', '场景类型', (SELECT id FROM scene));
