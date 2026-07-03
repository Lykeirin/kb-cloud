-- ============================================================
-- 知识库 Schema v2.0 — 知识复利层
-- P0: 操作审计日志 + 知识健康检查
-- P1: 单文档结构化摘要
-- P2: 跨文献知识图景 + 矛盾检测
-- P3: 对话记忆 + 多视角概念
-- ============================================================

-- ============================================================
-- P0: 操作审计日志（append-only，不可修改）
-- ============================================================
CREATE TABLE IF NOT EXISTS kb_operations_log (
    id              BIGSERIAL PRIMARY KEY,
    operation_type  TEXT NOT NULL,              -- ingest / index / extract_concepts / health_check / summary / landscape / conflict_detect
    entity_type     TEXT,                        -- document / concept / chunk / system
    entity_id       UUID,                        -- 对应实体 ID
    entity_title    TEXT,                        -- 人类可读的实体名称
    details         JSONB DEFAULT '{}',          -- 操作详情（参数、结果摘要等）
    operator        TEXT DEFAULT 'system',       -- system / watcher / web / mcp / api
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oplog_type ON kb_operations_log(operation_type);
CREATE INDEX IF NOT EXISTS idx_oplog_entity ON kb_operations_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_oplog_created ON kb_operations_log(created_at DESC);

-- ============================================================
-- P1: 单文档结构化摘要
-- ============================================================
CREATE TABLE IF NOT EXISTS document_summaries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    summary_type    TEXT NOT NULL DEFAULT 'structured',  -- structured / brief / detailed
    -- 7-module 结构化摘要（借鉴 NotebookLM prompt 设计）
    core_argument   TEXT,                        -- 核心论点
    key_findings    TEXT[],                      -- 关键发现（数组）
    methodology     TEXT,                        -- 研究方法/论证路径
    key_concepts    TEXT[],                      -- 核心概念列表
    limitations     TEXT,                        -- 局限性与不足
    connections     TEXT,                        -- 与既有知识的关联
    practical_value TEXT,                        -- 实践价值/应用前景
    raw_summary     TEXT,                        -- 完整 Markdown 摘要
    model_used      TEXT,                        -- 生成模型标识
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, summary_type)
);

CREATE INDEX IF NOT EXISTS idx_summaries_doc ON document_summaries(document_id);

-- ============================================================
-- P2: 概念矛盾检测
-- ============================================================
CREATE TABLE IF NOT EXISTS concept_conflicts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    concept_id      UUID REFERENCES concepts(id) ON DELETE CASCADE,
    doc_a_id        UUID REFERENCES documents(id) ON DELETE CASCADE,
    doc_b_id        UUID REFERENCES documents(id) ON DELETE CASCADE,
    conflict_type   TEXT NOT NULL,               -- definition / methodology / conclusion / scope
    description     TEXT,                        -- 矛盾描述
    evidence_a      TEXT,                        -- 文档 A 中的证据片段
    evidence_b      TEXT,                        -- 文档 B 中的证据片段
    severity        TEXT DEFAULT 'medium',       -- low / medium / high
    resolved        BOOLEAN DEFAULT FALSE,
    resolution_note TEXT,
    detected_by     TEXT DEFAULT 'semantic',     -- semantic / manual
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conflicts_concept ON concept_conflicts(concept_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_docs ON concept_conflicts(doc_a_id, doc_b_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_unresolved ON concept_conflicts(resolved) WHERE resolved = FALSE;

-- ============================================================
-- P3: 对话记忆（查询历史）
-- ============================================================
CREATE TABLE IF NOT EXISTS query_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT NOT NULL,               -- 会话 ID（用于关联同一对话的多个查询）
    query_text      TEXT NOT NULL,               -- 用户查询
    query_type      TEXT,                         -- search / semantic / concept / landscape
    result_count    INTEGER DEFAULT 0,
    result_doc_ids  UUID[],                       -- 返回的文档 ID 列表
    feedback        TEXT,                         -- positive / negative / null
    context_summary TEXT,                         -- AI 生成的上下文摘要（hot.md 等效）
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_session ON query_history(session_id);
CREATE INDEX IF NOT EXISTS idx_query_created ON query_history(created_at DESC);

-- ============================================================
-- P3: 概念多视角（ALTER 已有表）
-- ============================================================
-- 为 document_concepts 添加 perspective 列
-- perspective 标识该概念是从哪个视角提取的（如：法学/社会学/经济学/技术）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_concepts' AND column_name = 'perspective'
    ) THEN
        ALTER TABLE document_concepts ADD COLUMN perspective TEXT DEFAULT 'default';
    END IF;
END $$;

-- 为 concepts 表添加 summary 列（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'concepts' AND column_name = 'summary'
    ) THEN
        ALTER TABLE concepts ADD COLUMN summary TEXT;
    END IF;
END $$;

-- ============================================================
-- 视图：概念证据积累（P1 用）
-- 每个概念在各文档中的出现情况 + 证据片段
-- ============================================================
CREATE OR REPLACE VIEW concept_evidence AS
SELECT
    c.id AS concept_id,
    c.name AS concept_name,
    c.category,
    c.doc_count,
    d.id AS document_id,
    d.title AS document_title,
    d.author,
    d.domain,
    dc.relevance,
    dc.perspective,
    -- 从文档前 2000 字中提取包含概念名的上下文片段作为证据
    CASE
        WHEN position(c.name in d.content) > 0 THEN
            substring(d.content from greatest(position(c.name in d.content) - 100, 1) for 300)
        ELSE NULL
    END AS evidence_snippet
FROM concepts c
JOIN document_concepts dc ON c.id = dc.concept_id
JOIN documents d ON dc.document_id = d.id
WHERE d.status = 'ready'
ORDER BY c.doc_count DESC, dc.relevance DESC;

-- ============================================================
-- 视图：知识健康指标（P0 用）
-- ============================================================
CREATE OR REPLACE VIEW health_metrics AS
SELECT
    (SELECT COUNT(*) FROM documents WHERE status = 'ready') AS total_docs,
    (SELECT COUNT(*) FROM chunks) AS total_chunks,
    (SELECT COUNT(*) FROM concepts) AS total_concepts,
    (SELECT COUNT(*) FROM document_concepts) AS total_concept_links,
    (SELECT COUNT(*) FROM concepts WHERE doc_count = 1) AS orphan_concepts,
    (SELECT COUNT(*) FROM concepts WHERE doc_count >= 3) AS cross_doc_concepts,
    (SELECT COUNT(*) FROM documents d
     WHERE d.status = 'ready'
     AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)) AS unindexed_docs,
    (SELECT COUNT(*) FROM documents d
     WHERE d.status = 'ready'
     AND NOT EXISTS (SELECT 1 FROM document_concepts dc WHERE dc.document_id = d.id)) AS docs_without_concepts,
    (SELECT COUNT(*) FROM documents d
     WHERE d.status = 'ready'
     AND (d.summary IS NULL OR d.summary = '')) AS docs_without_summary,
    (SELECT COUNT(*) FROM document_summaries) AS structured_summaries,
    (SELECT COUNT(*) FROM concept_conflicts WHERE resolved = FALSE) AS unresolved_conflicts;
