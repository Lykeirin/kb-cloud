-- ============================================================
-- 概念系统迁移脚本 v1.1
-- 在已有数据库上添加 concepts 和 document_concepts 表
-- 运行方式：docker exec kb-postgres psql -U kb_admin -d knowledge_base -f /tmp/02-concepts-migration.sql
-- ============================================================

BEGIN;

-- 3b. 概念系统
CREATE TABLE IF NOT EXISTS concepts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    normalized      TEXT NOT NULL UNIQUE,
    category        TEXT,
    summary         TEXT,
    doc_count       INTEGER DEFAULT 0,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_concepts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    concept_id      UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relevance       REAL DEFAULT 1.0,
    context          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, concept_id)
);

-- 概念索引
CREATE INDEX IF NOT EXISTS idx_concepts_normalized ON concepts(normalized);
CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts(category);
CREATE INDEX IF NOT EXISTS idx_concepts_doc_count ON concepts(doc_count DESC);
CREATE INDEX IF NOT EXISTS idx_document_concepts_document ON document_concepts(document_id);
CREATE INDEX IF NOT EXISTS idx_document_concepts_concept ON document_concepts(concept_id);

COMMIT;
