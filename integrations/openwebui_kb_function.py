"""
OpenWebUI Function — 知识库 RAG 检索（含文献引用）

在用户消息发送前，自动调用知识库 API 做语义检索，
将检索结果作为上下文注入 system prompt，强制要求引用来源。

安装方式：
1. 打开 OpenWebUI → 管理面板 → Functions → 创建新 Function
2. 将此文件内容粘贴进去
3. 保存并启用
4. 在模型设置中将此 Function 选为 Filter

配置参数：
- KB_API_URL: http://kb-mcp-server:8765
- TOP_K: 5
- MIN_SCORE: 0.3
"""

import json
import urllib.request


class Filter:
    def __init__(self):
        self.type = "filter"
        self.id = "kb_rag_filter"
        self.name = "知识库 RAG 检索"
        self.valves = self.Valves()

    class Valves:
        priority = 0
        KB_API_URL = "http://kb-mcp-server:8765"
        TOP_K = 5
        MIN_SCORE = 0.3
        ENABLED = True

    def _semantic_search(self, query, valves):
        url = "%s/api/semantic_search" % valves.KB_API_URL
        payload = json.dumps({"query": query, "limit": valves.TOP_K}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("results", [])
        except Exception as e:
            print("[KB RAG] Search error: %s" % e)
            return []

    def _format_context(self, results, valves):
        if not results:
            return ""

        parts = []
        sources_seen = set()
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            if score < valves.MIN_SCORE:
                continue
            title = r.get("title", "\u672a\u77e5\u6587\u6863")
            author = r.get("author", "")
            content = r.get("content", "")[:800]

            # 构建来源标识
            if author:
                source_ref = "\u300a%s\u300b\uff08%s\uff09" % (title, author)
            else:
                source_ref = "\u300a%s\u300b" % title

            sources_seen.add(source_ref)

            parts.append(
                "[%d] %s\n%s" % (i, source_ref, content)
            )

        # 汇总所有引用来源
        ref_list = "\n".join("- %s" % s for s in sorted(sources_seen))

        context_body = "\n\n".join(parts)

        return "%s\n\n>> \u53c2\u8003\u6587\u732e\u6e90\uff1a\n%s" % (context_body, ref_list)

    def inlet(self, body, __user__=None):
        if not self.valves.ENABLED:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message or len(user_message) < 5:
            return body

        results = self._semantic_search(user_message, self.valves)
        if not results:
            return body

        context = self._format_context(results, self.valves)
        print("[KB RAG] Injected %d results for: %s" % (len(results), user_message[:50]))

        suffix = (
            "\n\n"
            "== \u77e5\u8bc6\u5e93\u68c0\u7d22\u7ed3\u679f ==\n"
            "%s\n"
            "\u4ee5\u4e0a\u662f\u4ece\u77e5\u8bc6\u5e93\u4e2d\u68c0\u7d22\u5230\u7684\u76f8\u5173\u6587\u732e\u5185\u5bb9\u3002\n"
            "\u3010\u91cd\u8981\u89c4\u5219\u3011\u56de\u7b54\u65f6\u5fc5\u987b\uff1a\n"
            "1. \u57fa\u4e8e\u68c0\u7d22\u7ed3\u679c\u6765\u56de\u7b54\uff0c\u4e0d\u80fd\u80e1\u4e71\u7f16\u9020\n"
            "2. \u5728\u6bcf\u4e2a\u89c2\u70b9\u6216\u8bba\u8bc1\u540e\u7528\u300e\u300f\u6807\u6ce8\u51fa\u5904\uff0c\u683c\u5f0f\u4e3a\uff1a"
            "\u300e\u53c2\u8003\uff1a\u300a\u6587\u732e\u6807\u9898\u300b\uff08\u4f5c\u8005\uff09\u300f\n"
            "3. \u5982\u679c\u68c0\u7d22\u7ed3\u679c\u4e0e\u95ee\u9898\u65e0\u5173\uff0c\u8bf7\u660e\u786e\u8bf4\u660e\u5e76\u5ffd\u7565\n"
            "== \u68c0\u7d22\u7ed3\u679f\u7ed3\u675f =="
        ) % context

        # 查找现有 system 消息
        system_found = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + suffix
                system_found = True
                break

        if not system_found:
            messages.insert(0, {
                "role": "system",
                "content": "\u4f60\u662f\u4e00\u4e2a\u6cd5\u5b66\u77e5\u8bc6\u52a9\u624b\u3002" + suffix,
            })

        body["messages"] = messages
        return body
