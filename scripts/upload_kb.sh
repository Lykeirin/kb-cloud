#!/bin/bash
# ============================================================
# 知识库文档上传脚本
# 支持: PDF, TXT, Markdown, DOCX, EPUB, HTML, PPTX, 图片(OCR), 压缩包
# 用法: ./upload_kb.sh file1.pdf [file2.docx ...]
#       ./upload_kb.sh *.pdf
#       ./upload_kb.sh *.docx *.txt
#
# 环境变量:
#   KB_REMOTE_HOST  远程服务器地址 (默认: localhost)
#   KB_REMOTE_USER  SSH 用户 (默认: root)
#   KB_REMOTE_PORT  SSH 端口 (默认: 22)
#   KB_REMOTE_DIR   远程目录 (默认: ~/kb-inbox)
# ============================================================

REMOTE_HOST="${KB_REMOTE_HOST:-localhost}"
REMOTE_USER="${KB_REMOTE_USER:-root}"
REMOTE_DIR="${KB_REMOTE_DIR:-~/kb-inbox}"
REMOTE_PORT="${KB_REMOTE_PORT:-22}"

SUPPORTED_EXT="pdf txt md markdown docx epub html htm pptx jpg jpeg png bmp tiff webp zip tar.gz tgz tar.bz2 tar.xz"

if [ $# -eq 0 ]; then
    echo "用法: $0 <文件1> [文件2 ...]"
    echo "支持格式: $SUPPORTED_EXT"
    echo "示例: $0 论文.pdf"
    echo "      $0 *.pdf"
    echo "      $0 笔记.txt 报告.docx"
    exit 1
fi

SUCCESS=0
FAILED=0
SKIPPED=0

for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "  [跳过] 文件不存在: $file"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    basename=$(basename "$file")
    ext="${file##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    # 检查格式
    if ! echo "$SUPPORTED_EXT" | grep -qw "$ext_lower"; then
        echo "  [跳过] 不支持格式: $basename (.$ext)"
        echo "         支持格式: $SUPPORTED_EXT"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "  上传中: $basename ... "

    scp -o StrictHostKeyChecking=no -P "$REMOTE_PORT" "$file" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "完成"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "失败"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "  成功: $SUCCESS | 失败: $FAILED | 跳过: $SKIPPED"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "  文件已上传到 ${REMOTE_DIR}/"
    echo "  watcher 将在 ~15 秒内自动处理（解析+向量化+入库）"
    echo ""
    echo "  查看处理日志:"
    echo "    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} 'docker logs kb-watcher --tail=30'"
fi
