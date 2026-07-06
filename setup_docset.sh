#!/bin/bash
# ============================================================
# SwingCue 文档集部署脚本
# 用法:把 docset.tar.gz 和本脚本放到项目根目录,然后 bash setup_docset.sh
# ============================================================
set -e
PROJECT_DIR="/home/jason/projects/swingcue-postest"
cd "$PROJECT_DIR"

echo "=== 1. 备份现有 docs 目录(以防万一)==="
if [ -d docs ]; then
  cp -r docs docs_backup_$(date +%Y%m%d_%H%M%S)
  echo "已备份现有 docs/"
fi

echo "=== 2. 解压新文档集到 docs_new/ ==="
mkdir -p docs_new
tar xzf docset.tar.gz -C docs_new
echo "解压完成"

echo "=== 3. 把现有旧文档归类到新结构 ==="
# 引擎规格类
mkdir -p docs_new/05_ENGINE_SPECS
for f in FAULT_VISUAL_STANDARDS.md JUDGMENT_CORE_SPEC.md SWING_PHASE_DETECTOR_SPEC.md \
         INDICATOR_ENGINEERING_SPEC.md API_AND_DATABASE_SPEC.md; do
  [ -f "docs/$f" ] && cp "docs/$f" docs_new/05_ENGINE_SPECS/ && echo "  → 05_ENGINE_SPECS/$f"
done
# GT 真值类
mkdir -p docs_new/06_GT_LABELS
[ -f docs/GT_LABELS.md ] && cp docs/GT_LABELS.md docs_new/06_GT_LABELS/ && echo "  → 06_GT_LABELS/GT_LABELS.md"
# 纪律/运维类
[ -f docs/HERMES_RUNBOOK.md ] && cp docs/HERMES_RUNBOOK.md docs_new/02_DISCIPLINE/ && echo "  → 02_DISCIPLINE/HERMES_RUNBOOK.md"
[ -f docs/RECORDING_GUIDE.md ] && cp docs/RECORDING_GUIDE.md docs_new/05_ENGINE_SPECS/ && echo "  → 05_ENGINE_SPECS/RECORDING_GUIDE.md"
# 旧执行计划归入会话/历史(可选保留)
[ -f docs/MVP0_EXECUTION_PLAN.md ] && cp docs/MVP0_EXECUTION_PLAN.md docs_new/07_SESSION_NOTES/ && echo "  → 07_SESSION_NOTES/MVP0_EXECUTION_PLAN.md(历史)"
# TASK_QUEUE 已废弃,标记后归档
[ -f docs/TASK_QUEUE.md ] && cp docs/TASK_QUEUE.md docs_new/07_SESSION_NOTES/TASK_QUEUE_DEPRECATED.md && echo "  → 07_SESSION_NOTES/TASK_QUEUE_DEPRECATED.md(已废弃)"

echo "=== 4. 用新结构替换 docs ==="
rm -rf docs_old_flat 2>/dev/null || true
mv docs docs_old_flat
mv docs_new docs
echo "docs 已替换为新结构(旧的平铺文档保留在 docs_old_flat/)"

echo "=== 5. 清理临时文件 ==="
rm -f docset.tar.gz

echo ""
echo "=== 完成!新文档集结构:==="
find docs -type f | sort

echo ""
echo "=== 下一步(手动执行 git 提交):==="
echo "  git add docs docs_old_flat .gitignore"
echo "  git commit -m 'docs: establish organized documentation set (blueprint v2.2 + discipline + stage logs)'"
echo "  git push"
