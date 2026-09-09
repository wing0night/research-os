#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
week_file="$(date +%G-W%V).md"
target="$repo_root/weekly/$week_file"

if [[ -e "$target" ]]; then
  echo "已存在：weekly/$week_file"
  exit 0
fi

cp "$repo_root/weekly/_template.md" "$target"
sed -i '' "s/YYYY-Www/$(date +%G-W%V)/g; s/YYYY-MM-DD/$(date +%Y-%m-%d)/g" "$target"
echo "已创建：weekly/$week_file"
