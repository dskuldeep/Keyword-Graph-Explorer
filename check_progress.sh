#!/bin/bash

echo "=== Crawler Progress ==="
echo ""

# Check if process is running
if ps aux | grep "run_crawler" | grep -v grep > /dev/null; then
    echo "✅ Crawler is RUNNING"
else
    echo "❌ Crawler is NOT running"
fi

echo ""
echo "=== Last 30 lines of output ==="
tail -30 crawler_output.log

echo ""
echo "=== Completed blogs ==="
for dir in output/*/; do
    if [ -f "${dir}graph.html" ]; then
        echo "✅ $(basename $dir)"
    else
        echo "⏳ $(basename $dir) (in progress or not started)"
    fi
done
