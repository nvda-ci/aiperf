#!/bin/bash
# Script to collect all Chrome dependencies recursively

chrome=/opt/aiperf/venv/lib/python3.13/site-packages/choreographer/cli/browser_exe/chrome-linux64/chrome

# Function to get all dependencies recursively
get_all_deps() {
    local binary=$1
    local seen_file=$2
    
    ldd "$binary" 2>/dev/null | grep "=>" | awk '{print $3}' | while read lib; do
        if [ -f "$lib" ] && ! grep -q "^$lib$" "$seen_file"; then
            echo "$lib" >> "$seen_file"
            get_all_deps "$lib" "$seen_file"
        fi
    done
}

# Collect all libraries
tmpfile=$(mktemp)
get_all_deps "$chrome" "$tmpfile"

# Output sorted unique list
sort "$tmpfile" | uniq
rm "$tmpfile"

