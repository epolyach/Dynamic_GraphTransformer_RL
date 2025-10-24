# File Creation Helper for Warp AI

## Problem
Warp AI cannot use multiline heredoc (`cat <<`, `python <<`) - these commands wait for stdin input and hang.

## Solution: Use printf

### Create a new file:
```bash
printf "%s\n" "line 1" "line 2" "line 3" > file.txt
```

### Append to file:
```bash
printf "%s\n" "line 4" "line 5" >> file.txt
```

## Examples

### Create a Python script:
```bash
printf "%s\n" "#!/usr/bin/env python3" "import sys" "" "print(\"Hello World\")" > script.py
chmod +x script.py
```

### Create a markdown file:
```bash
printf "%s\n" "# Title" "" "Some content" "- Bullet 1" "- Bullet 2" > document.md
```

### Create multi-section file:
```bash
# Section 1
printf "%s\n" "# README" "" "## Introduction" > README.md

# Append Section 2
printf "%s\n" "" "## Installation" "Run: pip install ..." >> README.md

# Append Section 3
printf "%s\n" "" "## Usage" "Run: python script.py" >> README.md
```

## Tips

1. Use single quotes to avoid escaping issues
2. Use empty string "" for blank lines
3. Use >> to append, > to overwrite
4. For backticks in content, escape them: \`code\`

## What NOT to do

```bash
# ❌ These HANG in Warp:
cat > file.txt << EOF
python << ENDPYTHON
cat << "HEREDOC"
```
