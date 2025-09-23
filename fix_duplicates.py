#!/usr/bin/env python3
"""Fix duplicate code blocks in advanced_trainer_gpu.py"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Find the duplicate blocks
# They start at line 418 and repeat 8 more times (9 total occurrences)
# Each block is approximately 10 lines

# Keep everything up to line 417 (before the second duplicate)
fixed_lines = lines[:418]  # Up to line 418 (0-indexed is 417)

# Skip the duplicate blocks (lines 418-497 approximately)
# Resume from where the duplicates end
skip_until = 498  # This is where the unique code resumes

# Add the rest of the file
fixed_lines.extend(lines[skip_until:])

# Write the fixed file
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed duplicate code blocks in advanced_trainer_gpu.py")
print(f"Original file: {len(lines)} lines")
print(f"Fixed file: {len(fixed_lines)} lines")
print(f"Removed: {len(lines) - len(fixed_lines)} duplicate lines")
