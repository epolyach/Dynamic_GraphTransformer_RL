with open('training_gpu/lib/critic_baseline.py', 'r') as f:
    lines = f.readlines()

# Find the line after baseline_config and add debug
for i, line in enumerate(lines):
    if 'baseline_config = config.get' in line:
        # Add debug after this line
        lines.insert(i+1, '        print(f"[DEBUG HybridBaseline] baseline_config: {baseline_config}")\n')
        lines.insert(i+2, '        sys.stdout.flush()\n')
        
        # Find next line that accesses config
        for j in range(i+3, min(i+20, len(lines))):
            if 'adv_config = config.get' in lines[j]:
                lines.insert(j, '        print("[DEBUG HybridBaseline] Getting adv_config")\n')
                lines.insert(j+1, '        sys.stdout.flush()\n')
            elif 'self.rollout_switch_epoch' in lines[j]:
                lines.insert(j, '        print("[DEBUG HybridBaseline] Setting rollout_switch_epoch")\n')
                lines.insert(j+1, '        sys.stdout.flush()\n')
            elif 'self.critic_switch_epoch' in lines[j]:
                lines.insert(j, '        print("[DEBUG HybridBaseline] Setting critic_switch_epoch")\n')
                lines.insert(j+1, '        sys.stdout.flush()\n')
            elif 'logger_print(' in lines[j] and 'Initialized' in lines[j]:
                lines.insert(j, '        print("[DEBUG HybridBaseline] About to call logger_print for initialization message")\n')
                lines.insert(j+1, '        sys.stdout.flush()\n')
                break
        break

with open('training_gpu/lib/critic_baseline.py', 'w') as f:
    f.writelines(lines)

print("✅ Added more detailed debug logging")
