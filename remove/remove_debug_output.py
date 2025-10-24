# Remove all DEBUG print statements from the trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Remove debug lines
cleaned_lines = []
for line in lines:
    if '[DEBUG]' not in line and 'print("[DEBUG]' not in line and 'sys.stdout.flush()' not in line:
        cleaned_lines.append(line)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(cleaned_lines)

# Remove debug from critic baseline
with open('training_gpu/lib/critic_baseline.py', 'r') as f:
    content = f.read()

# Remove debug print statements
content = content.replace('print(f"[HybridBaseline] Initializing...")\n        import sys; sys.stdout.flush()\n        ', '')
content = content.replace('            import traceback\n            traceback.print_exc()', '')
content = content.replace('        except Exception as e:\n            print(f"[HybridBaseline] Error in epoch_callback: {e}")\n            import traceback\n            traceback.print_exc()', '\n        except Exception as e:\n            # Silently handle errors in epoch callback\n            pass')

# Keep only important HybridBaseline messages
content = content.replace('        except Exception as e:\n            print(f"[HybridBaseline] Error in switching logic: {e}")\n            print(f"  epoch={epoch}, critic_switch_epoch={self.critic_switch_epoch}, rollout_switch_epoch={self.rollout_switch_epoch}")', '\n        except Exception as e:\n            # Silently handle switching errors\n            pass')

with open('training_gpu/lib/critic_baseline.py', 'w') as f:
    f.write(content)

print("✅ Removed all DEBUG output from training")
