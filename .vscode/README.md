# VSCode Remote Debugging Setup

This directory contains VSCode configuration for debugging the PyTorch training scripts on remote GPU nodes.

## Files

- **`launch.json`** - Debug configuration for GT+RL training with tiny_1.yaml
- **`settings.json`** - Python interpreter and analysis settings
- **`tasks.json`** - Common tasks (activate environment, run training, check GPU)

## Setup for Remote GPU Node

1. **Copy environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Update Python interpreter path** in `settings.json` if needed:
   ```json
   "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
   ```

3. **Ensure virtual environment is activated:**
   ```bash
   source activate_env.sh
   ```

## Usage

### Debugging
1. Open project in VSCode
2. Go to Run and Debug (Ctrl+Shift+D)
3. Select "Debug GT+RL Training (tiny_1)"
4. Set breakpoints and press F5

### Available CLI Arguments
You can modify `args` in `launch.json`:
- `--model GT+RL` - Model type
- `--config configs/tiny_1.yaml` - Config file
- `--epochs 3` - Number of epochs
- `--verbose` - Verbose output
- `--mixed_precision` - Enable mixed precision
- `--device cuda:0` - GPU device

### Quick Commands
Use Command Palette (Ctrl+Shift+P) > "Tasks: Run Task":
- **"Run GT+RL Training (CLI)"** - Run without debugging
- **"Check GPU Status"** - Verify CUDA availability

## Remote Development

For remote GPU nodes, use VSCode Remote-SSH extension:
1. Install "Remote - SSH" extension
2. Connect to GPU node
3. Open this project directory
4. The debug configurations will work automatically
