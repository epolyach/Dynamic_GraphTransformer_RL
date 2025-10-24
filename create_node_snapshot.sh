#!/bin/bash
# Run this script on each node (GPU1, GPU3, local) to create snapshot

set -e

NODE_NAME="${1:-unknown}"
REPO_DIR="$HOME/CVRP/Dynamic_GraphTransformer_RL"

echo "Creating snapshot for: $NODE_NAME"
echo "Repository: $REPO_DIR"

cd "$REPO_DIR"

# Fetch latest from origin
echo "Fetching from origin..."
git fetch origin

# Create snapshot branch
echo "Creating ${NODE_NAME}-snapshot branch..."
git checkout -b "${NODE_NAME}-snapshot"

# Create organization folders
echo "Creating organization folders..."
mkdir -p utils archive/docs archive/configs

# Move utility scripts to utils/
echo "Organizing utility scripts..."
for script in *.sh *.py; do
    if [ -f "$script" ] && [ "$script" != "setup_venv.sh" ]; then
        # Keep some scripts, move others
        case "$script" in
            activate_env.sh|setup_venv.sh|generate_results_index.py)
                [ ! -d utils ] || mv "$script" utils/ 2>/dev/null || true
                ;;
        esac
    fi
done

# Move old docs to archive
echo "Archiving old documentation..."
mv CLEANUP*.md CONFIG_*.md CPU_*.md CURRENT_*.md GPU_*.md PARALLEL_*.md PREFETCH*.md PROJECT_*.md TRAINING_*.md README*.backup* archive/docs/ 2>/dev/null || true

# Move config backups
echo "Archiving config backups..."
mv configs/*.backup* configs/*before* archive/configs/ 2>/dev/null || true
mv src/utils/*.backup src/models/*.backup* archive/configs/ 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
rm -f *.pid .gitignore.backup

# Stage all changes
echo "Staging changes..."
git add -A

# Commit
echo "Committing..."
git commit -m "${NODE_NAME} snapshot: preserve results and state"

# Push to origin
echo "Pushing to origin..."
git push origin "${NODE_NAME}-snapshot"

echo ""
echo "✅ ${NODE_NAME}-snapshot created and pushed successfully!"
echo "Branch: ${NODE_NAME}-snapshot"
