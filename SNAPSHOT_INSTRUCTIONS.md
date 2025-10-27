# Instructions for Creating Snapshots on Other Nodes

## Quick Method: Use the automated script

### On GPU1:
```bash
ssh evgeny.polyachenko@gpu1.sedan.pro
cd ~/CVRP/Dynamic_GraphTransformer_RL
git fetch origin
git checkout gpu2-snapshot
git pull origin gpu2-snapshot
./create_node_snapshot.sh gpu1
```

### On GPU3:
```bash
ssh evgeny.polyachenko@gpu3.sedan.pro
cd ~/CVRP/Dynamic_GraphTransformer_RL
git fetch origin
git checkout gpu2-snapshot
git pull origin gpu2-snapshot
./create_node_snapshot.sh gpu3
```

### On Local Machine:
```bash
cd ~/CVRP/Dynamic_GraphTransformer_RL
git fetch origin
git checkout gpu2-snapshot
git pull origin gpu2-snapshot
./create_node_snapshot.sh local
```

---

## Manual Method: Step by step

### On each node, run:

```bash
cd ~/CVRP/Dynamic_GraphTransformer_RL
git fetch origin
git checkout -b <NODE_NAME>-snapshot  # e.g., gpu1-snapshot
mkdir -p utils archive/docs archive/configs

# Organize files (move scripts, docs, configs)
# Similar to what was done on GPU2

git add -A
git commit -m "<NODE_NAME> snapshot: preserve results and state"
git push origin <NODE_NAME>-snapshot
```

---

## After all snapshots are pushed

1. Go to GitHub and verify all snapshot branches exist
2. Merge main-clean to main:
   ```bash
   cd ~/CVRP/Dynamic_GraphTransformer_RL
   git checkout main
   git merge main-clean
   git push origin main
   ```

3. Update all nodes:
   ```bash
   # On each node
   git checkout main
   git pull origin main
   ```
