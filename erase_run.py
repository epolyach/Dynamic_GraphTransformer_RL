#!/usr/bin/env python3
"""
Results Folder Cleanup Script

This script erases the contents of results/XXX/* folders created by run_train_validation.py
while preserving the folder structure. It determines the scale (small/medium/production)
from the configuration file and cleans the corresponding results folder.

Usage:
    python erase_run.py --config configs/small.yaml
    python erase_run.py --config configs/medium.yaml  
    python erase_run.py --config configs/production.yaml
    python erase_run.py --scale small
    python erase_run.py --scale medium --force
    python erase_run.py --all --force
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
from typing import List, Optional


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_scale_from_config(config_path: str) -> str:
    """Extract scale from config filename"""
    config_filename = Path(config_path).stem  # 'small', 'medium', 'production'
    if config_filename in ['small', 'medium', 'production']:
        return config_filename
    else:
        # Fallback: try to infer from config content
        try:
            config = load_config(config_path)
            num_customers = config.get('problem', {}).get('num_customers', 15)
            if num_customers <= 20:
                return 'small'
            elif num_customers <= 50:
                return 'medium'
            else:
                return 'production'
        except Exception:
            return 'custom'


def get_files_to_remove(results_dir: str) -> List[str]:
    """Get list of files to remove (excluding directories)"""
    files_to_remove = []
    
    if not os.path.exists(results_dir):
        return files_to_remove
    
    # Walk through all subdirectories and collect files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            files_to_remove.append(file_path)
    
    return files_to_remove


def count_directory_structure(results_dir: str) -> int:
    """Count number of directories in the structure"""
    if not os.path.exists(results_dir):
        return 0
    
    dir_count = 0
    for root, dirs, files in os.walk(results_dir):
        dir_count += len(dirs)
    
    return dir_count


def remove_files_preserve_structure(results_dir: str, dry_run: bool = False) -> tuple:
    """
    Remove all files from results directory while preserving folder structure
    
    Returns:
        (files_removed_count, directories_preserved_count)
    """
    if not os.path.exists(results_dir):
        print(f"   ⚠️  Directory does not exist: {results_dir}")
        return 0, 0
    
    files_to_remove = get_files_to_remove(results_dir)
    files_removed = 0
    
    # Remove files
    for file_path in files_to_remove:
        try:
            if dry_run:
                print(f"   [DRY RUN] Would remove: {file_path}")
            else:
                os.remove(file_path)
                files_removed += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {file_path}: {e}")
    
    # Count preserved directories
    dirs_preserved = count_directory_structure(results_dir)
    
    return files_removed, dirs_preserved


def clean_empty_subdirectories(results_dir: str, preserve_main_structure: bool = True) -> int:
    """
    Remove empty subdirectories, but preserve main structure if requested
    
    Main structure directories to preserve:
    - analysis/
    - checkpoints/ 
    - csv/
    - logs/
    - plots/
    - pytorch/
    - test_instances/
    - legacy/
    """
    if not os.path.exists(results_dir):
        return 0
    
    main_structure_dirs = {
        'analysis', 'checkpoints', 'csv', 'logs', 'plots', 
        'pytorch', 'test_instances', 'legacy'
    }
    
    removed_dirs = 0
    
    # Walk bottom-up to remove empty directories
    for root, dirs, files in os.walk(results_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            # Skip main structure directories if preservation is enabled
            if preserve_main_structure:
                relative_path = os.path.relpath(dir_path, results_dir)
                if '/' not in relative_path and relative_path in main_structure_dirs:
                    continue
            
            # Remove if empty
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    removed_dirs += 1
                    print(f"   🗑️  Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"   ⚠️  Could not remove directory {dir_path}: {e}")
    
    return removed_dirs


def erase_results_folder(scale: str, dry_run: bool = False, clean_empty_dirs: bool = True) -> bool:
    """
    Erase contents of results folder for given scale
    
    Returns:
        True if successful, False otherwise
    """
    results_dir = f"results/{scale}"
    
    print(f"🎯 Target: {results_dir}/")
    
    if not os.path.exists(results_dir):
        print(f"   ℹ️  No results directory found for scale '{scale}'")
        return True
    
    # Show what would be affected
    files_to_remove = get_files_to_remove(results_dir)
    dirs_count = count_directory_structure(results_dir)
    
    print(f"   📁 Directory structure: {dirs_count} subdirectories")
    print(f"   📄 Files to remove: {len(files_to_remove)}")
    
    if len(files_to_remove) == 0:
        print(f"   ✅ Already clean!")
        return True
    
    if dry_run:
        print(f"   🔍 DRY RUN - No files will be actually removed")
    
    # Remove files while preserving structure
    files_removed, dirs_preserved = remove_files_preserve_structure(results_dir, dry_run)
    
    if not dry_run:
        print(f"   🗑️  Removed: {files_removed} files")
        print(f"   📁 Preserved: {dirs_preserved} directories")
        
        # Optionally clean empty subdirectories (but preserve main structure)
        if clean_empty_dirs:
            removed_dirs = clean_empty_subdirectories(results_dir, preserve_main_structure=True)
            if removed_dirs > 0:
                print(f"   🧹 Cleaned: {removed_dirs} empty subdirectories")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Erase results folder contents while preserving directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python erase_run.py --config configs/small.yaml
    python erase_run.py --scale medium --dry-run  
    python erase_run.py --all --force
    python erase_run.py --scale small --no-clean-empty
        """
    )
    
    # Main options
    parser.add_argument('--config', type=str, help='Path to configuration file (determines scale)')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'production'], 
                       help='Directly specify scale to clean')
    parser.add_argument('--all', action='store_true', help='Clean all scales (small, medium, production)')
    
    # Control options
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--no-clean-empty', action='store_true', help='Do not remove empty subdirectories')
    
    args = parser.parse_args()
    
    # Determine what to clean
    scales_to_clean = []
    
    if args.all:
        scales_to_clean = ['small', 'medium', 'production']
    elif args.scale:
        scales_to_clean = [args.scale]
    elif args.config:
        scale = get_scale_from_config(args.config)
        scales_to_clean = [scale]
    else:
        parser.error("Must specify --config, --scale, or --all")
    
    # Show summary
    print("🧹 RESULTS FOLDER CLEANUP")
    print("=" * 50)
    print(f"📋 Scales to clean: {', '.join(scales_to_clean)}")
    if args.dry_run:
        print("🔍 Mode: DRY RUN (no files will be removed)")
    else:
        print("💥 Mode: ACTIVE CLEANUP")
    print("")
    
    # Confirmation (unless forced or dry run)
    if not args.force and not args.dry_run:
        response = input("⚠️  This will permanently delete files. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled by user")
            return
    
    # Clean each scale
    success_count = 0
    for scale in scales_to_clean:
        print(f"\n🎯 Cleaning scale: {scale}")
        print("-" * 30)
        
        try:
            success = erase_results_folder(
                scale, 
                dry_run=args.dry_run,
                clean_empty_dirs=not args.no_clean_empty
            )
            if success:
                success_count += 1
                print(f"   ✅ {scale}: Complete")
            else:
                print(f"   ❌ {scale}: Failed")
        except Exception as e:
            print(f"   ❌ {scale}: Error - {e}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    print(f"✅ Successfully cleaned: {success_count}/{len(scales_to_clean)} scales")
    
    if success_count == len(scales_to_clean):
        if args.dry_run:
            print("🔍 Dry run completed - no files were actually removed")
        else:
            print("🎉 All cleanup operations completed successfully!")
    else:
        print("⚠️  Some cleanup operations failed - check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
