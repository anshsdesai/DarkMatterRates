#!/bin/bash
#sometimes when getting the halo data there are annoying rounding errors where 
#ex. a folder with sigmaE 9.99e-34 gets created instead of sigmaE 1e-33
#This is a helper script to fix the problem. Just source it from whichever directory contains folders with this issue
# Usage: ./round_rename.sh [--dry-run]
#   --dry-run  : Preview changes without renaming filesy

DRY_RUN=false

# Parse command-line argument
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "=== DRY RUN MODE (no files will be changed) ==="
fi

# Process files
for file in *9.99e-*.csv; do
  if [[ -f "$file" && $file =~ 9\.99e-([0-9]+) ]]; then
    old_exp="${BASH_REMATCH[1]}"
    new_exp=$(( old_exp - 1 ))
    newfile="${file//9.99e-${old_exp}/1e-${new_exp}}"

    if [[ "$DRY_RUN" == true ]]; then
      echo "[DRY RUN] Would rename: '$file' -> '$newfile'"
    else
      mv -v "$file" "$newfile"
    fi
  fi
done

[[ "$DRY_RUN" == true ]] && echo "Dry run complete. Run without --dry-run to execute."
