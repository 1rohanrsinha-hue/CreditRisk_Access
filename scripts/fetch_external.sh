#!/usr/bin/env bash
# Fetch the two UCI files into external_data/ (git-ignored). Skips files that already exist.
# Give Me Some Credit (Kaggle cs-training.csv) needs a login: place it in external_data/ by hand.
set -u
cd "$(dirname "$0")/.." && mkdir -p external_data && cd external_data
[ -s german.data ] || curl -sS -L --max-time 60 -o german.data \
  https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data </dev/null
[ -s taiwan.xls ] || curl -sS -L --max-time 120 -o taiwan.xls \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" </dev/null
ls -l
