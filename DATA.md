# Data And Large Artifacts

[KNOWN, HIGH] GitHub has practical file-size limits and is not a good place to store the full M5 raw CSV files, pickle feature tables, or LightGBM binary models directly.

Keep these files locally in the project root when you need to reproduce the full pipeline:

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`
- `sample_submission.csv`
- `grid_part_*.pkl`
- `archive/*.pkl`
- `archive/*.bin`

The repository is prepared to publish code, documentation, small diagnostic outputs, and the interactive report. If you want the large artifacts versioned later, use Git LFS, Kaggle dataset references, or a separate cloud storage link.
