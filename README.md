# Using Domain Adaptation to Improve the Modeling of Water Quality with Sparse Data


## Requirements

```bash
pip install -r requirements.txt
```

## Usage
Uncomment and desired lines of code.
1. Download xlsx files from [HTLP dataset](https://ncwqr-data.org/HTLP/Portal) into Data/raw.
2. Use data_processing.py to create training data.
3. Use invariant.py to pretrain on source domain data.
4. Use Download/download.py to download relevant models.
5. Use variant.py to train fine-tune in the target domain.
6. Use Download/download.py to download results.
7. Use result_analysis.py to analyse the various metrics.
8. Use viz/viz_variant.py and viz/viz_invariant.py to make the graphs.

```

