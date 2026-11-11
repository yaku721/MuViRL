# MuViRL results

## Reproduce the summaries

From this directory, run:

```bash
python verify_results.py
```

The script recomputes accuracies and WER from counts or logits, sample standard deviations across seeds, unit similarities, and temporal offset profiles. 

To draw the four representation panels from the released CSV files, install Matplotlib and run:

```bash
python plot_results.py --output plots
```



The OuluVS2 split manifest in `distillation/dataset_splits.csv` also applies to the OuluVS2 representation experiment. `valid` denotes validation. 
