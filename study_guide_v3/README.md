# Study guide v3 (20 October 2026)

The full study guide, updated with the leak-free reruns of 19–20 October.
Earlier versions (v1, v2) are superseded and kept out of the repository.

| File | Purpose |
|------|---------|
| `ecoexplain_study_guide_v3.pdf` | The guide (open this) |
| `RESULTS_MASTER.md` | Every score with its source file, recomputed from the predictions |
| `RESULTS_MASTER.json` | The same for scripts |
| `chapters/generated_master_table.tex` | The results table used in the PDF (generated; do not edit) |

## After a new run

```bash
python study_guide_v3/build_results_master.py     # refresh RESULTS_MASTER + generated table
python study_guide_v3/make_figures_v3.py          # refresh figures
cd study_guide_v3 && latexmk -pdf ecoexplain_study_guide_v3.tex
```

To add a new run, add one line to `RUNS` in `build_results_master.py`.
