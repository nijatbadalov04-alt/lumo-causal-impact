# ODM files: the complete seven-year set

The origin–destination analysis loads **every CSV** in `data/raw/odm/` as one timeline. The loader is
year-agnostic: it parses the financial-year start from the filename (e.g. `…2021-22…` → year 2021), so
adding a year needs no code change, drop the file in and re-run.

The complete set is **seven financial years**:

| # | Financial year | Parsed as | Example filename |
|---|---|---|---|
| 1 | 2018-19 | 2018 | `ODM_for_rdm_2018-19.csv` |
| 2 | 2019-20 | 2019 | `ODM_for_rdm_2019-20.csv` |
| 3 | 2020-21 | 2020 | `ODM_for_rdm_2020-21.csv` |
| 4 | 2021-22 (Lumo launch FY) | 2021 | `ODM_for_rdm_2021-22.csv` |
| 5 | 2022-23 | 2022 | `ODM_for_rdm_2022-23.csv` |
| 6 | 2023-24 | 2023 | `ODM_for_rdm_2023-24.csv` |
| 7 | 2024-25 | 2024 | `ODM_for_rdm_2024-25.csv` |

> **Verification tip.** An ODM file's true year is its first column, `Financial_Year` (2021-22 reads
> `20212022`). The filename is only a hint; the loader and the integrity check use the column.

## Adding a year

1. Download the relevant "Origin and destination matrix (ODM)" year from the Rail Data Marketplace
   (<https://raildata.org.uk/dashboard>).
2. Save it into `data/raw/odm/` with the financial year in the name (e.g. `ODM_for_rdm_2021-22.csv`).
3. Re-run `python run_pipeline.py` (or the OD stages). The file slots in by its parsed year and flows
   into every OD module (substitution, inference, corridor robustness, event-study DiD, air modal
   shift, carbon).

## Why the headline does not hinge on the launch year

The pre/post windows are **pre = 2018-19 and 2019-20** (clean pre-COVID) and **post = 2023-24 and
2024-25** (recovered). The 2021-22 launch year is a transition year (Lumo launched mid-year, October
2021, amid the COVID recovery), so it is used in neither window, it only adds a mid-timeline point that
improves the event study's launch-year resolution. The headline estimates are computed from the clean
pre/post means and are unchanged by its presence.
