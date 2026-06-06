# Data sources

Every dataset used in the project, with its canonical URL and the date it was accessed/verified.
Only open or public data is used; all are released under the UK Open Government Licence (OGL).
ORR `/media/...` paths are relative to `https://dataportal.orr.gov.uk`. Access dates are ISO `YYYY-MM-DD`.

## 1. ORR Estimates of Station Usage (panel spine)

Landing page: <https://dataportal.orr.gov.uk/statistics/usage/estimates-of-station-usage/> · Licence: OGL v3.0.

| Item | File | Verified |
|---|---|---|
| Table 1415: time series of entries/exits/interchanges by station | `/media/1908/table-1415-time-series-of-passenger-entries-and-exits-and-interchanges-by-station.ods` | 2026-06-04 |
| Table 1410: latest annual snapshot 2024-25 (ODS / CSV) | `/media/1907/...ods`, `/media/1909/...csv` | 2026-06-04 |
| Quality & Methodology report (ticket→journey conversion, breaks) | `/media/1917/station-usage-quality-and-methodology-report.pdf` | 2026-06-04 |
| Historical snapshots 2019-20 … 2023-24 | ORR `/media/...` (per year) | 2026-06-04 |

## 2. ORR Passenger Rail Usage (operator / sector)

Landing: <https://dataportal.orr.gov.uk/statistics/usage/passenger-rail-usage/>

| Item | File | Verified |
|---|---|---|
| Table 1223: journeys by operator (LNER and Lumo separate) | `/media/1476/table-1223-passenger-journeys-by-operator.ods` | 2026-06-04 |
| Table 1221: journeys by sector (open-access vs franchised, from 1994) | `/media/2011/table-1221-passenger-journeys-by-sector.ods` | 2026-06-04 |
| Table 3113: Public Performance Measure (punctuality) by operator | `/media/1428/table-3113-public-performance-measure-by-operator-and-sector.ods` | 2026-06-05 |
| Table 7180: average change in fares by ticket type / sector | `/media/1692/table-7180-average-change-in-fares-by-regulated-and-unregulated-tickets.ods` | 2026-06-05 |

## 3. Origin–Destination Matrix (the decisive market-level dataset)

| Source | Use | URL |
|---|---|---|
| Rail Data Marketplace: Origin-Destination Matrix (ODM), all seven financial years 2018-19 → 2024-25 (including the 2021-22 Lumo launch year); OGL v3.0, free account | Station-pair flows → the substitution-vs-creation test | <https://raildata.org.uk/dashboard> |

## 4. Modal-shift, carbon, and validation data

| Source | Use | URL / product |
|---|---|---|
| CAA Table 12.2: Domestic Air Passenger Traffic Route Analysis (annual; each file carries the year and prior year) | Air → rail modal shift: London-area airports ↔ Edinburgh / Glasgow / Newcastle | <https://www.caa.co.uk> (Datasets → Airport data) |
| Rail Data Marketplace: Green Travel Emissions Output (by journey, with comparisons) | kg CO₂e per passenger by mode, per OD pair → carbon / welfare | RDM product `P-3acebfe3-…` |
| Rail Data Marketplace: NWR Daily Concourse Footfall (Network Rail) | Physical gate counts, 18 managed stations, 2023–2026 → external validation of modelled usage | RDM product `P-ba403ccb-…` |
| NaPTAN rail nodes (DfT) | Station coordinates → distance-to-London covariate | <https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv> |

CAA route totals are terminal passengers, both directions, counted once per passenger-journey, the
same unit as ODM journeys, so the air-versus-rail comparison is like-for-like. Air figures sum all
London-area airports (Heathrow, Gatwick, Stansted, Luton, London City, Southend) to each city.

## 5. Treatment timing (intervention dates)

| Operator | Event | Date |
|---|---|---|
| Hull Trains | Hull–London King's Cross open-access launch | 2000-09-25 |
| Grand Central | Sunderland–London launch | 2007-12-18 |
| Grand Central | Bradford–London launch | 2010-05-23 |
| Lumo | Edinburgh–London King's Cross launch (stops incl. Newcastle, Morpeth, Stevenage) | 2021-10-25 |
