# CAS Sourcing & Procurement Intelligence MVP v15

v15 builds on v13 and focuses on supplier parser coverage and price-trust clarity.

## What changed

- Full curated supplier registry now contains 38 suppliers/directories.
- Adds US Biological, Biosynth, and 1ClickChemistry to the price-first/mixed registry.
- Adds candidate-aware pricing: prices found on CAS-unconfirmed pages are retained as LOW-confidence review evidence, not discarded.
- Keeps those candidate rows out of the desired-quantity model until product identity is confirmed.
- Adds USA/location-aware request headers and cookies to help sites such as Ambeed that prompt for shipping country/region.
- Adds snippet/search-result price parsing for cases where a supplier page fails to fetch but a search result clearly exposes CAS + pack + price.
- Adds supplier coverage diagnostics export.
- UI now separates:
  - all visible price candidates
  - model-eligible trusted mass catalog rows
  - desired-quantity scale-up model

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested test cases

- CAS `487-41-2` with desired quantities `25 mg`, `75 mg`, `500 mg`, `50 g`, `1 kg`.
- CAS `151-21-3` for commodity/catalog behavior.
- CAS `64-17-5` for easy commodity baseline.

## Important interpretation rule

A supplier may show a price in the product evidence table but still be excluded from the desired-quantity model if the row is CAS-unconfirmed, solution/reference-only, below required purity, noisy, or not mass-catalog eligible.
