# Monitoring and Retraining Plan

The current repository uses a static public dataset. This plan defines the minimum monitoring layer needed before the framework can be operated as a repeated data product.

## Data Freshness

Track for every new batch:

- extract timestamp
- source filename or source URI
- row count
- min and max sale date
- duplicate `id` count
- checksum or source hash when available

Trigger review if:

- no new data arrives in the expected window
- sale dates regress unexpectedly
- row counts shift materially from historical batch sizes

## Schema and Quality Monitoring

Validate every batch against `configs/data_contracts/schema_contract.json`.

Track:

- missing required columns
- missingness by field
- invalid numeric values
- invalid dates
- invalid coordinates
- impossible structural values

Trigger review if:

- required columns are missing
- target missingness changes materially
- invalid coordinates or structural fields exceed a defined threshold

## Distribution Drift

Monitor drift for:

- `price`
- `sqft_living`
- `grade`
- `condition`
- `zipcode`
- `lat` / `long`
- `sale_month`
- `house_age`

Initial implementation can use simple, explainable checks:

- quantile comparison
- mean / median shift
- category share shift
- population stability index for major fields

## Model Performance Monitoring

When labels are available, track:

- MAE
- RMSE
- MAPE
- interval coverage
- average interval width
- anomaly rate
- abstention rate

Report these metrics overall and by:

- zipcode
- segment
- predicted price band
- high-value band
- renovated flag
- waterfront / view indicators when available

## Abstention Monitoring

Abstention is a product risk metric.

Track:

- overall abstention rate
- abstention by zipcode
- abstention by segment
- abstention by price band
- abstention among high-value homes

Trigger review if:

- abstention exceeds the agreed product threshold
- abstention concentrates in high-value homes
- abstention spikes in a new geographic area

## Retraining Policy

Retraining should be considered when:

- interval coverage drops below the accepted range
- abstention rises above the accepted threshold
- MAE/RMSE degrade materially on recent labeled data
- distribution drift is sustained across multiple batches
- a new dataset version or source definition is approved

Before promoting a new model:

1. Re-run the full pipeline.
2. Compare old and new models on holdout and slice metrics.
3. Compare abstention and interval coverage.
4. Record model metadata and selected parameters.
5. Keep the previous artifact until the new model passes validation.
