# Caption Bank

## Architecture Diagram
- Title: DC-REIF Architecture for Real Estate Intelligence
- Caption: Overview of the DC-REIF control plane, governance layer, market representation layer, valuation core, and trust-oriented decision support outputs.

## Pipeline Diagram
- Title: End-to-End DC-REIF Pipeline
- Caption: Reproducible workflow from raw-data download and validation through segmentation, valuation, uncertainty estimation, pricing anomaly detection, and report-ready exports.

## Price Distribution
- Title: Distribution of King County Sale Prices
- Caption: Histogram of observed sale prices used to characterize target skewness and overall market dispersion in the raw transaction data.

## Log-Price Distribution
- Title: Log-Transformed Sale Price Distribution
- Caption: Log-scale visualization of sale prices showing a more compact target distribution for descriptive analysis and outlier inspection.

## Spatial Price Map
- Title: Spatial Distribution of Sale Prices
- Caption: Latitude-longitude scatter plot illustrating geographic concentration and price heterogeneity across the King County housing market.

## Temporal Trend
- Title: Median Sale Price Over Time
- Caption: Monthly median sale price trend highlighting temporal market movement and motivating time-aware train-validation-test splits.

## Feature Importance
- Title: Global Feature Importance for the Official Valuation Model
- Caption: Ranked feature importance values from the selected valuation model, with structural quality, living area, and location-related variables contributing the strongest signal.

## SHAP Summary
- Title: SHAP Summary for the Selected Valuation Model
- Caption: SHAP-based summary of global driver effects for the final valuation model, included as an explainability reference where runtime permits.

## Valuation Performance Table
- Title: Official Valuation Model Performance
- Caption: Validation and holdout test metrics for the single official DC-REIF valuation model used in the final pricing anomaly workflow.

## Segmentation Summary Table
- Title: KMeans Submarket Representation Summary
- Caption: Cluster-level summary for the KMeans submarket encoding used as market-context representation within the valuation workflow.

## Anomaly Summary Table
- Title: Pricing Anomaly Category Counts
- Caption: Counts of sale transactions classified as within expected range, potentially over-valued, potentially under-valued, or insufficient history.

## Property Intelligence Example Table
- Title: Illustrative Property Intelligence Cases
- Caption: Selected property-level examples showing observed sale price, model-implied fair value, uncertainty interval, segment context, anomaly label, and top drivers.