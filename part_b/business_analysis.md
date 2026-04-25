# Part B: Business Case Analysis

## Scenario

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. The company wants to determine which promotion to deploy in each store each month to maximise items sold.

---

## B1. Problem Formulation (8 marks)

### (a) Problem Formulation (3 marks)

**Target Variable:** `items_sold` — the number of items sold in a store during a given month under a given promotion.

**Candidate Input Features:**
- Store-level: `store_id`, `store_size`, `location_type`, `monthly_footfall`, `competition_density`, `customer_demographics`
- Promotion-level: `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points)
- Temporal: `month`, `year`, `is_weekend`, `is_festival`, `season`
- Interaction features: `store_size × promotion_type`, `location_type × month`

**Type of Problem:** This is a **supervised regression problem**.

**Justification:** The target variable (`items_sold`) is continuous and numeric. The goal is to predict the expected number of items sold for each store-promotion-month combination, given a set of input features. Once trained, the model can be used to score all five promotion types for each store each month, and the promotion with the highest predicted `items_sold` is recommended.

An alternative framing as a **multi-class classification problem** (where the class is the best promotion type) is possible, but regression is preferred because:
1. It preserves the magnitude of differences between promotions (a promotion expected to sell 500 items vs 480 items carries quantitative meaning).
2. It allows the business to estimate expected uplift, not just rank promotions.
3. A single regression model can generate all five scores in one forward pass by varying the `promotion_type` input.

---

### (b) Why Items Sold is a Better Target than Revenue (3 marks)

**Argument for using `items_sold` over revenue:**

Revenue (= price × quantity) conflates two separate effects: the volume effect (how many items the promotion moves) and the pricing effect (what price those items were sold at). A "Flat Discount" promotion, by definition, reduces the per-unit price, so it will always suppress revenue relative to a non-discounted baseline — even if it moves significantly more units. Using revenue as the target would systematically penalise discount-based promotions and favour loyalty or gift-based promotions that do not cut the ticket price, regardless of actual sales volume performance.

`items_sold` isolates the **volume uplift** attributable to the promotion, which is what the marketing team is actually trying to optimise. It also avoids confounding from external pricing changes, markdowns, and product mix shifts.

**Broader Principle — Target Variable Selection:**

This illustrates the principle of **measuring what you actually want to optimise, not a proxy that is convenient to record**. In real-world ML projects, the most readily available metric (revenue, clicks, page views) is frequently a noisy or biased proxy for the true business objective (profitability, conversions, user value). Target leakage, confounding effects, and incentive misalignment all stem from choosing a target that is easy to measure rather than one that is conceptually correct. Good target variable selection requires close collaboration between data scientists and domain experts to ensure the label reflects the decision being automated.

---

### (c) Global Model vs. Location-Specific Strategy (2 marks)

**Problem with a single global model:**

A single model trained across all 50 stores assumes that the relationship between promotions and sales is the same everywhere. In reality, a BOGO promotion in an urban store with high-income footfall may drive very different behaviour than the same BOGO in a rural store where price sensitivity is the primary driver. A global model will learn an average effect, underperforming in every specific context.

**Alternative Strategy: Hierarchical / Stratified Modelling**

Rather than one global model, use a **stratified modelling strategy**:

1. **Segment stores** by location type (urban, semi-urban, rural) and store size. Train a separate model per segment. This preserves local response patterns while ensuring sufficient training data per model.
2. **Mixed-effects (hierarchical) model**: Include store-level random effects to allow each store to have its own intercept and promotion-response slope, while sharing strength across stores with similar profiles. This is particularly powerful when some stores have limited history.
3. **Feature-based personalisation**: Keep one global model but enrich it with store-level embeddings or aggregated historical promotion-response features per store, allowing the model to learn store-specific behaviour implicitly.

The recommended approach is **option 2 (hierarchical modelling)** — it balances data efficiency with local specificity, and can be implemented using libraries like `statsmodels` (mixed LME) or `LightGBM` with store-level leave-one-out encoding.

---

## B2. Data and EDA Strategy (10 marks)

### (a) Joining Tables and Dataset Grain (4 marks)

**Source Tables:**
1. `transactions` — one row per transaction or daily store sales record
2. `store_attributes` — one row per store (store_id, size, location_type, footfall, demographics, competition_density)
3. `promotion_details` — one row per promotion per store per month (store_id, month, promotion_type)
4. `calendar` — one row per date (date, is_weekend, is_festival, month, year, season)

**Join Strategy:**

```
transactions
  LEFT JOIN store_attributes   ON transactions.store_id = store_attributes.store_id
  LEFT JOIN calendar           ON transactions.date = calendar.date
  LEFT JOIN promotion_details  ON transactions.store_id = promotion_details.store_id
                               AND calendar.month = promotion_details.month
                               AND calendar.year  = promotion_details.year
```

**Grain of the Final Dataset:**

One row = **one store × one month × one promotion type**

This is the prediction unit: "what is the expected `items_sold` for store X using promotion Y in month M?"

**Aggregations before modelling:**
- Sum `items_sold` per store per month (from daily/transactional data)
- Average `competition_density` per store per month (if it varies)
- Join store attributes (static, no aggregation needed)
- Join calendar flags for the month (aggregate `is_festival` as "any festival day this month")

The result is a clean, wide-format dataset where each row has all store, promotion, and temporal features alongside the monthly `items_sold` target.

---

### (b) EDA Strategy (4 marks)

**Analysis 1 — Promotion Type vs. Average Items Sold (Bar Chart)**

*What to look for:* Which promotion type produces the highest average items sold across all stores and months? Are there clear leaders or is performance similar across promotions?

*Influence on modelling:* If one promotion dominates universally, the problem may be trivial and a simpler rule-based system suffices. If performance is highly context-dependent, it confirms the need for a store-specific model. Also reveals class imbalance in the promotion label distribution.

**Analysis 2 — Monthly Sales Trend Line (Time Series Plot)**

*What to look for:* Seasonal patterns — which months consistently show sales peaks? Are there year-over-year trends (growth or decline)? Are there anomalous months (COVID disruptions, supply issues)?

*Influence on modelling:* Strong seasonality means `month` and `year` must be included as features. If the trend is strong, time-based features (months since store opening, sales_last_month) or lag features may be needed. Anomalous periods may need to be flagged or excluded from training.

**Analysis 3 — Store Location Type × Promotion Interaction Heatmap**

*What to look for:* Does the best-performing promotion differ by location type? Does BOGO outperform Flat Discount in urban stores but not rural ones?

*Influence on modelling:* Significant interaction effects justify creating explicit interaction features (e.g., `location_type × promotion_type`) or confirm the need for separate models per segment. If the heatmap shows flat interactions, a simpler additive model may suffice.

**Analysis 4 — Competition Density vs. Items Sold (Scatter Plot with Regression Line)**

*What to look for:* Is there a negative linear relationship between competition density and items sold? Does the effect differ by promotion type (do some promotions overcome local competition better)?

*Influence on modelling:* Confirms `competition_density` as a continuous feature (rather than binning it). If the effect is non-linear (diminishing returns above a threshold), a polynomial feature or tree-based model is more appropriate. Interaction with promotion type may motivate a competition_density × promotion_type feature.

---

### (c) Handling the 80% No-Promotion Imbalance (2 marks)

**How this affects the model:**

When 80% of transactions occur without a promotion, the model will be heavily trained on the baseline (no-promotion) behaviour. This creates two risks:

1. **Biased predictions**: The model may underestimate the incremental uplift from promotions because it has seen relatively few promoted examples. It will default toward predicting the no-promotion baseline.
2. **Misleading evaluation**: If accuracy or RMSE is measured across all records including the 80% no-promotion majority, a model that always predicts the baseline will appear to perform well without actually learning promotion effects.

**Steps to address it:**

1. **Stratified sampling**: When splitting train/test, stratify on `has_promotion` flag to ensure promoted records are proportionally represented in both sets.
2. **Focused evaluation**: Report performance metrics separately for promoted vs. non-promoted records. The primary metric for the business decision should be performance on *promoted* records.
3. **Re-weighting**: Assign higher `sample_weight` to promoted records during model training so the loss function penalises errors on promoted observations more heavily.
4. **Promotion-only sub-model**: Train a secondary model exclusively on promoted store-months to learn the incremental effect of each promotion type. Use the no-promotion model as a baseline and the promotion model to estimate uplift.

---

## B3. Model Evaluation and Deployment (12 marks)

### (a) Train-Test Split, Evaluation Metrics (4 marks)

**Train-Test Split Strategy:**

With 3 years of monthly data across 50 stores (≈1,800 store-month records), use a **temporal split**:
- **Training set**: First 2 years (months 1–24)
- **Validation set**: Month 25–30 (for hyperparameter tuning)
- **Test set**: Final 6 months (months 31–36)

For robust evaluation, use **time-series cross-validation (walk-forward validation)**:
- Fold 1: Train on months 1–12, test on months 13–15
- Fold 2: Train on months 1–15, test on months 16–18
- ... expanding window forward in time

**Why random split is inappropriate:**
A random split leaks future information into training (the model sees December 2024 patterns while predicting July 2023), overestimates performance, and does not simulate the real deployment scenario where the model always predicts into the future.

**Evaluation Metrics:**

| Metric | Formula | Interpretation in Business Context |
|--------|---------|-------------------------------------|
| **RMSE** | √(mean((ŷ−y)²)) | Penalises large errors heavily. A store predicted to sell 500 items but selling 200 is more costly than a store off by 20 items. Use RMSE to flag models that make catastrophic predictions. |
| **MAE** | mean(\|ŷ−y\|) | Average absolute error in items. Directly interpretable: "on average, our prediction is off by X items per store per month." Use for operational planning (stock ordering, staffing). |
| **MAPE** | mean(\|ŷ−y\|/y) × 100 | Percentage error, useful for comparing accuracy across stores of very different sizes. A 10% error means the same regardless of whether a store sells 100 or 10,000 items. |
| **Promotion Rank Accuracy** | % of store-months where the recommended promotion matches the optimal promotion | The ultimate business metric — does the model recommend the right promotion, even if point estimates are slightly off? |

---

### (b) Feature Importance for Explaining Recommendations (4 marks)

**Context:** The model recommends Loyalty Points Bonus for Store 12 in December, but Flat Discount for Store 12 in March.

**Investigation using Feature Importance:**

1. **Global feature importance** (from Random Forest or Gradient Boosting): Identify which features the model relies on most overall — likely `month`, `promotion_type`, `competition_density`, `is_festival`. This establishes the baseline explanation framework.

2. **SHAP (SHapley Additive exPlanations)**: For each of the two predictions (Store 12 Dec and Store 12 Mar), compute SHAP values for every feature. SHAP decomposes each prediction into the contribution of each feature, relative to the mean prediction. This directly answers: "why did feature X push this prediction up or down?"

3. **Specific explanation for December vs. March:**
   - In December, `is_festival=1`, `month=12` (peak season), `competition_density` may be elevated. SHAP will show that `is_festival` and `month` strongly push the model toward **Loyalty Points Bonus** — during peak season, customers are already primed to buy; locking them into a loyalty scheme maximises long-term value without eroding margin through discounting.
   - In March (off-season, no festivals), `month=3`, `is_festival=0`. SHAP will show `month` now pushes toward **Flat Discount** — in low-footfall months, a direct price incentive is needed to stimulate demand that would not materialise otherwise.

**Communicating to the Marketing Team:**

Present a simple two-column SHAP waterfall chart for each month, showing the top 5 features pushing the recommendation up or down from the average. Frame it as: "In December, festive season and month effects strongly favour loyalty programmes. In March, the absence of festivals means price incentives are needed to drive traffic." This grounds the recommendation in features the team already understands intuitively.

---

### (c) End-to-End Deployment Process (4 marks)

**1. Saving the Model**

After training on the full historical dataset, serialise the complete pipeline (preprocessor + model) using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'promotion_recommender_v1.pkl')
```

Store the model artifact in a version-controlled model registry (e.g., MLflow, AWS S3 with versioning). Tag with training date, dataset version, and performance metrics for traceability.

**2. Preparing and Feeding New Monthly Data**

At the start of each month:
1. Pull the latest store attributes, calendar flags, and last month's sales from the data warehouse.
2. Construct the prediction dataset: one row per store × per promotion type (50 stores × 5 promotions = 250 rows).
3. Load the saved pipeline and call `pipeline.predict()` on these 250 rows.
4. For each store, select the promotion with the highest predicted `items_sold`.
5. Output the recommendations as a CSV/dashboard for the marketing team by day 1 of each month.

No retraining is required — the saved pipeline handles all preprocessing transformations internally (the fitted StandardScaler and OneHotEncoder are embedded in the pipeline object).

**3. Monitoring for Degradation and Retraining Triggers**

Deploy a monitoring layer that tracks:

| Signal | Method | Threshold |
|--------|--------|-----------|
| **Prediction drift** | Compare distribution of predicted `items_sold` this month vs. last 3 months (KS test or PSI score). | PSI > 0.2 triggers alert |
| **Actual vs predicted error** | Once actuals arrive, compute RMSE/MAE for the month and compare to the baseline validation score. | >20% increase in MAE triggers review |
| **Feature distribution shift** | Monitor input feature distributions (e.g., mean `competition_density`, promotion mix). Sudden changes signal market shifts. | PSI > 0.1 on key features |
| **Recommendation acceptance rate** | Track how often the marketing team overrides the model's recommendation. A rising override rate signals loss of trust. | >30% override rate triggers review |

**Retraining Protocol:**
- **Scheduled retraining**: Retrain quarterly (every 3 months) with a rolling window of the most recent 2 years of data. This keeps the model current without discarding long-term seasonal patterns.
- **Triggered retraining**: If any monitoring threshold is breached, immediately investigate the root cause. If the shift is structural (e.g., a new competitor opened, a new store was added, a promotion type was changed), retrain with updated data and recalibrate.
- Use the held-out most recent 2 months as a validation set during retraining to confirm the new model outperforms the previous version before deploying.
