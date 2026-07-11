# Video Game Sales Prediction

Predicting total global video game sales using pre-release information (critic score, console, genre, publisher, developer, release year), built as a supervised regression project in scikit-learn.

## Dataset

Source: [Maven Analytics](https://www.mavenanalytics.io/) — a free open data repository. The file `vgchartz-2024.csv` is used as `video_games_sales_data.csv`, containing 64,016 rows and 14 columns of video game data (title, console, genre, publisher, developer, critic score, regional/global sales, release date).

## Data quality — read this before trusting the numbers

This dataset is heavily incomplete, and that materially limits what any model built on it can claim:

| Column | Missing | % of rows |
|---|---|---|
| `critic_score` | 57,338 | ~90% |
| `total_sales` (target) | 45,094 | ~70% |
| `na_sales` / `jp_sales` / `pal_sales` / `other_sales` | 48,000–57,000 each | 75–90% |
| `release_date` | 7,051 | ~11% |
| `developer` | 17 | <1% |

`critic_score` in particular is missing for roughly 9 in 10 rows, meaning the majority of values that feature contributes to the model are median-imputed, not real reviews. This isn't disclosed anywhere in the original data description - it only shows up once you check `.isna().sum()` yourself. Any model performance numbers below should be read with that in mind: the ceiling on achievable accuracy is capped by how much of the input signal is real versus filled-in.

## Preprocessing

- **Median imputation** for skewed numerical columns (`critic_score`, `release_year`), chosen over mean/mode because distributions were skewed rather than symmetric or purely categorical.
- **Extracted `release_year`** from `release_date`, then dropped `release_date` and `last_update` (the latter is a record-edit timestamp, not a sales driver).
- **Dropped 17 rows** with missing `developer` — negligible relative to dataset size, no meaningful bias introduced.
- **Bucketed rare categories**: `publisher` and `developer` were collapsed - any publisher/developer with fewer than 20 games was grouped into `"Other"` - to avoid one-hot encoding thousands of near-empty sparse columns, which would have caused the model to memorize individual publishers rather than generalize.

## Feature leakage — the mistake this project deliberately avoids

An earlier version of this analysis (following a common tutorial pattern) used the four regional sales columns (`na_sales`, `jp_sales`, `pal_sales`, `other_sales`) as *predictors* of `total_sales`. Since `total_sales` is the arithmetic sum of those four columns, that setup isn't prediction - it's the model learning addition, and it produces a misleadingly high R² (~0.996) that reflects nothing about real predictive power.

This project excludes all four regional sales columns from the feature set. The model predicts `total_sales` using only information available **before** a game's commercial performance is known: `critic_score`, `console`, `genre`, `publisher_grouped`, `developer_grouped`, `release_year`.

## Modeling

Two models were trained and compared on an 80/20 train-test split:

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.405 | 0.111 |
| Random Forest | 0.348 | 0.343 |

Random Forest outperforms Linear Regression, consistent with the expectation that sales are driven by nonlinear interactions (e.g., a high critic score matters differently depending on genre and console) that a linear model can't capture.

An R² of ~0.34 is modest, not broken. It reflects two honest limits: the features available here don't include major real-world sales drivers (marketing spend, franchise recognition, install base, launch timing), and a large share of the input data itself is imputed rather than observed.

## Limitations

- Heavy imputation in `critic_score` and `total_sales` caps how much this model can be trusted as a predictor of real-world outcomes.
- Publisher/developer grouping means ~14–20% of rows fall into an `"Other"` or effectively-unknown bucket, diluting that feature's signal.
- No marketing, franchise, or platform install-base data - all known strong drivers of game sales — are present in this dataset.

## Possible next steps

- Try gradient-boosted models (XGBoost, LightGBM) as a stronger nonlinear baseline than Random Forest.
- Engineer franchise/series features (e.g., "is this a sequel") from game titles.
- Re-run the analysis restricted to rows with genuine (non-imputed) `critic_score` to see how much the imputation is dragging down performance.
