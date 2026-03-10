# WNBA Match Predictor

A machine learning project that predicts WNBA match outcomes and scorelines using team statistics scraped from the official WNBA website.

![Python](https://img.shields.io/badge/Python-3.9-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange) ![pandas](https://img.shields.io/badge/pandas-data--analysis-green)

---

## Overview

This project builds two predictive models on top of real 2025 WNBA team statistics:

1. **Match Outcome Classifier** — predicts which team wins a head-to-head matchup (75% accuracy)
2. **Points Regression Model** — estimates the scoreline and point differential between two teams

Both models are driven by engineered features combining each team's own stats with their opponent's defensive metrics, simulating a realistic matchup context.

---

## What It Does

Given any two WNBA teams, the predictor can:

- Tell you which team is more likely to win
- Estimate the projected score for each team
- Calculate the expected point differential

**Example output:**
```
predict_matchup("Golden State Valkyries", "New York Liberty")
→ 'New York Liberty is more likely to beat Golden State Valkyries.'

point_difference("Connecticut Sun", "Las Vegas Aces")
→ 'Connecticut Sun will score 75.89 and Las Vegas Aces will score 79.91
   The point difference is -4.02 points.'
```

---

## Data

- **Source:** Official WNBA website (stats.wnba.com)
- **Collected:** June 23, 2025 (mid-season snapshot)
- **Format:** CSV with 63 columns covering team offensive, defensive, and advanced metrics
- **Coverage:** All 13 WNBA teams, minimum 14 games played

The dataset includes both team stats and opponent stats per game, enabling the model to construct synthetic matchup contexts without requiring historical head-to-head data.

---

## Models

### Model 1 — Win Prediction (Random Forest Classifier)

Predicts the binary outcome (win/loss) for a given matchup.

**Features used:**

| Feature | Description |
|---|---|
| `TOV` | Team turnovers per game |
| `FG%` | Field goal percentage |
| `3P%` | Three-point percentage |
| `FT%` | Free throw percentage |
| `REB` | Total rebounds |
| `NetRtg` | Net rating (OffRtg − DefRtg) |
| `Opp_FG%` | Opponent field goal percentage |
| `Opp_3P%` | Opponent three-point percentage |
| `Opp_FTM` | Opponent free throws made |
| `Opp_REB` | Opponent rebounds |
| `Opp_TOV` | Opponent turnovers |

**Results:**
- Test accuracy: **75%**
- Train/test split: 70/30, `random_state=42`
- Estimators: 100 trees

### Model 2 — Points Regression (Random Forest Regressor)

Predicts the number of points each team scores, from which a point differential is derived.

**Features used:** `TOV`, `FG%`, `FGM`, `3P%`, `FT%`, `REB`, `AST`, `Opp_REB`, `Opp_TOV`, `Opp_BLK`, `Opp_STL`

**Results:**
- R² score: **0.12**
- Note: the small dataset size (13 teams) limits regression performance. The win classifier is the stronger model and the more reliable output to use.

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/kvmnisi/wnba.git
cd wnba
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn jupyter
```

### 3. Add the dataset
Place `wnba_stats_23June25.csv` in the project root and update the file path in the first cell:
```python
df = pd.read_csv('wnba_stats_23June25.csv')
```

### 4. Run the notebook
```bash
jupyter notebook wnba_predictor.ipynb
```

### 5. Make a prediction
```python
predict_matchup("Indiana Fever", "Seattle Storm")
point_difference("Minnesota Lynx", "Las Vegas Aces")
```

---

## Tech Stack

- Python 3.9
- pandas
- scikit-learn (RandomForestClassifier, RandomForestRegressor)
- Jupyter Notebook

---

## Limitations & Next Steps

- Dataset is a mid-season snapshot — predictions improve with more games played
- 13-team dataset is small for regression; points model would benefit from per-game historical data rather than season averages
- No injury or roster data factored in
- Potential next step: retrain at end of season, add player-level features, or pull historical seasons for a larger training set
