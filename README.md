# 📊 Surprise Index Investment Strategy

This is a quantitative investment strategy driven by the surpriseness of company's key financial index, using both linear and machine learning model to build an risk neutral stock seletion strategy

## 🧭 Strategy Workflow

### 1. 📉 Linear Model for Initial Variable Filtering
We first use a simple linear regression model to detect preliminary relationships between surprise indices and stock/sector returns.
````python
# 定義用來的特徵欄位
X_cols = [
    "surprised_index_gross_profit",
    "surprised_index_current_ratio",
    "surprised_index_eps",
    "surprised_index_roa_beforetta",
    "surprised_index_operating_cashflow",
]
# 去極值
cd = trim_outliers_joint(df, X_cols + ["this_season_return"])
cd = cd.dropna()

# X, y
X = cd[X_cols]
y = cd["this_season_return"]

# 建立並訓練模型
model = LinearRegression()
model.fit(X, y)

# 預測
y_pred = model.predict(X)
````
![Linear Model Output](output.png)

---

### 2. 🏭 Industry-Level Correlation Analysis
We examine how different sectors respond to surprise indices.

![Industry Correlation](industry_corr.png)

---

### 3. 🌳 Feature Importance via Random Forest
We apply Random Forest to further assess variable importance and capture nonlinear relationships.

- **Variable Importance Plot**
  ![Variable Importance](variable_impo.png)

- **Feature Heatmap**
  ![Random Forest Heatmap](randomforest_heat.png)

---

### 4. 🧪 Strategy Screening: Multiple Thresholds
Using the above signals, we construct multiple strategies based on different percentile thresholds to evaluate effectiveness.

- **Screening Strategy Performance**
  ![Screening Performance](screening_performance.png)

- **Return Comparison**
  ![Return Comparison](return_compare.png)

- **Sharpe Ratio Comparison**
  ![Sharpe Comparison](sharp_compare.png)

---

### 5. ⚖️ Final Strategy: Long-Short Top/Bottom 20%
We select the **top 20%** and **bottom 20%** ranked stocks to build a market-neutral long-short strategy.
````python
# 初始化
returns_80    = pd.Series(dtype='float64')
returns_50abv = pd.Series(dtype='float64')
returns_50blw = pd.Series(dtype='float64')
returns_20    = pd.Series(dtype='float64')

# quantiles
pr80 = np.quantile(y_pred, 0.8)
pr50 = np.quantile(y_pred, 0.5)
pr20 = np.quantile(y_pred, 0.2)

# 三個投組的股票列表
portfolio_80 = np.array([])
portfolio_50abv = np.array([])
portfolio_50blw = np.array([])
portfolio_20 = np.array([])

# 計算每日投組平均報酬的函數
def get_return(portfolio, date):
    if date not in return_df.index:
        return np.nan
    # 避免 portfolio 中有不存在的股票代碼
    valid = [s for s in portfolio if s in return_df.columns]
    if not valid:
        return np.nan
    return return_df.loc[date, valid].mean()

# 逐日更新
for i in date:

    today_new = cd.loc[cd["date"] == i].copy()

    
    # 移除今天剛公告的股票
    portfolio_80     = portfolio_80[~np.isin(portfolio_80, today_new["stock"].values)]
    portfolio_50abv  = portfolio_50abv[~np.isin(portfolio_50abv, today_new["stock"].values)]
    portfolio_50blw  = portfolio_50blw[~np.isin(portfolio_50blw, today_new["stock"].values)]
    portfolio_20     = portfolio_20[~np.isin(portfolio_20, today_new["stock"].values)]




    # 分別挑出今天要加入的股票
    buy80 = today_new["return_pred"] >= pr80
    buy50abv = today_new["return_pred"] >= pr50
    buy50blw = today_new["return_pred"] <= pr50
    buy20 = today_new["return_pred"] <= pr20


    
    # 分別加入新的股票
    portfolio_80     = np.concatenate([portfolio_80, today_new.loc[buy80, "stock"].values])
    portfolio_50abv  = np.concatenate([portfolio_50abv, today_new.loc[buy50abv, "stock"].values])
    portfolio_50blw  = np.concatenate([portfolio_50blw, today_new.loc[buy50blw, "stock"].values])
    portfolio_20     = np.concatenate([portfolio_20, today_new.loc[buy20, "stock"].values])

    # 計算三組投組的回報
    returns_80.loc[i] = get_return(portfolio_80, i)
    returns_50abv.loc[i] = get_return(portfolio_50abv, i)
    returns_50blw.loc[i] = get_return(portfolio_50blw, i)
    returns_20.loc[i] = get_return(portfolio_20, i)
````
- **Final Strategy Backtest**
  ![Strategy Performance](strat_perform.png)

---

### 6. 📉 Risk Analysis
Max drawdown during the backtest period.

![Drawdown](drawdown.png)

---

## 📊 Key Metrics (fill in manually)

| Metric          | Value    |
|-----------------|----------|
| Annual Return   | XX.X %   |
| Max Drawdown    | XX.X %   |
| Sharpe Ratio    | X.XX     |
| Win Rate        | XX %     |
| Backtest Period | 20XX–20XX |

---

## ▶️ Run This Project

```bash
git clone https://github.com/your_username/suprise-index-investment-strategy.git
cd suprise-index-investment-strategy
jupyter notebook strategy_pipeline.ipynb
