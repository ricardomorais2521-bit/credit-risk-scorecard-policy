# Personal Loans Credit Policy: Simplified Scorecard & Decision Optimization

An end-to-end, econometrically grounded credit scoring engine and cut-off optimization model for unsecured personal loans, developed using Lending Club's historical origination database.

---

##  Executive Summary & Business Questions

This project is built around two fundamental retail lending questions:
1. **Risk & Discrimination:** *Who poses the highest risk of non-payment using strictly application-time data, and how well can we predict default?*
2. **Policy & Strategy:** *Where should management set the approval cut-off to maximize profitability while keeping portfolio defaults within target risk appetite?*

Unlike naive machine learning workflows, this project follows Basel-compliant regulatory standards: no data leakage, no circular pricing features, explicit handling of right-censoring, Wilson confidence intervals for vintage stability, and complete economic cash-flow auditing.

---

##  Dataset & Population Design

* **Target Population:** 36-month unsecured personal loans with terminal outcomes (`Fully Paid` vs. `Charged Off / Default`).
* **Handling Right-Censoring:** Vintages 2016–2018 were excluded because 30% to 88% of loans remained active ("Current"), which would artificially underestimate real default rates. Vintages prior to 2012 were omitted due to low volume and post-crisis macroeconomic instability.
* **Development Window (Training):** 2012–2014 ($N = 306,462$ loans; Bad Rate: 13.25%).
* **Out-of-Time Window (Testing):** 2015 ($N = 283,026$ loans; Bad Rate: 14.89%).
* **Event Density:** Over 40,000 default events in development ($\text{EPV} = 2,388$ across 17 estimated parameters), completely eliminating risk of coefficient overfitting.

---

##  Feature Engineering & Selection Pipeline

1. **Information Value (IV) Auditing:** Evaluated over 30 candidate bureau and application features. Retained predictive, non-suspicious variables ($0.02 \le \text{IV} \le 0.50$).
2. **Missing Value Temporal Gate:** Screened variables across all vintages; excluded features with $>5\%$ missingness to prevent variables from acting as historical vintage detectors.
3. **Multicollinearity Control:** Applied pairwise Spearman rank correlation ($\vert{}\rho\vert{} < 0.70$) and recursive Variance Inflation Factor (VIF) pruning, ensuring all final predictors had $\text{VIF} < 1.40$.
4. **Log-Odds Linearity & Preprocessing:** Handled right-skewed variables (`annual_inc`) via $\log(1+x)$ transformations, capped 1st/99th percentile outliers with train-fitted winsorization, and pooled low-frequency categories ($<1\%$) into a `RARE` dummy group.

### Final Econometric Model Predictors
* `fico_range_low` ($\beta = -0.3214$, $\text{OR} = 0.73$): Delinquency history remains the primary risk barrier.
* `annual_inc` ($\beta = -0.2099$, $\text{OR} = 0.81$): Proportional liquidity cushion against income shocks.
* `inq_last_6mths` ($\beta = +0.1668$, $\text{OR} = 1.18$): Credit-seeking pressure.
* `dti` ($\beta = +0.1566$, $\text{OR} = 1.17$): Existing debt burden.
* `revol_util` ($\beta = +0.0425$, $\text{OR} = 1.04$): Utilization of revolving credit lines.
* Categorial controls: `home_ownership` and `purpose` (e.g., `purpose=small_business` exhibits $\text{OR} = 2.17$).

---

##  Model Validation & Diagnostics (OOT 2015)

* **Discrimination:**
  * **AUC-ROC:** $0.6479$ [95% Bootstrap CI: $0.6453$ – $0.6505$]
  * **Gini Coefficient:** $29.58\%$ ($0.2958$)
  * **Kolmogorov-Smirnov (KS):** $0.2116$
* **Calibration & Stability:**
  * Calibration slope: $0.996$ (ideal: $1.0$).
  * Sharp decile sorting: Bad rate strictly monotonic from **4.43% (Decile 1 - safest)** to **27.31% (Decile 10 - riskiest)**.
* **Drift Diagnostics:** Detected minor concept drift (mean predicted PD in 2015 was 13.16% vs. 14.89% actual default rate), confirming that individual borrower risk shifted slightly due to macroeconomic or unobserved underwriting shifts rather than population mix changes.

---

## Policy Decision & Profit Optimization

The lending policy simulates expected portfolio net profit ($P\&L$) across approval thresholds ($q$):
$$\mathbb{E}[\text{Profit}_i] = (1 - PD_i) \cdot \text{Interest}_i - PD_i \cdot \text{LGD} \cdot \text{EAD}_i - \text{Funding Cost}_i$$

* **Baseline Assumptions:** $\text{LGD} = 65\%$, $\text{Funding Cost} = 5\%$/year ($\text{Average Life} = 1.5\text{ years}$), Management Risk Appetite: $\text{Bad Rate} \le 10\%$.

| Policy Scenario | Optimal Approval Rate | Max Cut-off ($PD^*$) | Portfolio Bad Rate | Expected Net Profit | Realized Net Profit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unconstrained Maximum** | 70% | 15.8% | 11.49% | +$38.77M | -$12.9M |
| **Constrained Recommendation (Appetite $\le 10\%$)** | **50%** | **12.4%** | **9.54%** | **+$34.89M** | **-$3.77M** |
| *Strict 5% PD Tolerance* | 6.4% | 5.0% | 4.47% | +$5.2M | +$1.8M |

### Sensitivity & Stress Testing
* **Funding Cost Shock (7%):** Destroys net lending margins across all volume levels, forcing total origination closure ($0\%$ approval).
* **Macroeconomic Stress ($PD \times 1.2$):** Contracts optimal approval volume down to 35% ($PD \le 10.2\%$) to preserve solvency.

### Cash Flow Reconciliation (Profit Bridge)
A key finding in the audit was explaining the gap between Expected (+$34.9M) and Realized (-$3.77M) profit:
1. **Loan Prepayments:** Borrowers who repay early generate significantly less interest revenue than contractual terms dictate.
2. **Loss Accounting:** Applying flat LGD to original balance neglects principal recovered prior to default dates.

---

## Project Structure

```text
├── credit_scorecard_project.py   # Complete self-contained modeling pipeline
├── EDA_project.ipynb             # Exploratory Data Analysis & audit notebook
├── requirements.txt              # Environment dependencies
└── README.md                     # Technical documentation & business findings
