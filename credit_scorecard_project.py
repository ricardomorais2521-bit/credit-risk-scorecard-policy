# =============================================================================
# credit_scorecard_project.py
# Personal Loans Credit Policy: Simplified Scorecard
#
# Two business questions drive everything in this file:
#   Q1. Who poses the highest risk of non-payment, and how well can we predict it?
#       - Which variables best separate good from bad borrowers (IV), and how do they
#         relate to risk?
#       - What discriminatory power does the model achieve out of time (AUC, Gini, KS)?
#       - Are the predicted probabilities well calibrated against observed default rates?
#   Q2. Where do we set the approval cut-off to avoid losing money?
#       - Approval rate vs. bad rate for cut-offs defined by score percentiles, compared
#         with the baseline (bad rate when approving everyone).
#       - Expected profit by cut-off, and how sensitive the best cut-off is to the
#         funding cost assumption.
#
# HOW TO READ THIS FILE
#   PART A  Concepts used in the project (credit risk, sampling, statistics)
#   PART B  Dictionary of every variable we touch
#   PART C  Modelling assumptions and where each one is checked
#   PART D  Code, in numbered sections (each section starts with "# %%")
# =============================================================================


# #############################################################################
# PART A. CONCEPTS
# #############################################################################
#
# --- A1. Credit risk concepts -------------------------------------------------
# Credit risk     Risk that a borrower does not repay a loan as agreed.
# Default / bad   Here: a loan that ended as "Charged Off" or "Default"
#                 (the lender wrote it off as lost). Coded bad = 1.
# Good            A loan that was "Fully Paid". Coded bad = 0.
# Bad rate        Share of bad loans among loans whose outcome is known:
#                     bad rate = bad / (bad + good)
# PD              Probability of Default: the model's estimate of P(bad = 1)
#                 for one applicant, using only application data.
# LGD             Loss Given Default: share of the exposure that is NOT recovered
#                 when a loan defaults (LGD = 65% means 35% is recovered).
# EAD             Exposure At Default: amount owed when the default happens.
#                 Simplification here: EAD = original loan amount (conservative,
#                 because amortisation lowers the real exposure over time).
# Expected loss   EL = PD x LGD x EAD. The average loss we should expect per loan.
# Scorecard       Interpretable model that turns application data into a risk
#                 score; here a logistic regression that outputs a PD.
# Cut-off         Maximum risk we accept. Applicants riskier than the cut-off are
#                 rejected. Here defined on the PD ranking (approve the X% safest).
# Approval rate   Share of applicants approved at a given cut-off.
# Risk appetite   Maximum bad rate management is willing to accept (an assumption
#                 set by management, not estimated from data).
# Funding cost    Interest the lender pays to obtain the money it lends out.
# Expected profit Profit estimated BEFORE outcomes are known, using PD:
#                     (1 - PD) x interest income - PD x LGD x EAD - funding cost
#
# --- A2. Sample and data concepts --------------------------------------------
# Vintage         Group of loans granted in the same period (here: calendar year
#                 of issue_d). Comparing vintages shows how risk changes in time.
# Maturity        A loan is mature when its contractual life has ended, so the
#                 outcome is final. A 36-month loan issued in Dec-2015 matures in
#                 Dec-2018, the last month of this dataset.
# Right-censoring Loans still running ("Current") have an unknown final outcome.
#                 Treating them as good would understate the bad rate, so
#                 recent vintages are excluded instead.
# Resolved loan   Loan with a final outcome (Fully Paid, Charged Off, Default).
# Population      The loans the policy applies to: 36-month, resolved loans.
# Development     Data used to build the model (vintages 2012-2014).
# Out-of-time     Data from a LATER period, used only to test the model (2015).
# (OOT) test      It mimics real use: a model is built on the past and applied to
#                 the future. A random split would mix periods and overstate
#                 performance, because train and test would share the same
#                 economic conditions.
# Data leakage    Using information that would NOT be available at decision time
#                 (payments, recoveries, last FICO...). It makes a model look
#                 excellent in the lab and useless in practice.
# Circularity     Using the output of another risk model (grade, int_rate) as an
#                 input: our model would partly copy theirs instead of measuring
#                 applicant risk on its own.
# Selection bias  We only observe loans that were APPROVED. Applicants who were
#                 rejected have no outcome, so the model cannot learn about them
#                 (the "reject inference" problem). Results apply to the approved
#                 population only. This is a limitation, not something we fix.
# Fair lending    Variables such as location can act as proxies for protected
#                 characteristics. We exclude addr_state by design (see Part D).
#
# --- A3. Statistical concepts ------------------------------------------------
# Missing data    MCAR: missing at random, unrelated to anything.
#                 MAR: depends on other observed variables.
#                 MNAR: depends on the missing value itself.
#                 Structural missing: "not applicable" (e.g. months since last
#                 delinquency is empty when there was never one).
# Median imputation  Replace missing values by the median of the training data.
#                 Acceptable only when few values are missing (we cap at 5%).
# Winsorisation   Cap extreme values at chosen percentiles (1st and 99th) so a few
#                 outliers do not dominate the fit. Limits are learned on train.
# Skewness / log  Right-skewed variables (income, balances) have long tails; log1p
#                 makes the relationship closer to linear in the log-odds.
# Standardisation Subtract the mean and divide by the standard deviation (train
#                 values). Coefficients then mean "effect of +1 standard
#                 deviation", which makes variables comparable.
# WoE             Weight of Evidence of a group (bin): ln(%good / %bad).
#                 Positive = safer than average; negative = riskier.
# IV              Information Value: sum over bins of (%good - %bad) x WoE.
#                 How much a variable separates good from bad. Rule of thumb:
#                     < 0.02 useless | 0.02-0.10 weak | 0.10-0.30 medium |
#                     0.30-0.50 strong | > 0.50 suspicious (check for leakage).
# Spearman        Rank correlation (captures monotonic relationships, robust to
#                 outliers). |rho| >= 0.70 means two variables carry the same signal.
# VIF             Variance Inflation Factor: how much a variable's coefficient
#                 variance is inflated by correlation with the others.
#                 VIF = 1 none; > 5 concerning; > 10 severe.
# Logit model     Logistic regression: ln(p / (1 - p)) = b0 + b1 x1 + ... + bk xk.
#                 Fitted by maximum likelihood. exp(bj) is the ODDS RATIO: the
#                 multiplicative change in the odds of default per +1 unit of xj
#                 (here +1 standard deviation).
# EPV             Events per variable: bad loans divided by number of parameters.
#                 Rule of thumb: at least 10 to avoid unstable estimates.
# Separation      When a variable perfectly predicts the outcome, maximum
#                 likelihood does not converge. We check convergence.
# AUC             Probability that a random bad loan gets a higher PD than a
#                 random good loan. 0.5 = coin flip, 1.0 = perfect.
# Gini            Gini = 2 x AUC - 1. Same information rescaled to 0-1.
#                 In consumer credit, roughly 0.30-0.50 is typical/acceptable.
# KS              Kolmogorov-Smirnov statistic: maximum gap between the cumulative
#                 distributions of PD for bad and good loans. Higher = better
#                 separation (above ~0.30 is usually considered acceptable).
# Calibration     Do predicted PDs match observed default rates? A model can rank
#                 well (high AUC) and still be badly calibrated. We check:
#                   - Calibration-in-the-large: mean PD vs observed bad rate.
#                   - Calibration table by PD decile.
#                   - Calibration slope and intercept (ideal: 1 and 0).
# Sensitivity     Re-run the decision under a different assumption (here the funding
#                 cost) to see how robust the recommendation is.
#
#
# #############################################################################
# PART B. VARIABLE DICTIONARY
# (meanings follow the Lending Club data dictionary; all amounts in USD)
# #############################################################################
#
# --- Model candidates: known at the moment of application ---------------------
# loan_amnt             Amount of the loan requested by the borrower.
# term                  Number of monthly payments: 36 or 60. (Constant in our
#                       population, since we keep 36-month loans only.)
# emp_length            Years in current job ("< 1 year" ... "10+ years").
#                       Converted to the number emp_length_years (0 to 10).
# home_ownership        Housing status: RENT, OWN, MORTGAGE or OTHER.
# annual_inc            Self-reported annual income.
# verification_status   Whether Lending Club verified the reported income.
# purpose               Reason for the loan chosen by the borrower
#                       (debt consolidation, credit card, car...).
# addr_state            US state of the borrower. EXCLUDED from the model (possible
#                       proxy for protected characteristics).
# application_type      Individual or joint application. (Constant in our window.)
# dti                   Debt-to-income ratio: monthly debt payments (excluding the
#                       mortgage and this loan) divided by monthly income.
# delinq_2yrs           Number of payments 30+ days late in the past 2 years.
# earliest_cr_line      Month the borrower's first credit line was opened.
#                       Converted to credit_history_years (age of credit history).
# fico_range_low/high   Lower/upper bound of the borrower's FICO score at
#                       application. FICO is the US credit score (300-850; higher
#                       is safer). high = low + 4 by construction, so only
#                       fico_range_low is used (they carry identical information).
# inq_last_6mths        Credit inquiries by lenders in the last 6 months.
# mths_since_last_delinq   Months since the last delinquency (empty = never).
# mths_since_last_record   Months since the last public record (empty = never).
# open_acc              Number of open credit lines.
# pub_rec               Number of derogatory public records.
# revol_bal             Total balance on revolving credit (credit cards).
# revol_util            Revolving utilisation: balance / credit limit, in %.
# total_acc             Total number of credit lines ever in the credit file.
# mort_acc              Number of mortgage accounts.
# pub_rec_bankruptcies  Number of public-record bankruptcies.
# tot_cur_bal           Total current balance across all accounts.
# total_rev_hi_lim      Total revolving credit limit.
# bc_util               Balance / limit on bankcards (credit cards), in %.
# bc_open_to_buy        Unused available credit on bankcards.
# acc_open_past_24mths  Credit accounts opened in the last 24 months.
# num_tl_op_past_12m    Credit accounts opened in the last 12 months.
# pct_tl_nvr_dlq        Percentage of credit accounts never delinquent.
#
# --- Benchmark / pricing: NOT model inputs ------------------------------------
# int_rate              Interest rate Lending Club charged (set from its own grade).
# grade, sub_grade      Lending Club's own risk rating (A-G, A1-G5).
# installment           Monthly payment of the loan. With loan_amnt it gives the
#                       total interest the loan would earn if repaid in full.
#
# --- Outcome: what happened AFTER the loan was granted (never model inputs) ---
# id                    Loan identifier.
# issue_d               Month the loan was issued (defines the vintage).
# loan_status           Final or current state of the loan.
# total_pymnt           Total payments received so far.
# total_rec_prncp       Principal received so far.
# total_rec_int         Interest received so far.
# recoveries            Amount recovered after the loan was charged off.
# last_pymnt_d          Month of the last payment received.
#
# --- Variables created in this file -------------------------------------------
# emp_length_years      emp_length as a number (0 for "< 1 year", 10 for "10+ years").
# credit_history_years  Years between earliest_cr_line and issue_d.
# vintage               Year of issue_d.
# resolved              True if loan_status is a final outcome.
# bad                   1 if Charged Off / Default, 0 if Fully Paid.
# pd                    Predicted probability of default from the model.
#
#
# #############################################################################
# PART C. MODELLING ASSUMPTIONS AND WHERE THEY ARE CHECKED
# #############################################################################
#  1. Outcomes are final (no censoring)         -> section 2 (resolved share by vintage)
#  2. Risk is comparable across vintages        -> section 2 (bad rate by vintage)
#  3. Only information known at application     -> section 1 (column roles)
#  4. Few missing values, available in every    -> section 3 (missing % by vintage)
#     vintage used
#  5. No redundant predictors (multicollinearity)-> section 4 (Spearman, VIF)
#  6. Linearity in the log-odds                 -> section 5 (decile tables) and
#                                                  log transform of skewed variables
#  7. Outliers do not drive the fit             -> winsorisation (Preprocessor)
#  8. Enough events per parameter               -> section 5 (EPV)
#  9. No perfect separation / convergence       -> section 5 (convergence flag)
# 10. Independence of observations              -> NOT testable here (a borrower
#                                                  could appear more than once); stated
#                                                  as a limitation
# 11. Model works on a later period             -> section 6 (out-of-time test)
# 12. Predicted PDs are calibrated              -> section 6 (calibration)
# 13. Funding cost and LGD assumptions          -> section 9 (sensitivity to funding cost;
#                                                  LGD is fixed in the base case)
# No class re-weighting is applied: re-weighting would distort the PDs, and we
# need honest PDs for the expected-loss and profit calculations.


# #############################################################################
# PART D. CODE
# #############################################################################

# %% 0. Imports and configuration
from pathlib import Path                      # object-oriented file paths (works on Windows and Linux)

import numpy as np                            # numerical arrays and maths
import pandas as pd                           # tables (DataFrames)
import matplotlib.pyplot as plt               # charts
import statsmodels.api as sm                  # logistic regression with full statistical output
from scipy.stats import ks_2samp              # Kolmogorov-Smirnov statistic
from sklearn.metrics import roc_auc_score     # AUC
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF

pd.set_option("display.max_columns", 200)     # print all columns instead of "..."
pd.set_option("display.max_rows", 400)        # print up to 400 rows
pd.set_option("display.width", 200)           # wider console output

# ---- Paths -------------------------------------------------------------------
PATH = Path(r"C:/Users/Ricardo/Downloads/risk_analyst_project/accepted_2007_to_2018q4.csv")

# ---- Population and time window ---------------------------------------------
TARGET_TERM = "36 months"     # keep 36-month loans: all of them have reached a final outcome
DEV_YEARS = (2012, 2014)      # development vintages (inclusive)
TEST_YEAR = 2015              # out-of-time test vintage
BAD_STATUS = ["Charged Off", "Default"]   # statuses coded as bad = 1
GOOD_STATUS = ["Fully Paid"]              # statuses coded as bad = 0

# ---- Selection and preprocessing thresholds ----------------------------------
MISSING_MAX = 0.05    # a variable is eligible only if <= 5% missing in EVERY vintage 2012-2015
IV_MIN = 0.02         # below this a variable has no predictive power
IV_MAX = 0.50         # above this a variable is suspicious (possible leakage)
CORR_MAX = 0.70       # |Spearman| at or above this means two variables are redundant
VIF_MAX = 5.0         # variables above this VIF are removed one at a time
WINSOR = (0.01, 0.99) # percentiles used to cap outliers
SKEW_MAX = 1.0        # skewness above this triggers a log1p transform
RARE_LEVEL = 0.01     # categories below 1% of the training data are pooled into one "RARE" group
                      # (named RARE, not OTHER, because OTHER/other are real Lending Club categories)
IV_BINS = 10          # number of quantile bins for IV (and for decile tables)

# ---- Economic assumptions (Q2) -----------------------------------------------
LGD_BASE = 0.65            # conservative LGD for unsecured consumer credit (range 60-70%)
FUNDING_BASE = 0.05        # annual funding cost of the lender (assumption)
AVG_LIFE_YEARS = 1.5       # average outstanding life of an amortising 3-year loan (about half of 3 years)
RISK_APPETITE = 0.10       # maximum acceptable bad rate of the approved portfolio (management assumption)

# ---- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42     # fixed seed so the VIF sample is the same every run


# %% 1. DATA FOUNDATIONS: columns, loading, observations, variable types
# Goal: understand what we have before touching any model.

# The download comes as a folder named like the file, with the actual CSV inside.
if PATH.is_dir():                              # if PATH points to the folder...
    PATH = next(PATH.iterdir())                # ...use the first file inside it
print(PATH, f"-> {PATH.stat().st_size / 1e9:.2f} GB")   # confirm the file and its size

# Read ONLY the header (nrows=0): fast, and shows which columns exist before loading data.
all_columns = pd.read_csv(PATH, nrows=0).columns.tolist()
print(f"{len(all_columns)} columns in the file")

# ---- Column roles ------------------------------------------------------------
# FEATURES: information the lender has at the moment of application.
FEATURES = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "application_type", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies", "tot_cur_bal", "total_rev_hi_lim", "bc_util",
    "bc_open_to_buy", "acc_open_past_24mths", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
]
# BENCHMARK: Lending Club's own risk output and pricing. Never model inputs (circularity);
# int_rate and installment are used only to price the loan in the profit calculation.
BENCHMARK = ["int_rate", "grade", "sub_grade", "installment"]
# OUTCOME: what happened after origination. Defines the target. Never model inputs
# (data leakage).
OUTCOME = ["id", "issue_d", "loan_status", "total_pymnt", "total_rec_prncp",
           "total_rec_int", "recoveries", "last_pymnt_d"]
KEEP = FEATURES + BENCHMARK + OUTCOME          # all columns we load; the other ~100 are dropped

# Sanity check: stop immediately if any selected name does not exist in the file.
missing_cols = [c for c in KEEP if c not in all_columns]
assert not missing_cols, f"Columns not found in the file: {missing_cols}"
print(f"Kept {len(KEEP)} of {len(all_columns)} columns "
      f"(features {len(FEATURES)}, benchmark {len(BENCHMARK)}, outcome {len(OUTCOME)})")

# ---- Load once, cache as Parquet --------------------------------------------
# Parsing the full CSV takes minutes; Parquet reloads in seconds and keeps column types.
# Delete the Parquet file if KEEP changes.
PARQUET = PATH.parent / "accepted_selected.parquet"
if PARQUET.exists():                           # already cached: just read it
    data = pd.read_parquet(PARQUET)
else:                                          # first run: parse the CSV, then cache
    data = pd.read_csv(PATH, usecols=KEEP, low_memory=False)   # read only the chosen columns
    # The raw file ends with summary rows that have no loan_status: drop them.
    data = data[data["loan_status"].notna()].copy()
    data.to_parquet(PARQUET)                   # needs: pip install pyarrow

# ---- Observations and outcomes ----------------------------------------------
print(f"\nObservations: {len(data):,}")
print(f"Variables:    {data.shape[1]}")
print("\nLoan status (count):")
print(data["loan_status"].value_counts())      # shows every status so the target definition is transparent

# ---- Variable types ----------------------------------------------------------
# pandas only reports technical dtypes, and count variables (e.g. delinq_2yrs) are stored
# as float64 because they contain missing values, so the dtype alone cannot tell discrete
# from continuous. This rule-based classifier fills that gap.
DATE_COLS = ["issue_d", "earliest_cr_line", "last_pymnt_d"]
ORDINAL_COLS = ["term", "emp_length", "grade", "sub_grade"]   # categories with a natural order
ID_COLS = ["id"]
DISCRETE_MAX_UNIQUE = 200     # integer-valued numeric with <= 200 distinct values -> discrete
                              # (counts such as total_acc have ~150 distinct values)


def var_type(s):
    """Return the variable type of a pandas Series (identifier, date, categorical, discrete, continuous)."""
    if s.name in ID_COLS:                                  # identifiers carry no analytical meaning
        return "identifier"
    if s.name in DATE_COLS:                                # dates are stored as text at this stage
        return "date"
    if s.name in ORDINAL_COLS:                             # categories with a natural order
        return "categorical (ordinal)"
    if not pd.api.types.is_numeric_dtype(s):               # any other text column is nominal
        return "categorical (nominal)"
    values = s.dropna()                                    # ignore missing values for the test below
    # integer-valued AND few distinct values -> discrete; otherwise continuous
    if (values % 1 == 0).all() and values.nunique() <= DISCRETE_MAX_UNIQUE:
        return "discrete"
    return "continuous"


role = {c: "feature" for c in FEATURES}                    # role of each column in the project
role.update({c: "benchmark" for c in BENCHMARK})
role.update({c: "outcome" for c in OUTCOME})

summary = pd.DataFrame({
    "role": [role[c] for c in KEEP],
    "variable": KEEP,
    "type": [var_type(data[c]) for c in KEEP],
    "n_unique": [data[c].nunique() for c in KEEP],         # distinct values: supports the type rule
}).sort_values(["role", "type", "variable"]).reset_index(drop=True)
print("\n", summary.to_string())
print("\n", summary.groupby(["role", "type"]).size())


# %% 2. TARGET, POPULATION AND VINTAGES  (foundation of Q1 and Q2)
# Goal: define bad/good, show which vintages are usable, and measure the baseline risk.

# ---- Dates and keys ----------------------------------------------------------
# Dates are text like "Dec-2015"; convert so we can extract years and compute differences.
data["issue_d"] = pd.to_datetime(data["issue_d"], format="%b-%Y")
data["earliest_cr_line"] = pd.to_datetime(data["earliest_cr_line"], format="%b-%Y")
data["vintage"] = data["issue_d"].dt.year                  # vintage = year of origination
data["term"] = data["term"].astype(str).str.strip()        # raw values have a leading space (" 36 months")

# ---- Target ------------------------------------------------------------------
# resolved: the loan has a final outcome. Active loans (Current, Late...) do not, so they
# cannot be labelled good or bad yet.
data["resolved"] = data["loan_status"].isin(BAD_STATUS + GOOD_STATUS)
data["bad"] = data["loan_status"].isin(BAD_STATUS).astype(int)   # 1 = bad, 0 = good

# Transparency: statuses that fall in the development window but are left out of the population.
window = data[(data["term"] == TARGET_TERM) & data["vintage"].between(DEV_YEARS[0], TEST_YEAR)]
print("\nLoans in the 2012-2015, 36-month window by status (non-resolved ones are excluded):")
print(window["loan_status"].value_counts())

# ---- Bad rate by vintage and term -------------------------------------------
# pct_resolved shows maturity: a low value means many loans have no final outcome yet,
# so that vintage's bad rate is unreliable (right-censoring).
resolved_only = data[data["resolved"]]                     # only resolved loans enter the bad rate
keys = ["term", "vintage"]
vintage_table = pd.DataFrame({
    "loans": data.groupby(keys).size(),                                    # all loans in the cell
    "pct_resolved": (data.groupby(keys)["resolved"].mean() * 100).round(1),  # maturity indicator
    "resolved_loans": resolved_only.groupby(keys).size(),                  # loans with outcome
    "bad_rate_pct": (resolved_only.groupby(keys)["bad"].mean() * 100).round(2),
})
print("\nBad rate by term and vintage:")
print(vintage_table.to_string())

# Chart: bad rate by vintage, one line per term.
rates = vintage_table["bad_rate_pct"].unstack("term")      # rows = vintage, columns = term
ax = rates.plot(marker="o", figsize=(8, 4))
ax.set_title("Bad rate by vintage and term (resolved loans only)")
ax.set_xlabel("Vintage (origination year)")
ax.set_ylabel("Bad rate (%)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Population, development and test sets ----------------------------------
pop = data[(data["term"] == TARGET_TERM) & data["resolved"]].copy()   # 36-month resolved loans
dev = pop[pop["vintage"].between(DEV_YEARS[0], DEV_YEARS[1])].copy()  # development: 2012-2014
test = pop[pop["vintage"] == TEST_YEAR].copy()                        # out-of-time test: 2015
print(f"\nDevelopment {DEV_YEARS}: {len(dev):,} loans | Test {TEST_YEAR}: {len(test):,} loans")

# ---- Baseline risk -----------------------------------------------------------
# Baseline = bad rate if we approved EVERYONE. The model and the cut-off must beat it.
baseline_dev = dev["bad"].mean()
baseline_test = test["bad"].mean()
print(f"Baseline bad rate (development): {baseline_dev:.2%}")
print(f"Bad rate in the {TEST_YEAR} test vintage:   {baseline_test:.2%}")


# %% 3. VARIABLE AVAILABILITY AND MISSING VALUES  (Q1)
# Goal: keep only variables that exist consistently across all vintages we use.

def engineer(df):
    """Create model-ready variables from raw columns. Uses NO fitted parameters, so there is no leakage."""
    out = df.copy()                                        # never modify the caller's table
    # emp_length: text such as "10+ years" or "< 1 year" -> number of years.
    el = out["emp_length"].astype("string")                # nullable text type keeps missing values as NA
    years = pd.to_numeric(el.str.extract(r"(\d+)")[0], errors="coerce").astype(float)  # first number in the text
    out["emp_length_years"] = years.where(~el.str.startswith("<", na=False), 0.0)      # "< 1 year" -> 0
    # Age of credit history at application, in years (more informative than the raw date).
    out["credit_history_years"] = (out["issue_d"] - out["earliest_cr_line"]).dt.days / 365.25
    return out


NUMERIC_CANDIDATES = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "tot_cur_bal",
    "total_rev_hi_lim", "bc_util", "bc_open_to_buy", "acc_open_past_24mths",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "emp_length_years", "credit_history_years",
]
CATEGORICAL_CANDIDATES = ["home_ownership", "verification_status", "purpose"]
# Excluded a priori and why:
#   fico_range_high  identical information to fico_range_low (high = low + 4)
#   addr_state       fair-lending risk (geographic proxy) and ~50 levels inflate IV
#   term, application_type  constant in this population (checked below)
#   earliest_cr_line, emp_length  replaced by their numeric versions above
#   BENCHMARK and OUTCOME columns  circularity / leakage
ALL_CANDIDATES = NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES

pop_eng = engineer(pop)                                    # apply feature engineering to the population
dev_eng = pop_eng[pop_eng["vintage"].between(DEV_YEARS[0], DEV_YEARS[1])].copy()
test_eng = pop_eng[pop_eng["vintage"] == TEST_YEAR].copy()

# Evidence that fico_range_low and fico_range_high are redundant.
print("\nSpearman(fico_range_low, fico_range_high) =",
      round(dev_eng["fico_range_low"].corr(dev_eng["fico_range_high"], method="spearman"), 4))

# Constant variables in the development set (no information): report them.
constants = [c for c in ["term", "application_type"] if dev_eng[c].nunique(dropna=True) <= 1]
print("Constant in the development set (excluded):", constants)

# ---- Missing % by vintage ----------------------------------------------------
# Some bureau variables were only collected from certain dates, so they can be 100% missing
# in early vintages. If a variable is missing for whole vintages, median imputation would
# turn it into a "vintage detector" instead of a risk measure.
missing_by_vintage = (pop_eng.loc[pop_eng["vintage"] <= TEST_YEAR, ALL_CANDIDATES]
                      .isna().groupby(pop_eng.loc[pop_eng["vintage"] <= TEST_YEAR, "vintage"])
                      .mean() * 100).round(1).T            # rows = variables, columns = vintages
print("\nMissing % by vintage (36-month resolved loans):")
print(missing_by_vintage.to_string())

# Eligibility rule: <= MISSING_MAX missing in EVERY vintage of the development+test window.
window_missing = missing_by_vintage.loc[:, DEV_YEARS[0]:TEST_YEAR]    # columns 2012..2015
worst_missing = window_missing.max(axis=1)                 # worst vintage for each variable
eligible = worst_missing[worst_missing <= MISSING_MAX * 100].index.tolist()
excluded_missing = worst_missing[worst_missing > MISSING_MAX * 100]
print(f"\nExcluded for missing > {MISSING_MAX:.0%} in some vintage:")
print(excluded_missing.round(1).to_string())
# Note: mths_since_last_delinq / mths_since_last_record are structurally empty (never happened).
# A proper treatment needs a dedicated "never" category (WoE with a MISSING bin); out of scope here.

eligible_num = [c for c in NUMERIC_CANDIDATES if c in eligible]
eligible_cat = [c for c in CATEGORICAL_CANDIDATES if c in eligible]


# %% 4. VARIABLE SELECTION: IV, REDUNDANCY (Spearman), MULTICOLLINEARITY (VIF)  (Q1)
# Everything here is computed on the DEVELOPMENT set only; the test set is never used to choose variables.

def woe_table(x, y, bins=IV_BINS):
    """Weight of Evidence table for one variable.
    Numeric variables are cut into quantile bins; missing values form their own 'MISSING' bin;
    categorical and low-cardinality variables use their own values as groups."""
    if pd.api.types.is_numeric_dtype(x) and x.dropna().nunique() > bins:
        grp = pd.Series("MISSING", index=x.index, dtype=object)      # start with every row as MISSING
        valid = x.notna()                                            # rows with a value
        ids = pd.qcut(x[valid], q=bins, labels=False, duplicates="drop")   # quantile bin id 0..k-1
        grp.loc[valid] = ids.astype(int).astype(str).str.zfill(2).values   # zero-padded id keeps bins sorted
    else:
        grp = x.astype(object).where(x.notna(), "MISSING").astype(str)     # groups = distinct values
    tab = pd.DataFrame({"grp": grp.values, "bad": y.values})        # pair each group with its outcome
    t = tab.groupby("grp")["bad"].agg(total="count", bad="sum")     # loans and bad loans per group
    t["good"] = t["total"] - t["bad"]
    k = len(t)                                                       # number of groups
    # Add 0.5 to every cell so empty groups do not produce log(0) (standard adjustment).
    pct_bad = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5 * k)          # share of all bads in the group
    pct_good = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * k)       # share of all goods in the group
    t["woe"] = np.log(pct_good / pct_bad)                            # >0 safer than average, <0 riskier
    t["iv"] = (pct_good - pct_bad) * t["woe"]                        # contribution of the group to IV
    return t


def iv_strength(iv):
    """Translate an IV value into the usual verbal scale."""
    if iv < 0.02:
        return "useless (<0.02)"
    if iv < 0.10:
        return "weak (0.02-0.10)"
    if iv < 0.30:
        return "medium (0.10-0.30)"
    if iv <= IV_MAX:
        return "strong (0.30-0.50)"
    return "suspicious (>0.50)"


# ---- Step 4.1: Information Value ranking ------------------------------------
iv_rows = []
for c in eligible_num + eligible_cat:
    iv = woe_table(dev_eng[c], dev_eng["bad"])["iv"].sum()           # total IV = sum over groups
    iv_rows.append({"variable": c, "type": "numeric" if c in eligible_num else "categorical",
                    "iv": round(iv, 4), "strength": iv_strength(iv)})
iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
print("\nInformation Value ranking (development set):")
print(iv_table.to_string())

suspicious = iv_table.loc[iv_table["iv"] > IV_MAX, "variable"].tolist()
if suspicious:
    print("\nWARNING: IV above", IV_MAX, "-> check for leakage before using:", suspicious)
usable = iv_table[(iv_table["iv"] >= IV_MIN) & (iv_table["iv"] <= IV_MAX)]   # keep predictive, non-suspicious
usable_num = [c for c in usable["variable"] if c in eligible_num]            # already sorted by IV (desc)
usable_cat = [c for c in usable["variable"] if c in eligible_cat]

# ---- Step 4.2: remove redundant numeric variables (Spearman) ----------------
# Greedy rule: walk down the IV ranking; keep a variable only if it is not highly correlated
# with one already kept. The stronger variable (higher IV) always wins.
corr = dev_eng[usable_num].corr(method="spearman")                   # rank correlation, pairwise-complete
kept_num, dropped_corr = [], []
for c in usable_num:
    partner = next((k for k in kept_num if abs(corr.loc[c, k]) >= CORR_MAX), None)   # first conflicting kept variable
    if partner is None:
        kept_num.append(c)
    else:
        dropped_corr.append((c, partner, round(corr.loc[c, partner], 3)))
print("\nDropped for redundancy (variable, kept partner, Spearman):", dropped_corr)

# ---- Step 4.3: multicollinearity (VIF), removing the worst offender each round ---
# VIF is computed on the preprocessed numeric matrix (imputed and standardised on train), and
# on a sample for speed (VIF describes the correlation structure, which a sample reproduces).

class Preprocessor:
    """Learns preprocessing parameters on the development set and applies them unchanged to any other set.
    Fitting on development only and re-using the parameters on test is what avoids leakage.
    Numeric steps: winsorise -> log1p if skewed -> median imputation -> standardisation.
    Categorical steps: pool rare levels into RARE -> one-hot encoding (most frequent level = reference)."""

    def __init__(self, numeric, categorical):
        self.numeric = list(numeric)            # numeric variables to process
        self.categorical = list(categorical)    # categorical variables to process
        self.params = {}                        # learned numeric parameters
        self.keep_levels = {}                   # categories frequent enough to keep (per variable)
        self.dummy_levels = {}                  # categories that get a dummy column (per variable)
        self.reference = {}                     # reference category (per variable)

    def fit(self, df):
        for c in self.numeric:
            s = df[c].astype(float)                                   # work on floats
            lo, hi = s.quantile(WINSOR[0]), s.quantile(WINSOR[1])     # winsorisation limits from train
            s = s.clip(lo, hi)                                        # cap outliers
            use_log = bool(s.min() >= 0 and s.skew() > SKEW_MAX)     # log only for non-negative, right-skewed
            if use_log:
                s = np.log1p(s)                                       # log(1 + x), safe at zero
            median = s.median()                                       # imputation value
            s = s.fillna(median)                                      # fill missing values
            std = s.std()
            self.params[c] = {"lo": lo, "hi": hi, "log": use_log, "median": median,
                              "mean": s.mean(), "std": std if std > 0 else 1.0}   # guard against zero variance
        for c in self.categorical:
            s = df[c].astype(object).where(df[c].notna(), "MISSING")  # missing becomes its own category
            freq = s.value_counts(normalize=True)                     # share of each category
            keep = freq[freq >= RARE_LEVEL].index.tolist()            # categories above the rarity threshold
            grouped = s.where(s.isin(keep), "RARE")                   # rare ones -> RARE
            counts = grouped.value_counts()                           # sorted by frequency
            self.keep_levels[c] = keep
            self.reference[c] = counts.index[0]                       # most frequent level is the baseline
            self.dummy_levels[c] = [lvl for lvl in counts.index if lvl != counts.index[0]]
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)                            # result keeps the original row index
        for c in self.numeric:
            p = self.params[c]
            s = df[c].astype(float).clip(p["lo"], p["hi"])            # same caps as in training
            if p["log"]:
                s = np.log1p(s)                                       # same transform as in training
            s = s.fillna(p["median"])                                 # same imputation value
            out[c] = (s - p["mean"]) / p["std"]                       # same standardisation
        for c in self.categorical:
            s = df[c].astype(object).where(df[c].notna(), "MISSING")
            s = s.where(s.isin(self.keep_levels[c]), "RARE")          # unseen/rare levels -> RARE
            for lvl in self.dummy_levels[c]:
                out[f"{c}={lvl}"] = (s == lvl).astype(float)          # 1 if the loan has this category
        return out


vif_sample = dev_eng.sample(min(len(dev_eng), 150_000), random_state=RANDOM_STATE)   # sample for speed
while True:
    X_vif = Preprocessor(kept_num, []).fit(dev_eng).transform(vif_sample)          # numeric matrix, imputed + standardised
    X_vif = sm.add_constant(X_vif, has_constant="add")                              # VIF needs the intercept column
    vifs = pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])],
                     index=kept_num)                                                # skip column 0 (the constant)
    worst = vifs.idxmax()                                                           # variable with the highest VIF
    if vifs[worst] <= VIF_MAX or len(kept_num) <= 1:
        break                                                                       # all VIFs acceptable
    print(f"Removing {worst} (VIF = {vifs[worst]:.2f})")
    kept_num.remove(worst)                                                          # drop it and recompute

print("\nFinal VIFs:")
print(vifs.round(2).sort_values(ascending=False).to_string())

final_num = kept_num                         # numeric variables in the model
final_cat = usable_cat                       # categorical variables in the model
print("\nFinal numeric variables:", final_num)
print("Final categorical variables:", final_cat)


# %% 5. MODEL: LOGISTIC REGRESSION  (Q1)
# statsmodels (not scikit-learn) because it gives standard errors, p-values and confidence
# intervals. No penalty and no class weights: we want unbiased coefficients and honest PDs.

pre = Preprocessor(final_num, final_cat).fit(dev_eng)          # learn preprocessing on development only
X_dev = pre.transform(dev_eng)                                 # development design matrix
X_test = pre.transform(test_eng)                               # test matrix, SAME parameters
y_dev = dev_eng["bad"].to_numpy()                              # development outcome (0/1)
y_test = test_eng["bad"].to_numpy()                            # test outcome (0/1)

logit = sm.Logit(y_dev, sm.add_constant(X_dev, has_constant="add"))   # intercept + predictors
result = logit.fit(disp=0, maxiter=200)                        # maximum likelihood; disp=0 silences iterations
print("\nConverged:", result.mle_retvals.get("converged"))     # assumption 9: convergence / no perfect separation

# Events per variable (assumption 8): bad loans divided by estimated parameters.
epv = y_dev.sum() / len(result.params)
print(f"Events per parameter (EPV): {epv:,.0f}  (rule of thumb: >= 10)")

# Coefficient table. Variables are standardised, so odds ratios are per +1 standard deviation
# (for dummies: versus the reference category).
ci = result.conf_int()                                          # 95% confidence intervals of the coefficients
coef_table = pd.DataFrame({
    "coef": result.params,
    "std_err": result.bse,
    "z": result.tvalues,
    "p_value": result.pvalues,
    "odds_ratio": np.exp(result.params),                        # exp(coef): multiplicative change in odds
    "or_ci_low": np.exp(ci[0]),
    "or_ci_high": np.exp(ci[1]),
})
print("\nLogit coefficients (sorted by |z|):")
print(coef_table.reindex(coef_table["z"].abs().sort_values(ascending=False).index).round(4).to_string())

# Predicted PDs for development and test.
pd_dev = np.asarray(result.predict(sm.add_constant(X_dev, has_constant="add")))
pd_test = np.asarray(result.predict(sm.add_constant(X_test, has_constant="add")))
test_eng["pd"] = pd_test                                       # keep the test PDs next to the loans (sections 6, 8, 9)

# ---- Linearity in the log-odds (assumption 6) --------------------------------
# For each numeric variable: bad rate by decile of the RAW variable. A roughly monotonic
# pattern supports using it linearly; a U-shape or plateaus would call for binning (WoE).
def decile_table(x, y, q=IV_BINS):
    valid = x.notna()
    bucket = pd.qcut(x[valid], q=q, labels=False, duplicates="drop")        # decile id per loan
    t = pd.DataFrame({"x": x[valid], "bad": y[valid], "bucket": bucket})
    return t.groupby("bucket").agg(x_min=("x", "min"), x_max=("x", "max"),
                                   n=("bad", "size"), bad_rate=("bad", "mean"))


print("\nLinearity check: Spearman between decile order and bad rate (+1/-1 = monotonic):")
for c in final_num:
    g = decile_table(dev_eng[c], dev_eng["bad"])
    mono = g["bad_rate"].corr(pd.Series(g.index, index=g.index), method="spearman")
    print(f"  {c:<25} monotonicity = {mono:+.2f}   bad rate from {g['bad_rate'].iloc[0]:.1%} to {g['bad_rate'].iloc[-1]:.1%}")


# %% 6. VALIDATION OUT OF TIME  (Q1)
# Discrimination (AUC, Gini, KS) and calibration, on development and on the 2015 test set.

def ks_stat(y, p):
    """KS = largest gap between the PD distributions of bad and good loans."""
    return ks_2samp(p[y == 1], p[y == 0]).statistic


auc_dev, auc_test = roc_auc_score(y_dev, pd_dev), roc_auc_score(y_test, pd_test)
metrics = pd.DataFrame({
    "AUC": [auc_dev, auc_test],
    "Gini": [2 * auc_dev - 1, 2 * auc_test - 1],               # Gini = 2 x AUC - 1
    "KS": [ks_stat(y_dev, pd_dev), ks_stat(y_test, pd_test)],
    "Bad rate": [y_dev.mean(), y_test.mean()],                 # observed
    "Mean PD": [pd_dev.mean(), pd_test.mean()],                # predicted (calibration-in-the-large)
}, index=[f"Development {DEV_YEARS}", f"Test {TEST_YEAR}"])
print("\nDiscrimination and calibration-in-the-large:")
print(metrics.round(4).to_string())
# A large drop from development to test would signal overfitting or a change in the population.

# ---- Calibration -------------------------------------------------------------
# Calibration slope and intercept: regress the outcome on logit(PD). Ideal: intercept 0, slope 1.
# slope < 1 means PDs are too spread out (overconfident); slope > 1 means too compressed;
# an intercept above 0 means observed defaults are higher than predicted.
p_clip = np.clip(pd_test, 1e-6, 1 - 1e-6)                      # avoid log(0)
logit_pd = np.log(p_clip / (1 - p_clip))                       # PD on the log-odds scale
cal_fit = sm.Logit(y_test, sm.add_constant(pd.DataFrame({"logit_pd": logit_pd}))).fit(disp=0)
print(f"\nCalibration intercept = {cal_fit.params.iloc[0]:+.3f} (ideal 0) | slope = {cal_fit.params.iloc[1]:.3f} (ideal 1)")

# Calibration table by PD decile on the test set: predicted vs observed bad rate.
test_eng["pd_decile"] = pd.qcut(test_eng["pd"], 10, labels=False, duplicates="drop") + 1   # 1 = safest
cal_table = test_eng.groupby("pd_decile").agg(loans=("bad", "size"), mean_pd=("pd", "mean"), observed_bad_rate=("bad", "mean"))
print("\nCalibration by PD decile (test):")
print(cal_table.round(4).to_string())

plt.figure(figsize=(5, 5))
plt.plot(cal_table["mean_pd"], cal_table["observed_bad_rate"], marker="o", label="Model")
top = float(max(cal_table["mean_pd"].max(), cal_table["observed_bad_rate"].max()))
plt.plot([0, top], [0, top], "--", color="grey", label="Perfect calibration")
plt.xlabel("Mean predicted PD")
plt.ylabel("Observed bad rate")
plt.title(f"Calibration by decile (test {TEST_YEAR})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 7. ANSWER TO QUESTION 1  (written from the numbers, as associations)
# Write the narrative answer in the notebook AFTER reading the tables above. Rules:
#   - Use the sign and size of the odds ratios (risk increases when OR > 1, decreases when OR < 1).
#   - Describe ASSOCIATIONS ("higher dti is associated with higher default risk"), not causes.
#   - Quote Gini, KS and calibration, and compare with the baseline bad rate.
drivers = coef_table.drop(index="const")
drivers = drivers.reindex(drivers["z"].abs().sort_values(ascending=False).index)   # strongest first
print("\nRisk drivers, strongest to weakest (OR per +1 SD):")
for name, row in drivers.iterrows():
    direction = "raises" if row["odds_ratio"] > 1 else "lowers"
    print(f"  {name:<30} OR = {row['odds_ratio']:.2f}  ({direction} default odds)   p = {row['p_value']:.3g}")


# %% 8. APPROVAL RATE VS BAD RATE  (Q2)
# Cut-offs are defined by percentiles of the PD ranking (approve the X% safest applicants),
# which stays meaningful whatever the baseline risk is. We report the PD value of each cut-off.
ranked = test_eng.sort_values("pd").reset_index(drop=True)     # safest first
shares = np.round(np.arange(0.10, 1.0001, 0.05), 2)            # approval rates 10%, 15%, ..., 100%

rows = []
for q in shares:
    n_app = int(round(q * len(ranked)))                        # number of approved applicants
    app = ranked.iloc[:n_app]                                  # the n_app lowest-PD applicants
    rows.append({"approval_rate": q, "n_approved": n_app,
                 "pd_cutoff": app["pd"].max(),                 # highest PD still approved
                 "mean_pd": app["pd"].mean(),                  # predicted bad rate of the approved
                 "bad_rate": app["bad"].mean(),                # OBSERVED bad rate of the approved
                 "bad_rate_vs_baseline": app["bad"].mean() / baseline_test})
policy = pd.DataFrame(rows)
print(f"\nApproval rate vs bad rate (test {TEST_YEAR}, baseline bad rate {baseline_test:.2%}):")
print(policy.round(4).to_string(index=False))

# Risk appetite: the largest approval rate whose observed bad rate stays within the ceiling.
within = policy[policy["bad_rate"] <= RISK_APPETITE]
if len(within):
    best_app = within.iloc[-1]
    print(f"\nRisk appetite {RISK_APPETITE:.0%}: approve up to {best_app['approval_rate']:.0%} "
          f"(PD cut-off {best_app['pd_cutoff']:.3f}, bad rate {best_app['bad_rate']:.2%})")
else:
    print(f"\nNo approval rate keeps the bad rate within {RISK_APPETITE:.0%} (baseline {baseline_test:.2%}).")

plt.figure(figsize=(7, 4))
plt.plot(policy["approval_rate"] * 100, policy["bad_rate"] * 100, marker="o", label="Observed bad rate of approved")
plt.axhline(baseline_test * 100, color="grey", linestyle="--", label="Baseline (approve everyone)")
plt.axhline(RISK_APPETITE * 100, color="red", linestyle=":", label="Risk appetite")
plt.xlabel("Approval rate (%)")
plt.ylabel("Bad rate (%)")
plt.title("Trade-off between volume and risk")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 9. EXPECTED PROFIT BY CUT-OFF AND SENSITIVITY TO FUNDING COST  (Q2)
# Expected profit per loan, computed BEFORE the outcome is known:
#     (1 - PD) x interest income  -  PD x LGD x EAD  -  funding cost
#   interest income = installment x 36 - loan amount (interest if every instalment is paid)
#   EAD             = loan amount
#   LGD             = LGD_BASE (fixed assumption)
#   funding cost    = funding rate x loan amount x AVG_LIFE_YEARS
def expected_profit(df, funding):
    """Expected profit per loan for a given annual funding cost. Needs a column 'pd' in df."""
    p = df["pd"]                                               # predicted probability of default
    ead = df["loan_amnt"]                                      # exposure at default = original amount
    interest = df["installment"] * 36 - ead                    # contractual interest over 36 payments
    return (1 - p) * interest - p * LGD_BASE * ead - funding * ead * AVG_LIFE_YEARS


# Approval grid with 0% added: "approve nobody" (profit 0) must be an option, otherwise the best
# cut-off would be forced to be positive even when lending loses money at every cut-off.
shares_ext = np.concatenate([[0.0], shares])
counts = np.array([int(round(q * len(ranked))) for q in shares_ext])   # approved loans at each rate
bad_rate_by_q = np.concatenate([[np.nan], policy["bad_rate"].to_numpy()])   # observed bad rate at each rate (from section 8)

funding_scenarios = {"Funding 3%": 0.03, f"Funding {FUNDING_BASE:.0%} (base)": FUNDING_BASE, "Funding 7%": 0.07}

rows, curves = [], {}
for name, funding in funding_scenarios.items():
    per_loan = expected_profit(ranked, funding)                            # expected profit of every loan, safest first
    totals = np.concatenate([[0.0], per_loan.cumsum().to_numpy()])[counts]  # total profit if we approve the first n loans
    curves[name] = totals
    i = int(np.argmax(totals))                                             # approval rate with the highest expected profit
    n_i = counts[i]
    rows.append({"scenario": name, "best_approval_rate": shares_ext[i],
                 "pd_cutoff": ranked["pd"].iloc[n_i - 1] if n_i > 0 else np.nan,
                 "bad_rate": bad_rate_by_q[i],
                 "expected_profit_M": totals[i] / 1e6,
                 "verdict": "NO PROFITABLE CUT-OFF" if i == 0 else ""})
print("\nProfit-maximising approval rate by funding cost (0% = lending is not profitable at any cut-off):")
print(pd.DataFrame(rows).round(4).to_string(index=False))

curve_table = pd.DataFrame({"approval_rate": shares_ext, "n_approved": counts,
                            **{k: v / 1e6 for k, v in curves.items()}})
print("\nExpected profit by approval rate (USD million, test vintage):")
print(curve_table.round(2).to_string(index=False))

# Recommendation: maximise expected profit (base funding) among approval rates that respect the risk appetite.
base_name = f"Funding {FUNDING_BASE:.0%} (base)"
allowed = np.isnan(bad_rate_by_q) | (bad_rate_by_q <= RISK_APPETITE)       # 0% (nan) is always allowed
constrained = np.where(allowed, curves[base_name], -np.inf)                # forbid rates above the appetite
j = int(np.argmax(constrained))
if j == 0:
    print(f"\nNo approval rate is both profitable and within the {RISK_APPETITE:.0%} risk appetite.")
else:
    print(f"\nRecommended (base funding, bad rate <= {RISK_APPETITE:.0%}): approve {shares_ext[j]:.0%} of applicants, "
          f"PD cut-off {ranked['pd'].iloc[counts[j] - 1]:.3f}, bad rate {bad_rate_by_q[j]:.2%}, "
          f"expected profit {curves[base_name][j] / 1e6:,.1f} M")

plt.figure(figsize=(8, 4.5))
for name, totals in curves.items():
    plt.plot(shares_ext * 100, totals / 1e6, label=name)
plt.axhline(0, color="red", linewidth=0.8)
plt.xlabel("Approval rate (%)")
plt.ylabel("Expected profit (USD million)")
plt.title(f"Expected profit by approval rate and funding cost (test {TEST_YEAR})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 10. LIMITATIONS (to report with the recommendation)
# 1. Selection bias: only approved loans are observed; conclusions apply to that population
#    (reject inference is not attempted).
# 2. Simplified costs: expected profit assumes contractual interest, a constant LGD applied to the
#    original amount, a flat funding cost over a fixed average life, and no operating costs.
#    Comparing it with the cash actually received would test these assumptions (not done here),
#    so profit levels should be read as indicative and the cut-off as sensitive to them.
# 3. A single credit cycle: 2012-2015 was a benign, fast-growing period; the PDs may not hold in
#    a downturn, and the observed bad rate of the test vintage can differ from the predicted one
#    (see the calibration section).
# 4. Observations may not be independent (a borrower can hold several loans).
# 5. Only 36-month loans are covered; the 60-month product needs its own analysis.
# 6. Predictive association is not causation: coefficients describe who tends to default,
#    not why.
