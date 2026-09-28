"""
DineIQ Analytics - Independent Statistical & Econometric Modeling (SRS Step 13)
Implemented using SciPy and Statsmodels:
 1. Price Elasticity of Demand (Log-Log OLS Regression via Statsmodels)
 2. Two-Sample Independent t-test: Weekend vs Weekday Spend (SciPy)
 3. One-Way ANOVA: Revenue across Restaurant Location Tiers (SciPy)
 4. Chi-Square Test of Independence: Order Channel vs Payment Method (SciPy)
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

def analyze_price_elasticity(orders_df: pd.DataFrame, order_items_df: pd.DataFrame) -> dict:
    """
    Fits a log-log econometric demand model using Statsmodels OLS:
    ln(Quantity) = beta_0 + beta_1 * ln(Unit_Price) + beta_2 * Is_Promo + epsilon
    beta_1 is the direct Price Elasticity of Demand (Ed).
    """
    print("[Statsmodels] Estimating Price Elasticity of Demand via OLS...")
    merged = pd.merge(
        order_items_df[["order_id", "item_id", "quantity", "unit_price", "item_total"]],
        orders_df[["order_id", "promotion_id", "order_date"]],
        on="order_id",
        how="inner"
    )

    merged["has_promo"] = (merged["promotion_id"].notnull() & (merged["promotion_id"] != "")).astype(int)
    merged = merged[(merged["quantity"] > 0) & (merged["unit_price"] > 0)].copy()

    merged["week"] = pd.to_datetime(merged["order_date"]).dt.to_period("W").astype(str)
    weekly_item_sales = merged.groupby(["item_id", "week"]).agg(
        weekly_qty=("quantity", "sum"),
        avg_price=("unit_price", "mean"),
        promo_rate=("has_promo", "mean")
    ).reset_index()

    weekly_item_sales = weekly_item_sales[(weekly_item_sales["weekly_qty"] > 0) & (weekly_item_sales["avg_price"] > 0)]

    weekly_item_sales["ln_quantity"] = np.log(weekly_item_sales["weekly_qty"])
    weekly_item_sales["ln_price"] = np.log(weekly_item_sales["avg_price"])

    X = weekly_item_sales[["ln_price", "promo_rate"]]
    X = sm.add_constant(X)
    y = weekly_item_sales["ln_quantity"]

    ols_model = sm.OLS(y, X).fit()
    elasticity_coef = float(ols_model.params["ln_price"])
    elasticity_pvalue = float(ols_model.pvalues["ln_price"])
    promo_coef = float(ols_model.params["promo_rate"])
    r_squared = float(ols_model.rsquared)

    abs_e = abs(elasticity_coef)
    if abs_e > 1.05:
        category = "Price Elastic (|Ed| > 1) - Highly sensitive to price increases"
    elif abs_e < 0.95:
        category = "Price Inelastic (|Ed| < 1) - Low sensitivity; pricing power exists"
    else:
        category = "Unitary Elastic (|Ed| ~ 1) - Revenue remains relatively stable"

    return {
        "model_type": "Log-Log Econometric Demand Regression (Statsmodels OLS)",
        "price_elasticity_coefficient": round(elasticity_coef, 4),
        "elasticity_category": category,
        "p_value": round(elasticity_pvalue, 6),
        "is_statistically_significant": elasticity_pvalue < 0.05,
        "promo_lift_coefficient": round(promo_coef, 4),
        "r_squared": round(r_squared, 4),
        "f_pvalue": round(float(ols_model.f_pvalue), 6),
        "total_observations": int(ols_model.nobs),
        "ols_summary_str": str(ols_model.summary())
    }

def perform_hypothesis_tests(orders_df: pd.DataFrame, restaurants_df: pd.DataFrame) -> dict:
    """
    Executes 3 foundational inferential tests using SciPy:
    1. Independent 2-Sample t-test: Weekend vs Weekday Spend
    2. One-Way ANOVA: Revenue across Location Tiers
    3. Chi-Square Test of Independence: Channel vs Payment Method
    """
    print("[SciPy] Conducting rigorous hypothesis testing & inferential statistics...")

    orders = orders_df.copy()
    orders["day_of_week"] = pd.to_datetime(orders["order_date"]).dt.dayofweek
    orders["is_weekend"] = orders["day_of_week"].isin([5, 6])               

    weekend_spend = orders[orders["is_weekend"]]["total_amount"].dropna()
    weekday_spend = orders[~orders["is_weekend"]]["total_amount"].dropna()

    t_stat, t_pval = stats.ttest_ind(weekend_spend, weekday_spend, equal_var=False)

    orders_with_rest = pd.merge(orders, restaurants_df[["location_id", "location_tier"]], on="location_id", how="inner")
    tier_groups = [group["total_amount"].dropna().values for _, group in orders_with_rest.groupby("location_tier")]
    f_stat, f_pval = stats.f_oneway(*tier_groups)

    contingency_table = pd.crosstab(orders["order_type"], orders["payment_method"])
    chi2_stat, chi2_pval, dof, _ = stats.chi2_contingency(contingency_table)

    spend_skew = float(stats.skew(orders["total_amount"].dropna()))
    spend_kurt = float(stats.kurtosis(orders["total_amount"].dropna()))

    return {
        "two_sample_ttest_weekend_vs_weekday": {
            "test_name": "Welch's Two-Sample Independent t-test",
            "null_hypothesis": "Mean ticket spend on weekends equals weekday spend",
            "weekend_mean": round(float(weekend_spend.mean()), 2),
            "weekday_mean": round(float(weekday_spend.mean()), 2),
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(t_pval),
            "reject_null_at_5pct": bool(t_pval < 0.05)
        },
        "anova_location_tiers": {
            "test_name": "One-Way Analysis of Variance (ANOVA)",
            "null_hypothesis": "Mean order amounts are identical across Urban, Suburban, and Downtown tiers",
            "f_statistic": round(float(f_stat), 4),
            "p_value": float(f_pval),
            "reject_null_at_5pct": bool(f_pval < 0.05)
        },
        "chi_square_channel_payment": {
            "test_name": "Pearson's Chi-Square Test of Independence",
            "null_hypothesis": "Customer fulfillment channel and payment method are statistically independent",
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": float(chi2_pval),
            "degrees_of_freedom": int(dof),
            "reject_null_at_5pct": bool(chi2_pval < 0.05)
        },
        "distribution_metrics": {
            "spend_skewness": round(spend_skew, 4),
            "spend_kurtosis": round(spend_kurt, 4)
        }
    }
