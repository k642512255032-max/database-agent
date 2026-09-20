"""Statistical models the agent can run on query results.

Each function returns a dict: {"method", "tables": {name: DataFrame}, "facts": {...},
"interpretation": str}. Interpretations are deterministic (rule-based), so the
explanation never depends on the LLM guessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

METHODS = {
    "describe": "Descriptive statistics (mean, std, quartiles, counts) for columns",
    "correlation": "Pearson/Spearman correlation matrix between numeric columns",
    "group_summary": "Mean/median/count of a numeric target per group",
    "ttest": "Welch t-test: does a numeric target differ between TWO groups?",
    "anova": "One-way ANOVA: does a numeric target differ across 3+ groups?",
    "chi_square": "Chi-square test of independence between two categorical columns",
    "normality": "Shapiro-Wilk normality test for a numeric column",
    "linear_regression": "OLS linear regression: numeric target ~ feature columns",
    "logistic_regression": "Logistic regression: binary target ~ feature columns",
}


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    """MySQL DECIMAL comes back as objects; convert anything mostly-numeric to float."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().mean() >= 0.95 and out[c].notna().any():
                out[c] = conv
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _sig(p: float) -> str:
    return "statistically significant" if p < ALPHA else "not statistically significant"


def run(method: str, df: pd.DataFrame, target: str | None = None, group: str | None = None,
        columns: list[str] | None = None) -> dict:
    df = numericize(df)
    fn = globals().get(f"_m_{method}")
    if fn is None:
        raise ValueError(f"Unknown statistical method '{method}'. Options: {list(METHODS)}")
    result = fn(df, target=target, group=group, columns=columns or [])
    result["method"] = method
    return result


# ------------------------------------------------------------------ methods
def _m_describe(df, columns, **_):
    cols = columns or list(df.columns)
    desc = df[cols].describe(include="all").T.reset_index().rename(columns={"index": "column"})
    return {"tables": {"summary": desc}, "facts": {"rows": len(df), "columns": cols},
            "interpretation": f"Descriptive statistics for {len(cols)} column(s) over {len(df)} rows."}


def _m_correlation(df, columns, **_):
    cols = [c for c in (columns or numeric_columns(df)) if c in numeric_columns(df)]
    if len(cols) < 2:
        raise ValueError("Correlation needs at least two numeric columns.")
    pear = df[cols].corr(method="pearson")
    spear = df[cols].corr(method="spearman")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = df[[a, b]].dropna()
            r, p = stats.pearsonr(sub[a], sub[b]) if len(sub) > 2 else (np.nan, np.nan)
            pairs.append({"a": a, "b": b, "pearson_r": round(r, 4), "p_value": round(p, 5),
                          "spearman_rho": round(spear.loc[a, b], 4)})
    pairs_df = pd.DataFrame(pairs).sort_values("pearson_r", key=np.abs, ascending=False)
    top = pairs_df.iloc[0]
    strength = abs(top.pearson_r)
    word = "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak"
    return {"tables": {"pairs": pairs_df, "pearson_matrix": pear.round(4).reset_index()},
            "facts": {"strongest_pair": [top.a, top.b], "r": top.pearson_r, "p": top.p_value},
            "interpretation": f"Strongest relationship: {top.a} vs {top.b}, r={top.pearson_r:+.3f} "
                              f"({word}, {'positive' if top.pearson_r > 0 else 'negative'}), "
                              f"p={top.p_value:.4g} -> {_sig(top.p_value)}. Correlation is not causation."}


def _m_group_summary(df, target, group, **_):
    _need(df, target, group)
    g = df.groupby(group)[target].agg(["count", "mean", "median", "std"]).round(4).reset_index()
    g = g.sort_values("mean", ascending=False)
    return {"tables": {"by_group": g}, "facts": {"highest_group": str(g.iloc[0][group]), "highest_mean": float(g.iloc[0]["mean"])},
            "interpretation": f"'{g.iloc[0][group]}' has the highest mean {target} ({g.iloc[0]['mean']:.4g}); "
                              f"'{g.iloc[-1][group]}' the lowest ({g.iloc[-1]['mean']:.4g})."}


def _m_ttest(df, target, group, **_):
    _need(df, target, group)
    groups = df[group].dropna().unique()
    if len(groups) != 2:
        top2 = df[group].value_counts().index[:2]
        note = f" (column had {len(groups)} groups; compared the two largest: {list(top2)})"
        groups = top2
    else:
        note = ""
    a = df.loc[df[group] == groups[0], target].dropna().astype(float)
    b = df.loc[df[group] == groups[1], target].dropna().astype(float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
    d = (a.mean() - b.mean()) / pooled
    table = pd.DataFrame({group: [groups[0], groups[1]], "n": [len(a), len(b)],
                          "mean": [a.mean(), b.mean()], "std": [a.std(), b.std()]}).round(4)
    return {"tables": {"groups": table}, "facts": {"t": round(float(t), 4), "p": float(p), "cohens_d": round(float(d), 3)},
            "interpretation": f"Mean {target}: {groups[0]}={a.mean():.4g} vs {groups[1]}={b.mean():.4g}. "
                              f"Welch t={t:.3f}, p={p:.4g} -> the difference is {_sig(p)} at alpha={ALPHA}; "
                              f"effect size Cohen's d={d:.2f}{note}."}


def _m_anova(df, target, group, **_):
    _need(df, target, group)
    samples = [s[target].dropna().astype(float) for _, s in df.groupby(group) if s[target].notna().sum() > 1]
    if len(samples) < 2:
        raise ValueError("ANOVA needs at least two groups with data.")
    f, p = stats.f_oneway(*samples)
    table = df.groupby(group)[target].agg(["count", "mean", "std"]).round(4).reset_index()
    return {"tables": {"groups": table}, "facts": {"F": round(float(f), 4), "p": float(p)},
            "interpretation": f"One-way ANOVA across {len(samples)} groups of {group}: F={f:.3f}, p={p:.4g} -> "
                              f"differences in mean {target} are {_sig(p)}."}


def _m_chi_square(df, target, group, **_):
    _need(df, target, group)
    ct = pd.crosstab(df[group], df[target])
    chi2, p, dof, _exp = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0
    return {"tables": {"contingency": ct.reset_index()},
            "facts": {"chi2": round(float(chi2), 4), "dof": int(dof), "p": float(p), "cramers_v": round(float(v), 3)},
            "interpretation": f"Chi-square={chi2:.3f} (dof={dof}), p={p:.4g}: the association between {group} and "
                              f"{target} is {_sig(p)}; Cramer's V={v:.2f}."}


def _m_normality(df, target, **_):
    _need(df, target)
    x = df[target].dropna().astype(float)
    x = x.sample(5000, random_state=0) if len(x) > 5000 else x
    w, p = stats.shapiro(x)
    return {"tables": {}, "facts": {"W": round(float(w), 4), "p": float(p), "skew": round(float(x.skew()), 3)},
            "interpretation": f"Shapiro-Wilk W={w:.3f}, p={p:.4g}: {target} "
                              f"{'does NOT look normally distributed' if p < ALPHA else 'is consistent with a normal distribution'}."}


def _m_linear_regression(df, target, columns, **_):
    import statsmodels.api as sm
    X, y, used = _design(df, target, columns)
    model = sm.OLS(y.astype(float), sm.add_constant(X, has_constant="add")).fit()
    coefs = _coef_table(model)
    sig = coefs[(coefs.p_value < ALPHA) & (coefs.term != "const")]
    return {"tables": {"coefficients": coefs},
            "facts": {"r2": round(model.rsquared, 4), "adj_r2": round(model.rsquared_adj, 4), "n": int(model.nobs),
                      "significant_terms": sig.term.tolist()},
            "interpretation": f"OLS on {int(model.nobs)} rows explains {model.rsquared:.1%} of the variance in {target} "
                              f"(adj. R2={model.rsquared_adj:.3f}). Significant predictors (p<{ALPHA}): "
                              f"{', '.join(f'{r.term} ({r.coef:+.4g})' for r in sig.itertuples()) or 'none'}. "
                              f"Each coefficient = change in {target} for +1 unit, holding others fixed."}


def _m_logistic_regression(df, target, columns, **_):
    import statsmodels.api as sm
    X, y, used = _design(df, target, columns)
    classes = sorted(pd.unique(y))
    if len(classes) != 2:
        raise ValueError(f"Logistic regression needs a binary target; {target} has {len(classes)} values.")
    y01 = (y == classes[1]).astype(float)
    model = sm.Logit(y01, sm.add_constant(X, has_constant="add")).fit(disp=0)
    coefs = _coef_table(model)
    coefs["odds_ratio"] = np.exp(coefs["coef"]).round(4)
    sig = coefs[(coefs.p_value < ALPHA) & (coefs.term != "const")]
    return {"tables": {"coefficients": coefs},
            "facts": {"pseudo_r2": round(model.prsquared, 4), "n": int(model.nobs), "positive_class": str(classes[1]),
                      "significant_terms": sig.term.tolist()},
            "interpretation": f"Logit model for P({target}={classes[1]}), pseudo-R2={model.prsquared:.3f}. "
                              f"Significant predictors: "
                              f"{', '.join(f'{r.term} (odds x{r.odds_ratio:.3g})' for r in sig.itertuples()) or 'none'}. "
                              f"Odds ratio >1 raises the odds per +1 unit."}


# ------------------------------------------------------------------ helpers
def _need(df, *cols):
    for c in cols:
        if not c or c not in df.columns:
            raise ValueError(f"Column '{c}' not found in the query result. Columns: {list(df.columns)}")


def _design(df, target, columns):
    _need(df, target)
    feats = [c for c in (columns or [c for c in df.columns if c != target]) if c in df.columns and c != target]
    feats = [c for c in feats if not (c.lower().endswith("id") and df[c].nunique() > 0.9 * len(df))]
    if not feats:
        raise ValueError("No usable feature columns for the regression.")
    data = df[feats + [target]].dropna()
    X = pd.get_dummies(data[feats], drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    return X.astype(float), data[target], feats


def _coef_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    return pd.DataFrame({"term": model.params.index, "coef": model.params.values.round(5),
                         "std_err": model.bse.values.round(5), "p_value": model.pvalues.values.round(5),
                         "ci_low": ci[0].values.round(5), "ci_high": ci[1].values.round(5)})
