"""Train on NASA C-MAPSS FD001 and write artifacts.json. Run: python ml.py"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GroupKFold

DS, CAP, H, P = "FD001", 125, 30, dict(max_iter=300, learning_rate=0.05, random_state=0)
cols = ["unit", "cycle", "o1", "o2", "o3"] + [f"s{i}" for i in range(1, 22)]
ld = lambda n: pd.read_csv(Path("data") / f"{n}_{DS}.txt", sep=r"\s+", header=None, names=cols) \
    .sort_values(["unit", "cycle"]).reset_index(drop=True)
tr, te = ld("train"), ld("test")
rul_te = pd.read_csv(Path("data") / f"RUL_{DS}.txt", header=None)[0].values
tr["rul"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
S = [c for c in cols[5:] if tr[c].nunique() > 10]  # drop constant sensors, found from training data

SM = {"s1": ("Fan inlet temperature (T2)", "Fan"), "s2": ("LPC outlet temperature (T24)", "LPC"),
      "s3": ("HPC outlet temperature (T30)", "HPC"), "s4": ("LPT outlet temperature (T50)", "LPT"),
      "s5": ("Fan inlet pressure (P2)", "Fan"), "s6": ("Bypass-duct pressure (P15)", "Fan"),
      "s7": ("HPC outlet pressure (P30)", "HPC"), "s8": ("Fan speed (Nf)", "Fan"),
      "s9": ("Core speed (Nc)", "HPC"), "s10": ("Engine pressure ratio (epr)", "Fan"),
      "s11": ("HPC outlet static pressure (Ps30)", "HPC"), "s12": ("Fuel flow to Ps30 ratio (phi)", "Combustor"),
      "s13": ("Corrected fan speed (NRf)", "Fan"), "s14": ("Corrected core speed (NRc)", "HPC"),
      "s15": ("Bypass ratio (BPR)", "Fan"), "s16": ("Burner fuel-air ratio (farB)", "Combustor"),
      "s17": ("Bleed enthalpy (htBleed)", "HPC"), "s18": ("Demanded fan speed", "Fan"),
      "s19": ("Demanded corrected fan speed", "Fan"), "s20": ("HPT coolant bleed (W31)", "HPT"),
      "s21": ("LPT coolant bleed (W32)", "LPT")}


def feats(df):
    """Features use only the current and past rows of each engine (no future leakage)."""
    parts = []
    for u, g in df.groupby("unit"):
        X = g[S].reset_index(drop=True)
        Z = (X - X.iloc[:30].mean()) / (X.iloc[:30].std() + 1e-6)  # deviation from own healthy baseline
        F = pd.concat([Z.add_suffix("_dev"), Z.rolling(5, min_periods=1).mean().add_suffix("_m5"),
                       Z.rolling(15, min_periods=1).mean().add_suffix("_m15"),
                       Z.rolling(10, min_periods=2).std().fillna(0).add_suffix("_sd10"),
                       (Z - Z.shift(10)).fillna(0).add_suffix("_d10")], axis=1)
        F["cycle"], F["unit"] = g.cycle.values, u
        parts.append(F)
    return pd.concat(parts, ignore_index=True)


Ftr, Fte = feats(tr), feats(te)
XC = [c for c in Ftr.columns if c != "unit"]
y = np.minimum(tr.rul.values, CAP)

# out-of-fold residuals (split by engine, never by row) give the prediction interval
oof = np.zeros(len(y))
for a, b in GroupKFold(5).split(Ftr, y, tr.unit):
    oof[b] = HGR(**P).fit(Ftr.iloc[a][XC], y[a]).predict(Ftr.iloc[b][XC])
q10, q90 = np.percentile(y - oof, [10, 90])

reg = HGR(**P).fit(Ftr[XC], y)
clf = HGC(max_iter=200, learning_rate=0.05, random_state=0).fit(Ftr[XC], (tr.rul <= H).astype(int))
pred = np.clip(reg.predict(Fte[XC]), 0, CAP)
p30 = clf.predict_proba(Fte[XC])[:, 1]

# evaluation: last cycle of each test engine vs the true RUL file (both capped at CAP)
last = Fte.groupby("unit").tail(1).index.values
ac, pl = np.minimum(rul_te, CAP), pred[last]
d = pl - ac
rmse, mae = float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))
nasa = float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())
base_mean = float(np.sqrt(np.mean((y.mean() - ac) ** 2)))
lin = LinearRegression().fit(tr[["cycle"]], y).predict(te.loc[last, ["cycle"]])
base_lin = float(np.sqrt(np.mean((lin - ac) ** 2)))
a30, c30 = (rul_te <= H).astype(int), (p30[last] >= 0.5).astype(int)
pr, rc, f1, _ = precision_recall_fscore_support(a30, c30, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(a30, c30, labels=[0, 1]).ravel()
auc = float(roc_auc_score(a30, p30[last]))
cover = float(np.mean((ac >= np.clip(pl + q10, 0, CAP)) & (ac <= np.clip(pl + q90, 0, CAP))))

f = lambda x: round(float(x), 3)
band = lambda r: "Critical" if r <= 30 else "High" if r <= 60 else "Medium" if r <= 100 else "Low"
machines = []
for u, g in Fte.groupby("unit"):
    i, raw = g.index.values, te[te.unit == u]
    p = pred[i]
    dv = g[[s + "_dev" for s in S]].tail(10).mean()
    dv.index = S
    top = dv.reindex(dv.abs().sort_values(ascending=False).index)[:5]
    sub = {}
    for s, v in top.items():
        sub[SM[s][1]] = sub.get(SM[s][1], 0) + abs(v)
    comp = max(sub, key=sub.get)
    drivers = [dict(sensor=s, label=SM[s][0], deviationSigma=f(v), direction="up" if v > 0 else "down")
               for s, v in top[:4].items()]
    r = float(p[-1])
    machines.append(dict(
        machineId=f"ENG-{DS}-{u:03d}", dataset=DS, cyclesObserved=int(len(g)), predictedRul=f(r),
        rulLower=f(np.clip(r + q10, 0, CAP)), rulUpper=f(np.clip(r + q90, 0, CAP)),
        failureProb30=f(p30[i[-1]]), riskScore=int(round(100 * (1 - min(r, CAP) / CAP))), band=band(r),
        component=dict(name=comp, basis="indicative", drivers=drivers), cycles=[int(c) for c in g.cycle],
        series={s: dict(label=SM[s][0], values=[f(v) for v in raw[s]],
                        baselineMean=f(raw[s].iloc[:30].mean()), baselineStd=f(raw[s].iloc[:30].std() + 1e-6)) for s in S},
        rulHistory=[dict(cycle=int(c), predictedRul=f(v), lower=f(np.clip(v + q10, 0, CAP)),
                         upper=f(np.clip(v + q90, 0, CAP))) for c, v in zip(g.cycle, p)]))
machines.sort(key=lambda m: -m["riskScore"])

model = dict(
    name="Gradient boosting (scikit-learn HistGradientBoosting)", version="1.0",
    trainedOn=f"C-MAPSS {DS}, {tr.unit.nunique()} run-to-failure engines", rulCap=CAP,
    evaluation=f"Last cycle of each of the {len(last)} test engines compared with the true RUL file. "
               f"Both predicted and true RUL are capped at {CAP} cycles. Validation during training was split by engine.",
    regression=dict(rmse=f(rmse), mae=f(mae), nasaScore=f(nasa),
                    baselines=[dict(name="Predict training mean", rmse=f(base_mean)),
                               dict(name="Linear model on cycle count", rmse=f(base_lin))]),
    classification=dict(horizon=H, precision=f(pr), recall=f(rc), f1=f(f1), rocAuc=f(auc),
                        confusion=dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))),
    intervalCoverage=f(cover),
    predictedVsActual=[dict(machineId=f"ENG-{DS}-{k + 1:03d}", actual=int(a), predicted=f(p)) for k, (a, p) in enumerate(zip(ac, pl))],
    limitations=["The data is simulated. A model trained on it will not work on real machines without retraining.",
                 "The dataset has no component labels. The likely component is an indicative guess from sensor deviation, not a diagnosis.",
                 f"Only {len(last)} test engines and {int(a30.sum())} of them fail within {H} cycles, so classification metrics are noisy.",
                 "RUL is capped, so very healthy machines all show the same upper value."])
json.dump(dict(model=model, machines=machines), open("artifacts.json", "w"))
print(f"RMSE {rmse:.2f} | MAE {mae:.2f} | NASA score {nasa:.0f} | baselines {base_mean:.1f} / {base_lin:.1f}")
print(f"Fail<=30: precision {pr:.2f} recall {rc:.2f} F1 {f1:.2f} AUC {auc:.2f} | interval coverage {cover:.2f}")
print("Wrote artifacts.json")
