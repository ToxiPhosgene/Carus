import nolds
import torch
from scipy import stats


def dfa_and_herst_analysis(signals: list):
    alpha = []
    h = []
    for signal in signals:
        h.append(nolds.hurst_rs(signal.transpose(dim0=1, dim1=0)))
        alpha.append(nolds.dfa(signal.transpose(dim0=1, dim1=0)))

    h = torch.tensor(h)
    alpha = torch.tensor(alpha)

    t_value_h = stats.t.ppf(0.9, df=h.size(-1))

    data_error_h = (h > h.mean() + t_value_h).any() | (h < h.mean() - t_value_h).any()
    while data_error_h:
        h = h[(h < h.mean() + t_value_h) & (h > h.mean() - t_value_h)]
        t_value_h = stats.t.ppf(0.9, df=h.size(-1))
        data_error_h = (h > h.mean() + t_value_h).any() | (h < h.mean() - t_value_h).any()

    t_value_alpha = stats.t.ppf(0.9, df=h.size(-1))
    data_error_alpha = (alpha > alpha.mean() + t_value_alpha).any() | (alpha < alpha.mean() - t_value_alpha).any()
    while data_error_alpha:
        alpha = alpha[(alpha < alpha.mean() + t_value_alpha) & (alpha > alpha.mean() - t_value_alpha)]
        t_value_alpha = stats.t.ppf(0.9, df=h.size(-1))
        data_error_alpha = (alpha > alpha.mean() + t_value_alpha).any() | (alpha < alpha.mean() - t_value_alpha).any()

    ratio_h = ((h <= 0.9) & (h >= 0.7)).sum()
    ratio_alpha = ((alpha <= 1.2) & (alpha >= 0.8)).sum()

    result = {
        "h": {
            "mean": h.mean(),
            "std": h.std() * stats.t.ppf(0.9, df=h.size(-1))
        },
        "ratio_h": ratio_h / h.size(-1) * 100,
        "dfa": {
            "mean": alpha.mean(),
            "std": alpha.std() * stats.t.ppf(0.9, df=alpha.size(-1))
        },
        "ratio_dfa": ratio_alpha / alpha.size(-1) * 100
    }
    return result
