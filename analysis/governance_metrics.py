# import numpy as np

# def calmar_ratio(returns):
#     # Rewards systems that protect against drawdowns
#     ann_return = np.mean(returns) * 252
#     mdd = abs(np.min(returns.cumsum() - np.maximum.accumulate(returns.cumsum())))
#     return ann_return / mdd if mdd != 0 else 0

# def downside_deviation(returns):
#     # Only counts the "pain" (losses), ignores the "gain"
#     neg_returns = returns[returns < 0]
#     return np.std(neg_returns) * np.sqrt(252)

# def recovery_factor(returns):
#     # How fast the system recovers from its max drawdown
#     mdd = abs(np.min(returns.cumsum() - np.maximum.accumulate(returns.cumsum())))
#     return (np.mean(returns) * 252) / mdd if mdd != 0 else 0

import numpy as np

def calmar_ratio(returns):
    ann_return = np.mean(returns) * 252
    mdd = abs(np.min(returns.cumsum() - np.maximum.accumulate(returns.cumsum())))
    return ann_return / mdd if mdd != 0 else 0

def downside_deviation(returns):
    # Only penalizes losses (the "Pain")
    neg_returns = returns[returns < 0]
    return np.std(neg_returns) * np.sqrt(252)

def recovery_factor(returns):
    mdd = abs(np.min(returns.cumsum() - np.maximum.accumulate(returns.cumsum())))
    return (np.mean(returns) * 252) / mdd if mdd != 0 else 0