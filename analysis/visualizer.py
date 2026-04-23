import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_defense_impact(baseline, protected):
    plt.figure(figsize=(10, 5))
    plt.plot(baseline.cumsum(), label="Baseline (Unprotected)", alpha=0.5)
    plt.plot(protected.cumsum(), label="Governance Engine (Protected)", linewidth=2)
    plt.title("Defensive Mechanism: Capital Preservation Audit")
    plt.legend()
    plt.show()

def plot_governance_activation(market_returns, gov_returns):
    """Shows the 'Gate' opening and closing."""
    activation = (market_returns != gov_returns).astype(int)
    plt.figure(figsize=(10, 2))
    plt.imshow(activation.reshape(1, -1), aspect='auto', cmap='Reds', interpolation='nearest')
    plt.title("Governance Gate Activity (Red = Defensive Mode)")
    plt.yticks([])
    plt.show()



def plot_return_distribution(baseline, protected):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(baseline, label="Baseline (Fat Tails)", fill=True)
    sns.kdeplot(protected, label="Governance (Truncated Tails)", fill=True)
    plt.title("Return Distribution Comparison")
    plt.legend()
    plt.show()

def plot_drawdown_surface(baseline, protected):
    # Calculate running drawdown
    def get_dd(r):
        cumulative_returns = (1 + r).cumprod()
        # Use np.maximum.accumulate for the running high-water mark
        running_max = np.maximum.accumulate(cumulative_returns)
        return cumulative_returns / running_max - 1
    
    plt.figure(figsize=(10, 4))
    plt.plot(get_dd(baseline), label="Baseline Drawdown", alpha=0.6)
    plt.plot(get_dd(protected), label="Governance Drawdown", color='red')
    plt.title("Drawdown Surface Comparison")
    plt.legend()
    plt.show()