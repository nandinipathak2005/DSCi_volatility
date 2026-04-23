# import numpy as np
# from governance_metrics import *
# from visualizer import plot_defense_impact

# def run_defense_test():
#     # Simulate market crash
#     market = np.random.normal(0.0005, 0.01, 252)
#     market[150:200] = -0.04  # The "Crash"

#     # Defensive Mechanism: Governance Gate
#     # If loss > 2%, switch to cash (return 0)
#     protected = np.where(market < -0.02, 0, market)

#     # Audit
#     print("--- CAPITAL PRESERVATION AUDIT ---")
#     print(f"Calmar (Safety)   : {calmar_ratio(protected):.2f}")
#     print(f"Downside (Pain)   : {downside_deviation(protected):.2f}")
#     print(f"Recovery (Speed)  : {recovery_factor(protected):.2f}")
    
#     plot_defense_impact(market, protected)

# if __name__ == "__main__":
#     run_defense_test()

import numpy as np
from governance_metrics import *
from visualizer import *

def run_defense_test():
    # 1. Simulate data
    np.random.seed(42)
    market = np.random.normal(0.0005, 0.01, 252)
    market[150:200] = -0.04  # The "Crash"

    # 2. Defensive Mechanism
    protected = np.where(market < -0.02, 0, market)

    # 3. Audit Printout
    print("--- CAPITAL PRESERVATION AUDIT ---")
    print(f"Calmar (Safety)   : {calmar_ratio(protected):.2f}")
    print(f"Downside (Pain)   : {downside_deviation(protected):.2f}")
    print(f"Recovery (Speed)  : {recovery_factor(protected):.2f}")
    
    # 4. Generate all Visuals for the Viva
    plot_defense_impact(market, protected)
    plot_governance_activation(market, protected)
    plot_return_distribution(market, protected)
    plot_drawdown_surface(market, protected)

if __name__ == "__main__":
    run_defense_test()