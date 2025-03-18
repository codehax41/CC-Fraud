#### Context:
Credit card companies need to detect fraudulent transactions to prevent customers from being charged for unauthorized purchases.

#### Content:
- **Dataset**: Transactions made by European cardholders in September 2013.
- **Duration**: Two days, with 492 frauds out of 284,807 transactions.
- **Class Imbalance**: Fraudulent transactions (positive class) account for 0.172% of all transactions.
- **Features**:
  - Numerical input variables resulting from PCA transformation.
  - 'Time': Seconds elapsed between each transaction and the first transaction.
  - 'Amount': Transaction amount, suitable for cost-sensitive learning.

- **Target**:
  - 'Class': Response variable, 1 for fraud, 0 otherwise.
  
#### Source:
- The dataset has been collected and analyzed by Worldline and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB) as part of a research collaboration on big data mining and fraud detection.
- More details on the current and past projects related to fraud detection are available on the [MLG website](http://mlg.ulb.ac.be) and [ResearchGate](https://www.researchgate.net/project/Fraud-detection-5).

#### Recommendations:
- Due to class imbalance, accuracy should be measured using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful.