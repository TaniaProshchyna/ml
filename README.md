This project focuses on predicting whether a bank client will subscribe to a term deposit, using a dataset from a direct marketing campaign. The primary objective is to build a classification model that can effectively identify potential subscribers, balancing the need to capture as many positive cases as possible (recall) with minimizing wasted marketing efforts (precision), especially given the class imbalance in the dataset.

### Models Explored:

*   **Logistic Regression**: A linear baseline model.
*   **k-Nearest Neighbors (k-NN)**: A non-parametric, instance-based learning algorithm.
*   **Decision Tree**: An interpretable tree-based model.
*   **LightGBM**: A powerful gradient boosting framework, explored with:
    *   Arbitrary hyperparameters.
    *   Hyperparameters optimized using Randomized Search.
    *   Hyperparameters optimized using Bayesian Optimization (Hyperopt).

### Evaluation Metric:

**Area Under the Precision-Recall Curve (AUPRC)** was chosen as the primary evaluation metric due to the significant class imbalance (approximately 11.3% positive class). This metric provides a more robust assessment of model performance for the minority class compared to AUC-ROC or simple accuracy.

### Best Performing Model:

The **LightGBM model with hyperparameters tuned using Bayesian Optimization** achieved the best performance:
*   **Validation AUPRC: 0.5002**

### Most Important Features:

The feature importance analysis for the best LightGBM model revealed the following as most influential:
1.  **`euribor3m`**: 3-month EURIBOR rate.
2.  **`age`**: Client's age.
3.  **`campaign`**: Number of contacts performed during this campaign.
4.  **`day_of_week`**: Day of the week of the last contact.
5.  **`cons.price.idx`**: Consumer Price Index.

### Suggested Probability Threshold:

For practical application in a marketing campaign, a probability threshold of **0.3797** is suggested. This threshold was selected to achieve a balance between Precision (0.4986) and Recall (0.5819) within the desired range of 0.4 to 0.6 for both metrics, while maximizing the F1-score (0.5370). This approach aims to identify a good number of potential subscribers without excessive outreach to unlikely candidates, optimizing marketing efficiency and effectiveness.
