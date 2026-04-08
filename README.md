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

*  ## Model Performance Comparison

| Model Name                             | Hyperparameters                                                                                                                                                                                                                                                                                              | Training Time | Train AUCPR | Validation AUCPR | Comment                                                                                                                                                                                                                                                                                                                                       |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------- | :---------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**                  | `solver='liblinear'`, `random_state=42`                                                                                                                                                                                                                                                                                      | 5s               | 0.4582      | 0.4677           | A solid baseline model. Its performance on training and validation sets is very close, indicating good generalization and no obvious overfitting. It serves as a good benchmark for more complex models.                                                                                                                                                                        |
| **k-Nearest Neighbors (k-NN)**           | Default (`n_neighbors=5`)                                                                                                                                                                                                                                                                                                    | 44s               | 0.5554      | 0.3266           | Shows clear signs of overfitting, with a much higher AUCPR on the training set compared to the validation set. This suggests the model is memorizing the training data. Further tuning or a different approach is needed for better generalization.                                                                                                                            |
| **Best Decision Tree**                   | `max_depth=6`, `max_leaf_nodes=35`, `random_state=42`                                                                                                                                                                                                                                                                        | 3mins               | 0.4554      | 0.4486           | Provides a reasonable balance and better generalization than k-NN, performing comparably to Logistic Regression. It's highly interpretable, which can be valuable for understanding feature interactions.                                                                                                                                                            |
| **LightGBM (Arbitrary Hyperparameters)** | `max_depth=3`, `n_estimators=100`, `learning_rate=0.1`, `cat_feature=cat_feature_indexes`, `missing=np.nan`                                                                                                                                                                                                                  | 3s               | 0.4686      | 0.4686           | Demonstrates strong and balanced performance right out of the box, with very similar AUCPR on both training and validation sets. This indicates good generalization and robust learning, showing its potential before extensive tuning.                                                                                                                           |
| **LightGBM (Random Search)**             | `subsample=1.0`, `scale_pos_weight=1`, `reg_lambda=0`, `reg_alpha=10`, `num_leaves=4`, `n_estimators=800`, `min_child_samples=40`, `max_depth=-1`, `learning_rate=0.05`, `colsample_bytree=0.6`                                                                                                                                     | 3min            | 0.4943      | 0.4856           | Achieved a significant improvement over the arbitrary LightGBM parameters, demonstrating the value of hyperparameter tuning. The scores are well-balanced, suggesting a good trade-off between bias and variance.                                                                                                                                                      |
| **LightGBM (Bayesian Optimization)**     | `colsample_bytree=0.8959`, `learning_rate=0.0392`, `max_bin=160`, `max_depth=6`, `min_child_samples=90`, `min_split_gain=0.0767`, `n_estimators=1950`, `num_leaves=61`, `reg_alpha=7.9481`, `reg_lambda=130.48`, `scale_pos_weight=1.7806`, `subsample=0.6556` (Approx. 200 evaluations for fmin) | 1hour      | 0.5125      | 0.5002           | Achieved the highest validation AUCPR among all tested models, showcasing the effectiveness of Bayesian optimization in finding a near-optimal set of hyperparameters. The performance is robust, with a good balance between training and validation scores. This model represents the best performance found so far.

### Most Important Features:

The feature importance analysis for the best LightGBM model revealed the following as most influential:
1.  **`euribor3m`**: 3-month EURIBOR rate.
2.  **`age`**: Client's age.
3.  **`campaign`**: Number of contacts performed during this campaign.
4.  **`day_of_week`**: Day of the week of the last contact.
5.  **`cons.price.idx`**: Consumer Price Index.

### Suggested Probability Threshold:

For practical application in a marketing campaign, a probability threshold of **0.3797** is suggested. This threshold was selected to achieve a balance between Precision (0.4986) and Recall (0.5819) within the desired range of 0.4 to 0.6 for both metrics, while maximizing the F1-score (0.5370). This approach aims to identify a good number of potential subscribers without excessive outreach to unlikely candidates, optimizing marketing efficiency and effectiveness.
