import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, f1_score, PrecisionRecallDisplay


# Function for model evaluation
def predict_and_plot(model_pipeline, inputs, targets, name=''):
    preds = model_pipeline.predict(inputs)

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cf, annot=True, fmt='.2f')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));

    # Predict probabilities
    y_pred_proba = model_pipeline.predict_proba(inputs)[:, 1]

    # Compute ROC curve and AUROC (for completeness, though not plotted now)
    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f'AUROC for {name}: {roc_auc:.4f}')

    # Compute Average Precision Score (AUCPR)
    auprc = average_precision_score(targets, y_pred_proba)
    print(f'AUCPR for {name}: {auprc:.4f}')

    # Compute F1-score
    f1 = f1_score(targets, preds)
    print(f'F1-score for {name}: {f1:.4f}')

    # Plot the Precision-Recall curve
    plt.subplot(1, 2, 2)
    display_pr = PrecisionRecallDisplay.from_predictions(targets, y_pred_proba, ax=plt.gca(), name=f'Precision-Recall curve (AP = {auprc:.2f})')
    plt.title(f'Precision-Recall Curve for {name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show();

    return preds