from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, 
    
)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap



class ClassificationEvaluator:
    def __init__(self, model):
        self.model = model


    def _validate_inputs(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y).ravel()
        return X, y


    def _calculate_metrics(self, y_true, y_pred, y_proba):
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_proba)
            logloss = log_loss(y_true, y_proba)
        else:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            logloss = log_loss(y_true, y_proba)

        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'ROC AUC': roc_auc,
            'Matthews Corrcoef': matthews_corrcoef(y_true, y_pred),
            'Cohen Kappa': cohen_kappa_score(y_true, y_pred),
            'Log Loss': logloss
        }

        return {k: round(v * 100, 2) if k not in ['Matthews Corrcoef', 'Cohen Kappa'] else round(v, 2)
                for k, v in metrics.items()}


    def fit_evaluate(self, X, y, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        if predict_proba and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)
            y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
        else:
            y_proba = np.zeros_like(y, dtype=float)

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        return pd.DataFrame([metrics]), X, y, y_pred, y_proba


    def cross_validate(self, X, y, n_splits=5, predict_proba=True):
        X, y = self._validate_inputs(X, y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        metrics_agg = {k: [] for k in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC',
                                       'Matthews Corrcoef', 'Cohen Kappa', 'Log Loss']}
        X_test_all, y_true_all, y_pred_all, y_proba_all = [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if predict_proba and hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_test)
                y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            else:
                y_proba = np.zeros_like(y_pred, dtype=float)

            X_test_all.extend(X_test)
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba if y_proba.ndim == 1 else y_proba.max(axis=1))

            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            for k in metrics_agg:
                metrics_agg[k].append(metrics[k])

        final_metrics = pd.DataFrame([{k: round(np.mean(v), 2) for k, v in metrics_agg.items()}])
        return final_metrics, X_test_all, y_true_all, y_pred_all, y_proba_all
    
    
    def cross_val_predict_summary(self, X, y, cv=5):
        X, y = self._validate_inputs(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=cv, method='predict')

        if hasattr(self.model, "predict_proba"):
            y_proba = cross_val_predict(self.model, X, y, cv=cv, method='predict_proba')
            y_proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
        else:
            y_proba = np.zeros_like(y_pred, dtype=float)

        metrics = self._calculate_metrics(y, y_pred, y_proba)
        return pd.DataFrame([metrics]), X, y, y_pred, y_proba
    
    
class ClassificationPlotter:
    def __init__(self, figsize=(4, 4)):
        self.figsize = figsize

    def plot_confusion_matrix(self, y_true, y_pred, normalize='true'):
        fig, ax = plt.subplots(figsize=self.figsize)
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        raw_cm = confusion_matrix(y_true, y_pred)
        labels = np.array([
            f"{raw}\n{pct:.1%}" for raw, pct in zip(raw_cm.flatten(), cm.flatten())
        ]).reshape(cm.shape)
        sns.heatmap(cm, annot=labels, fmt='', ax=ax, cmap=sns.cubehelix_palette(as_cmap=True),
                    linewidths=0.5, linecolor='gray')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true, y_proba):
        fig, ax = plt.subplots(figsize=self.figsize)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.legend(loc="lower right")
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_proba):
        fig, ax = plt.subplots(figsize=self.figsize)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(recall, precision, lw=2)
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        plt.tight_layout()
        plt.show()

    def plot_probability_distribution(self, y_proba):
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(y_proba, bins=20, kde=True, ax=ax)
        ax.set_title("Probability Distribution")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    def plot_calibration_curve(self, y_true, y_proba, n_bins=10):
        fig, ax = plt.subplots(figsize=self.figsize)
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', label='Calibration')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        ax.set_title('Calibration Curve')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.legend()
        plt.tight_layout()
        plt.show()
      
      
        
    def plot_learning_curve(self, X, y, scoring='accuracy', cv=5):
        
        fig, ax = plt.subplots(figsize=self.figsize)
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        ax.plot(train_sizes, train_mean, label="Training score")
        ax.plot(train_sizes, val_mean, label="Cross-validation score")
        ax.set_title("Learning Curve")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(scoring.capitalize())
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        
    def plot_shap_summary(self, X, model):
        explainer = shap.Explainer(model.steps[-1][1] if hasattr(model, 'steps') else model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)