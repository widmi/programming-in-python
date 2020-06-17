# -*- coding: utf-8 -*-
"""09_evaluation.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

In this file we will learn how to evaluate a binary classifier using
scikit-learn.
"""

import numpy as np
import torch
np.random.seed(0)

#
# Create some example targets and predictions
#

# Create some labels/targets (assuming a binary classification task)
n_samples = 8
labels = np.random.randint(low=0, high=2, size=n_samples, dtype=np.int)

# We have a binary classification task, so we use a sigmoid output activation
# function in the output layer of our network.
# Create some network output before sigmoid activation:
logits = torch.tensor(np.random.uniform(low=-10, high=10, size=n_samples))
# Create network output after sigmoid activation:
outputs = torch.sigmoid(logits)
# Create network predictions using a threshold. We will chose 0.5 as border
# between negative vs. positive predictions. We could adjust this threshold
# to have more negative or positive predictions if we have to meet different
# costs for FP and FN.
threshold = 0.5
predictions = (outputs >= threshold).long()

print(f"Labels:      {labels.tolist()}")
print(f"Predictions: {predictions.tolist()}")
print(f"Logits:  {logits.tolist()}")
print(f"Outputs: {outputs.tolist()}")


###############################################################################
# Using sklearn metrics
###############################################################################

#
# We can simply apply the evaluation functions from scikit-learn
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# to our targets and outputs/predictions.
#

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=labels, y_pred=predictions.numpy())
tn, fp, fn, tp = cm.reshape(-1)
print(f"Confusion matrix:\nTP={tp} FN={fn}\nFP={fp} TN={tn}")

# Accuracy (ACC)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true=labels, y_pred=predictions.numpy())
print(f"Accuracy: {acc}")

# Balanced accuracy (BACC)
from sklearn.metrics import balanced_accuracy_score
bacc = balanced_accuracy_score(y_true=labels, y_pred=predictions.numpy())
print(f"Balanced accuracy: {bacc}")

# Receiver operating characteristic (ROC) and AUC
from sklearn.metrics import roc_curve
# We can use logits or outputs to compute the ROC
fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=logits.numpy(),
                                 drop_intermediate=False)
print(f"FPR:            {fpr}")
print(f"TPR:            {tpr}")
print(f"ROC thresholds: {thresholds}")

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true=labels, y_score=logits.numpy())
print(f"AUC: {roc_auc}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='our model')
ax.scatter(fpr, tpr)
ax.plot([0, 1], [0, 1], linestyle='--', label='random performance')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC curve (AUC={roc_auc})')
ax.legend()
plt.show(block=True)
