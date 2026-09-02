# Section 2: Pre-Experimental Prediction Sheet

---

## 1. Lower Bound Prediction (Sequential Training without Mitigation)

* **Predicted Final Average Accuracy:** ~18% - 20%
* **Predicted Backward Transfer (BWT):** ~ -80% to -90%
* **Predicted Average Forgetting:** ~ 80% - 90%

### Reasoning
In a single-head class-incremental setting with 10 classes, standard SGD training across 5 sequential 2-class tasks will suffer catastrophic forgetting. As training progresses to Task 5 (digits 8 and 9), backpropagating gradients will overwrite early feature representations. Furthermore, because the output head only sees digits 8 and 9 during the final task, its bias terms and weight vectors for earlier classes (0–7) will receive negative updates, causing the model to almost exclusively predict the most recently learned classes.

---

## 2. Upper Bound Prediction (Joint / Offline Training)

* **Predicted Final Average Accuracy (A_5):** ~97% - 98%
* **Predicted Backward Transfer (BWT):** 0.0% *( No sequential degradation)*

### Reasoning
Training an identical 3-layer MLP architecture on all 10 classes simultaneously (i.i.d. assumption) provides a balanced gradient signal across all outputs, eliminating task-recency bias and allowing the hidden layers to form globally optimal decision boundaries.

---

## 3. Diagnostic Hypothesis Prediction (H1 vs. H2)

* **Predicted Winner:** **H2 (Output Head Bias) will account for the majority of the immediate accuracy drop, but H1 (Feature Destruction) will also be present.**

### Reasoning
The output classification head will quickly develop strong negative biases for unobserved classes during later tasks, driving Task 1 accuracy to near 0%. However, because a small MLP has limited parameter capacity and no weight-regularization penalty, backpropagating gradients from Tasks 2–5 through shared hidden layers will also partially degrade lower-level feature detectors tuned for early digits.

---

*Note: This file represents the locked pre-experimental prediction sheet required prior to code execution.*