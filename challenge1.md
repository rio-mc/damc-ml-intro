# Easy Challenge: Tree-Based Models & Molecular Descriptors

## Bronze: Model Capacity & Overfitting
Train a series of Random Forest models varying the `max_depth` (e.g., from 2 to 20) to predict your chemical property. Plot and compare the training RMSE, validation RMSE, and the train-validation gap.

> ### Think:
> * At what specific structural complexity (`max_depth`) does the model stop learning generalisable chemical trends and start memorising the training set? 

---

## Silver: Robustness & Ensemble Behaviour
Fix your depth, but experiment with `n_estimators` (number of trees) and `min_samples_leaf` (minimum molecules per leaf node). Observe how these affect your validation performance and prediction variance.

> ### Think:
> * Which of these hyperparameters forces the model to learn broader, group-level chemical trends rather than getting distracted by structural outliers?

---

## Gold: Feature Importance & Chemical Intuition
Train your best Random Forest model, extract the feature importances, and then retrain the model using only the top 5 or 10 descriptors. 

> ### Think:
> * If a purely statistical model relies heavily on a descriptor you wouldn't traditionally use in the lab, does that reveal a novel chemical correlation, or a bias in your dataset?
> * What does this tell us about the limitations of relying on "handcrafted" chemical representations versus letting a model learn features directly from a molecular graph or spectrum?

