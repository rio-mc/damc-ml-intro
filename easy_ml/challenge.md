# Easy challenge: Classical ML

## Bronze: Overfitting and model capacity
Train random forests with different `max_depth` values and compare:
- train RMSE
- validation RMSE
- train-validation gap

Think:
- At what point does increasing depth stop improving generalisation?
- How can you identify overfitting from the metrics alone?

## Silver: Stability and ensemble behaviour
Change `n_estimators` and `min_samples_leaf` and compare the resulting validation performance.

Think:
- Which hyperparameters improve robustness rather than simply reducing training error?
- Does increasing ensemble size always improve the model meaningfully?

## Gold: Representation dependence
Train the model using reduced descriptor subsets.

Think:
- Which descriptors appear most important?
- Are the most influential descriptors chemically interpretable?
- What does this reveal about the dependence of classical ML on handcrafted representations?