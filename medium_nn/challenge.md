# Medium challenge: Neural Networks

## Bronze: Capacity and generalisation
Train MLPs with different hidden layer sizes and compare:
- training time
- train RMSE
- validation RMSE
- train-validation gap

Think:
- Does increasing capacity improve generalisation?
- At what point does additional capacity mainly improve training performance?

## Silver: Regularisation study
Compare the same MLP with and without dropout.

Evaluate:
- validation performance
- training stability
- train-validation gap

Think:
- Does dropout improve generalisation, or mainly constrain optimisation?
- Under what conditions might dropout become unnecessary?

## Gold: Optimisation instability
Increase the learning rate substantially and inspect the training logs.

Think:
- How can you tell the optimiser is becoming unstable?
- Which metrics or behaviours indicate divergence or failed convergence?
- What intervention would you try first to recover training?