# Hard challenge: Deep Learning

## Bronze: Identify Overfitting
Run the baseline CNN and inspect the train and validation loss curves.

At what epoch does validation performance stop improving?
What evidence suggests the model begins to overfit?

Try at least one intervention:

- reduce model capacity
- increase dropout
- enable early stopping

Did generalisation improve?

## Silver: Capacity versus generalisation
Modify the CNN while keeping:

- CPU training time low
- parameter count scientifically reasonable
- training stable

Compare at least two architectures with different:

- channel counts
- kernel sizes
- depths
- regularisation settings

Your goal is not maximum train accuracy.

Instead, justify which configuration gives the best balance between:

- validation performance
- train-validation gap
- robustness
- runtime
- architectural simplicity

## Gold: Scientific robustness audit
Stress-test your best-performing CNN by perturbing the data or training setup.

Examples:

- reduce training-set size
- enable augmentation to add noise and shift peaks
- add stronger noise augmentation
- disable learning-rate scheduling
- increase model capacity substantially

Then analyse:

- Does the model fail gracefully?
- Does validation behaviour predict test behaviour?
- Which crystal systems become harder to classify?
- What experimental conditions might break this model?