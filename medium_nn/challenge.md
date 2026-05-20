# Medium Challenge: Neural Networks & Property Prediction

## Bronze: Network Capacity & Representation
Train Multilayer Perceptrons (MLPs) with varying hidden layer architectures (e.g., a single small layer vs. deep, wide layers). Track training time, training RMSE, validation RMSE, and the generalisation gap.

> ### Think:
> * The network capacity is the model's "cognitive budget." Does giving the network more layers allow it to capture complex, non-linear chemical interactions, or does it just give it more room to memorise noise in the experimental data? 
> * At what architecture size do you start paying a steep price in training time for zero gain in chemical accuracy?

---

## Silver: Regularisation (Smoothing the Chemical Landscape)
Compare the training behavior and final validation performance of your best MLP architecture with and without **Dropout** layers. 

> ### Think:
> * Chemical space can be incredibly rugged. Dropout forces the network to learn redundant pathways for predicting properties rather than relying on a single "magic combination" of descriptors. Does this regularisation actually improve the model’s ability to predict properties for entirely new chemical families, or does it just slow down optimisation? 

---

## Gold: Optimisation Instability (When the Chemistry Breaks)
Intentionally break your model's training by increasing the learning rate by several orders of magnitude (e.g., set `lr = 0.5` or `1.0`). Inspect the loss curves and training logs closely.

> * *Analogy:* Think of the learning rate as temperature in a crystallisation experiment. If it's too low, the system lacks the energy to move and gets trapped in a poor, local minimum; if it's too high, the system is too chaotic to settle, preventing a well-ordered structure from forming. More specifically, the learning rate acts as a scaling factor (or step size) that determines how drastically the model adjusts its parameters during each optimisation update.