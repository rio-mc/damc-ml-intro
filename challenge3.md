# Hard Challenge: Deep Learning for Chemical Data

## Bronze: Identifying Overfitting in Spectral or Structural Profiles
Run the baseline Convolutional Neural Network (CNN) on your structural data (such as XRD patterns or spectra) and inspect the training and validation loss curves.

> ### Think:
> * At what precise epoch does the validation performance stop improving and start to diverge? 
> 
> ### Intervention:
> Implement and evaluate at least one of the following strategies to penalise memorisation:
> * **Reduce model capacity:** Lower the number of convolutional filters.
> * **Increase structural dropout:** Force the network to learn redundant feature maps.
> * **Enable early stopping:** Terminate training before the model begins tracking noise.
> 
> *Did your generalisation to the validation set improve, or did the model underfit the chemical features entirely?*

---

## Silver: Architectural Trade-offs and Physical Constraints
Modify the baseline CNN architecture to optimise its efficiency. Your constraint is practical utility: the model must remain lightweight enough for rapid CPU evaluation in a typical chemistry lab setup.

Compare at least two distinct architectures by systematically altering:
* **Channel counts** (width of feature extraction)
* **Kernel sizes** (receptive field, such as wide kernels for broad amorphous halos versus narrow kernels for sharp crystalline peaks)
* **Layer depth** (hierarchical feature complexity)
* **Regularisation profiles**

> ### Think:
> * Your goal is not maximum training accuracy. Instead, provide a scientific justification for the architecture that strikes the best balance between:
>   1. Validation performance
>   2. Train-validation gap (generalisation)
>   3. Operational robustness
>   4. Training runtime and CPU overhead
>   5. Architectural simplicity (Occam's razor for models)

---

## Gold: Scientific Robustness Audit and Edge Cases
In the real world, experimental chemical data is messy. Instrument calibrations drift, samples contain impurities, and signal-to-noise ratios vary. Stress-test your best-performing CNN architecture by intentionally perturbing the data or training environment.

* **Peak-shift augmentation:** Add synthetic noise or slight horizontal shifts to the peaks to simulate thermal expansion or minor instrument miscalibration.
* **Amorphous noise injection:** Add a high noise floor to simulate poorly crystalline or impure samples.
* **Unscheduled optimisation:** Disable learning-rate scheduling to see if the model can settle into a clean global minimum without manual dampening.

> ### Think:
> * **Does the model fail gracefully?** When performance drops, does it degrade mildly or collapse entirely?
> * **Does validation behavior reliably predict test behavior** under stress, or does the validation set mask hidden vulnerabilities?
> * **Which specific crystal systems or symmetries become harder to classify** under these perturbations? What is the structural or physical reason behind this vulnerability?
