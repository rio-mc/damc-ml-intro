import torch


def augment_xrd(x, noise_std: float, shift_max: int):
    """
    Apply simple XRD-like perturbations during training.

    noise_std:
        adds small intensity noise

    shift_max:
        randomly shifts each pattern left or right by a few grid points

    These perturbations test whether the model learns robust diffraction
    features rather than memorising exact peak positions.
    """

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    if shift_max > 0:
        shifts = torch.randint(
            -shift_max,
            shift_max + 1,
            size=(x.shape[0],),
            device=x.device,
        )

        x = torch.stack(
            [
                torch.roll(pattern, int(shift.item()), dims=-1)
                for pattern, shift in zip(x, shifts)
            ]
        )

    return x