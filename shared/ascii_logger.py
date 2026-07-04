def log_random_forest_ascii(cfg: dict) -> None:
    features = cfg["features"]["columns"]
    model = cfg["model"]

    print("\nArchitecture sketch:")
    print("------------------------------------------------------------")
    print("Descriptor table")
    print(f"  rows x {len(features)} selected features")
    print(f"  columns: {', '.join(features)}")
    print("        |")
    print("        v")
    print(f"Random forest ({model['n_estimators']} decision trees)")
    print(f"  max_depth        = {model['max_depth']}")
    print(f"  min_samples_leaf = {model['min_samples_leaf']}")
    print("        |")
    print("        v")
    print("Average tree predictions")
    print("        |")
    print("        v")
    print(f"Predicted {cfg['data']['target_column']}")


def log_mlp_ascii(cfg: dict, input_dim: int | None = None) -> None:
    features = cfg["features"]["columns"]
    model = cfg["model"]
    layer_dims = [input_dim or len(features), *model["hidden_dims"], 1]

    print("\nArchitecture sketch:")
    print("------------------------------------------------------------")
    print("Descriptor vector")
    print(f"  {len(features)} configured features: {', '.join(features)}")
    print("        |")

    for layer_index, (in_dim, out_dim) in enumerate(zip(layer_dims, layer_dims[1:]), start=1):
        label = "output" if out_dim == 1 else f"hidden {layer_index}"
        print("        v")
        print(f"Linear({in_dim} -> {out_dim})  [{label}]")
        if out_dim != 1:
            print("        |")
            print("        v")
            print("ReLU")
            if model["dropout"]:
                print("        |")
                print("        v")
                print(f"Dropout(p={model['dropout_p']})")
        if out_dim != layer_dims[-1]:
            print("        |")

    print("        |")
    print("        v")
    print(f"Predicted {cfg['data']['target_column']}")


def log_cnn_ascii(
    cfg: dict,
    n_points: int | None = None,
    n_classes: int | None = None,
    theta=None,
) -> None:
    model = cfg["model"]
    channels = model["channels"]
    input_label = f"1 x {n_points} intensity values" if n_points is not None else "1 x N intensity values"
    class_label = f"{n_classes} class scores" if n_classes is not None else "class scores"
    theta_step = _average_step(theta)
    kernel_width = model["kernel_size"] * theta_step if theta_step is not None else None

    print("\nArchitecture sketch:")
    print("------------------------------------------------------------")
    print("XRD pattern")
    print(f"  shape: {input_label}")
    print("        |")

    in_channels = 1
    for block_index, out_channels in enumerate(channels, start=1):
        print("        v")
        print(
            f"Conv block {block_index}: "
            f"Conv1d({in_channels} -> {out_channels}, kernel={model['kernel_size']})"
        )
        print("        |")
        print("        v")
        print("ReLU")
        print("        |")
        print("        v")
        print("MaxPool1d(2)")
        print("        |")
        in_channels = out_channels

    print("        v")
    print("Flatten learned peak features")
    print("        |")
    print("        v")
    print("Linear(flattened features -> 32)")
    print("        |")
    print("        v")
    print("ReLU")
    if model["dropout"]:
        print("        |")
        print("        v")
        print(f"Dropout(p={model['dropout_p']})")
    print("        |")
    print("        v")
    print(f"Linear(32 -> {class_label})")


def _average_step(values) -> float | None:
    if values is None or len(values) < 2:
        return None

    steps = [float(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return sum(steps) / len(steps)
