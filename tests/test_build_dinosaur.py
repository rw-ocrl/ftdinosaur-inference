import itertools

import torch

from ftdinosaur_inference import build_dinosaur


def test_build():
    for model_name in build_dinosaur.list_models():
        model = build_dinosaur.build(model_name, pretrained=False)

        # Sanity check that model works with the configured input size.
        config = build_dinosaur.get_config(model_name)
        inp = torch.randn(1, *config["input_size"])
        model(inp)


def test_build__pretrained():
    for model_name in build_dinosaur.list_checkpoints():
        _ = build_dinosaur.build(model_name, pretrained=True)


def test_build_preprocessing():
    for model_name in itertools.chain(
        build_dinosaur.list_models(), build_dinosaur.list_checkpoints()
    ):
        _ = build_dinosaur.build_preprocessing(model_name)
