# FT-DINOSAUR Inference

## Setup

This package only requires `torch` and `torchvision` as dependencies.
Simply install the package using pip:

```bash
pip install .
```

We recommend using an environment manager like virtualenv or conda.

## Quick Start

Import the package and print the list of defined models:

```Python
from ftdinosaur_inference import build_dinosaur
print(build_dinosaur.list_models)
```

We provide pre-trained checkpoints. Run the following to get a list of available checkpoints

```Python
print(build_dinosaur.list_checkpoints())
```

A model can be built by passing the model name to the `build` function:

```Python
model_name = "dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k"
model = build_dinosaur.build(model_name)
```

By default, this will load a pre-trained checkpoint if it exists.

Then define the pre-processing pipeline that turns a numpy or pillow image into a torch tensor usable as the model's input:

```Python
preproc = build_dinosaur.build_preprocessing(model_name)
```

run the model:

```Python
from PIL import Image
image = Image.open("./images/gizmos.jpg").convert("RGB")
inp = preproc(image).unsqueeze(0)
outp = model(inp)
```

That's it! The model returns a dictionary with the output of the different modules.
See the [DINOSAUR module](ftdinosaur_inference/modules/dinosaur.py) for an overview.

## Example Notebook

An example notebook with mask visualization is contained in [here](notebooks/example.ipynb).
Install with the `notebook` optional dependencies and start a Jupyter server to view it:

```bash
pip install .[notebook]
jupyter notebook
```

## List of Model Checkpoints

The model key follows the convention `<model-architecture>.<checkpoint>`.

| Model   | Model Key | ViT | Input Size | Description |
| ------- | ------ | ------ | ------ | ------ |
| FT-DINOSAUR | dinosaur_small_patch14_224_topk3.coco_dv2_ft_s7_300k | small14 | 224x224 | Top-k decoding
| FT-DINOSAUR | dinosaur_small_patch14_518_topk3.coco_dv2_ft_s7_300k+10k | small14 | 518x518 |  Top-k decoding, hi-res finetuned
| FT-DINOSAUR | dinosaur_base_patch14_224_topk3.coco_dv2_ft_s7_300k | base14 | 224x224 | Top-k decoding.
| FT-DINOSAUR | dinosaur_base_patch14_518_topk3.coco_dv2_ft_s7_300k+10k | base14 | 518x518 | Top-k decoding, hi-res finetuned


## Development

Development dependencies can be installed using

```bash
pip install .[dev]
```

Tests can be run using `pytest`. The codebase uses `pre-commit` to manage linting.

## Citation

If you make use of this work, please cite us:

```
@misc{Didolkar2024ZeroShotOCRL,
  title={Zero-Shot Object-Centric Representation Learning},
  author={Didolkar, Aniket and Zadaianchuk, Andrii and Goyal, Anirudh and Mozer, Mike and Bengio, Yoshua and Martius, Georg and Seitzer, Maximilian},
  year={2024},
}
```

## Licenses

This codebase is released under Apache License 2.0.

This codebase adapts parts of the [timm library](https://github.com/huggingface/pytorch-image-models) and the [OCLF library](https://github.com/amazon-science/object-centric-learning-framework).
The concerned source files contain a note of their origin.
Both are released under Apache License 2.0.
We like to thank their authors!

The [*example image*](notebooks/images/example.jpg) is from the [COCO dataset](https://cocodataset.org/#explore?id=442321), originally from [Flickr](http://farm8.staticflickr.com/7003/6649994945_c5e92895f7_z.jpg), under [CC BY-NC-SA 2.0 license](https://creativecommons.org/licenses/by-nc-sa/2.0/).
The [*"Gizmos" image*](notebooks/images/gizmos.jpg) is from [Greff, van Steenkiste, Schmidhuber, 2020: On the Binding Problem in Artificial Neural Networks](https://arxiv.org/abs/2012.05208), under [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
