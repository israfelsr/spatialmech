import os
import torch
from PIL import Image
from torchvision import transforms


def get_model(model_name, device, method="base", root_dir="data"):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """

    if model_name == "llava1.5":
        from .llava15 import LlavaWrapper

        llava_model = LlavaWrapper(root_dir=root_dir, device=device, method=method)
        image_preprocess = None
        return llava_model, image_preprocess

    elif model_name == "qwen2vl":
        from .qwen import QwenWrapper

        qwen_model = QwenWrapper(root_dir=root_dir, device=device, method=method)
        image_preprocess = None
        return qwen_model, image_preprocess

    elif model_name == "qwen2vl-vllm":
        from .qwen_vllm import QwenVLLMWrapper

        qwen_model = QwenVLLMWrapper(root_dir=root_dir, device=device, method=method)
        image_preprocess = None
        return qwen_model, image_preprocess

    elif model_name == "paligemma":
        from .paligemma import PaligemmaWrapper

        paligemma_model = PaligemmaWrapper(
            root_dir=root_dir, device=device, method=method
        )
        image_preprocess = None
        return paligemma_model, image_preprocess

    elif model_name == "gemma3":
        from .gemma3 import Gemma3Wrapper

        gemma3_model = Gemma3Wrapper(
            root_dir=root_dir, device=device, method=method
        )
        image_preprocess = None
        return gemma3_model, image_preprocess

    else:
        raise ValueError(f"Unknown model {model_name}")
