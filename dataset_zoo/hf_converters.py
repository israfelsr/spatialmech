from datasets import Dataset, Image as HFImage
from dataset_zoo.aro_datasets import COCO_QA
import os
import re


def convert_coco_qa_to_hf(coco_qa_dataset, has_two_objects=True):
    data = {
        "image": [],
        "caption_correct": [],
        "caption_incorrect": [],
        "preposition": [],
        "objects": [],
    }
    for i, sample in enumerate(coco_qa_dataset):
        caption_correct = sample["caption_options"][0]
        caption_incorrect = sample["caption_options"][1:]
        if not has_two_objects:
            match = re.search(r"A photo of a (.+?) on", caption_correct)
            data["objects"].append(match.group(1))
        else:
            match = re.search(
                r"A photo of a (.+?) to the (?:left|right) of a (.+?)$", caption_correct
            )
            if not match:
                match = re.search(
                    r"A photo of a (.+?) (?:above|below) a (.+?)$", caption_correct
                )
            data["objects"].append([match.group(1), match.group(2)])

        data["image"].append(sample["image_options"])
        data["caption_correct"].append(caption_correct)
        data["caption_incorrect"].append(caption_incorrect)
        data["preposition"].append(coco_qa_dataset.all_prepositions[i])

    hf_dataset = Dataset.from_dict(data)
    hf_dataset = hf_dataset.cast_column("image", HFImage())
    return hf_dataset
