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
    return hf_dataset


def convert_controlled_to_hf(controlled_dataset):
    data = {
        "image": [],
        "caption_correct": [],
        "caption_incorrect": [],
        "preposition": [],
        "objects": [],
    }
    for i, sample in enumerate(controlled_dataset):
        caption_correct = sample["caption_options"][0]
        caption_incorrect = sample["caption_options"][1:]
        # Updated regex to handle various prepositions
        # Supports: to the left/right of, in front of, behind, on, under, above, below
        match = re.search(
            r"A (.+?) (?:to the (?:left|right) of|in front of|behind|on|under|above) a (.+?)$",
            caption_correct,
        )
        data["objects"].append([match.group(1), match.group(2)])

        data["image"].append(sample["image_options"])
        data["caption_correct"].append(caption_correct)
        data["caption_incorrect"].append(caption_incorrect)
        data["preposition"].append(controlled_dataset.all_prepositions[i])

    hf_dataset = Dataset.from_dict(data)
    return hf_dataset
