"""
Dataset loading helpers.
Wraps HuggingFace for SST-2, CrowS-Pairs, and StereoSet into
simple convenience functions.
"""
from typing import Literal, Union
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

StereoSubset = Literal["intrasentence", "intersentence", "all"]


def load_stereoset(subset: StereoSubset = "intrasentence") -> Union[Dataset, DatasetDict]:
    """
    Load the StereoSet dataset from Hugging Face.
    """
    if subset == "all":
        ds_intra = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
        ds_inter = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
        return DatasetDict(intrasentence=ds_intra, intersentence=ds_inter)

    if subset not in ("intrasentence", "intersentence"):
        raise ValueError(f"Unsupported StereoSet subset: {subset}")

    return load_dataset("McGill-NLP/stereoset", subset, split="validation")


def load_crows_pairs() -> Dataset:
    """
    Load the CrowS-Pairs dataset directly from the official GitHub CSV.
    """
    url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

    df = pd.read_csv(url)

    expected_cols = {"sent_more", "sent_less", "stereo_antistereo", "bias_type"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CrowS-Pairs CSV is missing columns: {missing}")

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds

def load_sst2() -> DatasetDict:
    """
    Load the SST-2 dataset from GLUE.
    """
    ds = load_dataset("glue", "sst2")
    return ds
