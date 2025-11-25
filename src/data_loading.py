from typing import Literal, Union
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

StereoSubset = Literal["intrasentence", "intersentence", "all"]


def load_stereoset(subset: StereoSubset = "intrasentence") -> Union[Dataset, DatasetDict]:
    """
    Load the StereoSet dataset from Hugging Face.

    Args:
        subset:
            - "intrasentence": intra-sentence context association tests
            - "intersentence": inter-sentence/discourse level tests
            - "all": returns a DatasetDict with both subsets

    Returns:
        Either a single Dataset (if subset != "all")
        or a DatasetDict with keys {"intrasentence", "intersentence"}.
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

    Returns:
        A datasets.Dataset with the same fields as the original HF version:
        - id (int)
        - sent_more (str)
        - sent_less (str)
        - stereo_antistereo (str: 'stereo' or 'antistereo')
        - bias_type (str: e.g. 'gender', 'race-color', ...)
    """
    url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

    df = pd.read_csv(url)

    # Make sure columns are what we expect
    # (CrowS-Pairs CSV already uses these names.)
    expected_cols = {"sent_more", "sent_less", "stereo_antistereo", "bias_type"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CrowS-Pairs CSV is missing columns: {missing}")

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds

def load_sst2() -> DatasetDict:
    """
    Load the SST-2 dataset from GLUE.

    Returns:
        A DatasetDict with splits: 'train', 'validation', 'test'.
    """
    ds = load_dataset("glue", "sst2")
    return ds
