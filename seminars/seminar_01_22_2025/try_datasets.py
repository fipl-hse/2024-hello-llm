"""
HuggingFace datasets listing.
"""
from pathlib import Path

try:
    from pandas import DataFrame
except ImportError:
    print('Library "pandas" not installed. Failed to import.')
    DataFrame = None  # type: ignore

try:
    from datasets import load_dataset
except ImportError:
    print('Library "datasets" not installed. Failed to import.')
    load_dataset = None  # type: ignore


def main() -> None:
    """
    Entrypoint for the listing.
    """
    # 1. Obtain dataset from HuggingFace
    df = load_dataset("IlyaGusev/gazeta", split="train", revision='v2.0').to_pandas()
    n_empty_rows = df.isna().sum(axis=1) + df[df == ''].sum(axis=1)
    sample_lens = df.text.apply(len)
    min_len, max_len = min(sample_lens), max(sample_lens)

    ds_properties = {
        'dataset_number_of_samples': df.shape[0],
        'dataset_columns': df.shape[1],
        'dataset_duplicates': df.duplicated().sum(),
        'dataset_empty_rows': n_empty_rows,
        'dataset_sample_min_len': min_len,
        'dataset_sample_max_len': max_len
    }
    print(ds_properties)

    # 2. Check dataset's subset
    # print(dataset.data.keys())
    #
    # # 3. Get needed subset
    # subset = dataset.get("validation")
    #
    # # 4. Get number of samples
    # print(f"Obtained dataset step-by-step: # of samples is {len(subset)}")
    #
    # # 5. Get dataset with particular subset at once
    # dataset = load_dataset("RussianNLP/russian_super_glue", name="danetqa", split="validation")
    # print(f"Obtained dataset with one call: # of samples is {len(dataset)}")
    #
    # # 6. Dataset without a name
    # dataset = load_dataset("sberquad", split="validation")
    # print(f"Obtained sberquad dataset with one call: # of samples is {len(dataset)}")
    #
    # # 7. Cast dataset to pandas
    # dataset_df: DataFrame = subset.to_pandas()
    #
    # # 8. Optionally save head of dataframe
    # (
    #     dataset_df.head(100).to_csv(
    #         Path(__file__).parent / "assets" / "danetqa_example.csv", index=False, encoding="utf-8"
    #     )
    # )


if __name__ == "__main__":
    main()
