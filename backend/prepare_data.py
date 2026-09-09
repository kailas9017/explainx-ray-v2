import pandas as pd
from sklearn.model_selection import train_test_split

from config import CSV_PATH, TRAIN_CSV, VAL_CSV, TEST_CSV, DEFAULT_PATHOLOGIES


def main():
    df = pd.read_csv(CSV_PATH)

    for pathology in DEFAULT_PATHOLOGIES:
        df[pathology] = df["Finding Labels"].apply(
            lambda x: 1.0 if pathology in x else 0.0
        )

    patient_ids = df["Patient ID"].unique()
    # Patient-level split (never image-level) to avoid leakage between splits —
    # the same patient's follow-up images must not end up in both train and test.
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    train_df = df[df["Patient ID"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Pathologies ({len(DEFAULT_PATHOLOGIES)}): {DEFAULT_PATHOLOGIES}")
    print(f"Train images: {len(train_df)}  ({train_df['Patient ID'].nunique()} patients)")
    print(f"Val images:   {len(val_df)}  ({val_df['Patient ID'].nunique()} patients)")
    print(f"Test images:  {len(test_df)}  ({test_df['Patient ID'].nunique()} patients)")
    print("Saved: train.csv, val.csv, test.csv -> backend/data/")


if __name__ == "__main__":
    main()
