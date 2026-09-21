import pandas as pd
import numpy as np

def balance_word_count(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df['word_count'] = new_df['sentence'].str.findall(r'[\w-]+').str.len()

    while True:
        mean_false = new_df[new_df['is_spurious'] == False]['word_count'].mean()
        mean_true = new_df[new_df['is_spurious'] == True]['word_count'].mean()

        if abs(mean_false - mean_true) <= 1.0:
            break

        if mean_false > mean_true:
            subset = new_df[new_df['is_spurious'] == False]
            threshold = subset['word_count'].quantile(0.9)
            candidates = subset[subset['word_count'] >= threshold].index
        else:
            subset = new_df[new_df['is_spurious'] == True]
            threshold = subset['word_count'].quantile(0.9)
            candidates = subset[subset['word_count'] >= threshold].index

        if len(candidates) == 0:
            break

        drop_idx = np.random.choice(candidates)
        new_df = new_df.drop(drop_idx)

    return new_df

def trim_entries(entries: pd.DataFrame, limit: float = 0.50) -> pd.DataFrame:
    if 'is_spurious' in entries:
        spurious_mask = entries["is_spurious"] == True
    else:
        spurious_mask = entries['No error'] == False
    
    num_spurious = spurious_mask.astype(int).sum()
    total_rows = len(entries)

    current_ratio = num_spurious / total_rows

    if current_ratio <= limit:
        return entries

    num_to_drop = int(((num_spurious - limit * total_rows) / (1 - limit)) + 1)
    spurious_indices = entries[spurious_mask].index
    indices_to_drop = entries.loc[spurious_indices].sample(n=num_to_drop).index
    df_cleaned = entries.drop(indices_to_drop)
    return df_cleaned


if __name__ == "__main__":
    print("ligmaine eier")
