import pandas as pd
import numpy as np

def get_revisions(df):
    """
    This function gets the original name and timestamp for all the revisions in the dataset, so it is possible to compare the time to remove any problematic content.

    -------
    Parameters: df (pandas DataFrame): The dataset extracted from download_wiki_revisions.py

    """
    # Create a data-set just for reversions
    reversions = df[df['comment'].str.lower().str.contains('(revert(ed)?|undid)', regex=True, na=False)].copy()

    # 1. EXTRACT IDS AND NAMES

    # Extract different ways to talk about reversions
    # 1: Ids are presented in "undid revision" (just removed the revision)
    reversions['reversion_number'] = reversions['comment'].str.lower().str.extract(r'undid revision (\d+)', expand=False)

    # 2: Ids are presented in "reverted to revision" (came back to the original revision)
    reversions['reversion_number'] = reversions['reversion_number'].fillna(
        reversions['comment'].str.lower().str.extract(r'revert(?:ed)? to revision (\d+)', expand=False)
    )

    # 3: Ids are presented as prior to revision (removed the revision)
    reversions['reversion_number'] = reversions['reversion_number'].fillna(
        reversions['comment'].str.lower().str.extract(r'prior to revision (\d+)', expand=False)
    )   

    # 4: No Ids present, we'd have to go by name
    reversions['reversion_name'] = reversions['comment'].str.extract(r'Special:(?:Contributions|Contribs)/([^|]+)', expand=False)


    # MAP THE DIFFERENT TYPES OF REVISION

    reversions_rule = [
        reversions['comment'].str.lower().str.contains('undid', regex=False, na=False),
        reversions['comment'].str.lower().str.contains('revert(?:ed)? to', regex=True, na=False),
        reversions['comment'].str.lower().str.contains('revert(?:ed)?', regex=True, na=False),
        reversions['comment'].str.lower().str.contains('prior to', regex=True, na=False),
    ]

    reversions_label = ['reverted', 'revert to', 'reverted', 'reverted']

    # np select allows us to use multiple criteria
    reversions['type_revision'] = np.select(reversions_rule, reversions_label, default='unknown')

    # 2. MERGE BACK TO DATASET

    # If we have the reversion ID we will merge with the original data

    id_revisions = df[['revision_id', 'timestamp', 'username']].rename(columns={'revision_id': 'reversion_number',
                                                                                    'timestamp': 'original_pub_time',
                                                                                    'username': 'original_user'})

    final_reversions = reversions.merge(id_revisions, on='reversion_number', how='left')

    def get_last_edit(user_id, current_ts):
        last_edit = df[(df['username'] == user_id) & (df['timestamp'] < current_ts)].sort_values('timestamp', ascending=False)
        if last_edit.empty:
            return None
        return last_edit.iloc[0]['timestamp']

    # Apply the function row by row using .apply
    final_reversions['original_pub_time'] = final_reversions.apply(
        lambda row: row['original_pub_time'] if pd.notnull(row['original_pub_time']) else get_last_edit(row['reversion_name'], row['timestamp']),
        axis=1
    )

    # 3. MAKE FINAL ADJUSTMENTS

    final_reversions['original_user'] = np.where(final_reversions['original_user'].isnull(), final_reversions['reversion_name'], final_reversions['original_user']) # Add the name for those that only have IDs
    final_reversions['time_to_correct'] = (final_reversions['timestamp'] - final_reversions['original_pub_time']) # Add the time in seconds to correct the mistake

    # 4. RETURN DATASET

    return final_reversions


