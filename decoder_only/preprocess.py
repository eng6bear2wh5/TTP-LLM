import pandas as pd
import re
import ast


def remove_citations(text):
    pattern = r'\[\d+\]'
    preprocessed_text = re.sub(pattern, '', str(text))
    return preprocessed_text


def convert_tactics_to_binary(df):
    mitre_tactics = [
        'COLLECTION', 'COMMAND_AND_CONTROL', 'CREDENTIAL_ACCESS',
        'DEFENSE_EVASION', 'DISCOVERY', 'EXECUTION', 'EXFILTRATION',
        'IMPACT', 'INITIAL_ACCESS', 'LATERAL_MOVEMENT', 'PERSISTENCE',
        'PRIVILEGE_ESCALATION', 'RECONNAISSANCE', 'RESOURCE_DEVELOPMENT'
    ]

    # Initialize an empty DataFrame with desired columns
    encoded_df = pd.DataFrame(columns=[
        'collection', 'command and control', 'credential access', 'defense evasion',
        'discovery', 'execution', 'exfiltration', 'impact', 'initial access',
        'lateral movement', 'persistence', 'privilege escalation',
        'reconnaissance', 'resource development'
    ])

    for index, row in df.iterrows():
        # Concatenate tactics for the current row
        tactics = ', '.join([row[tactic] for tactic in ['Tactic1', 'Tactic2'] if pd.notna(row[tactic])])

        # Encode each tactic as 1 or 0
        encoded_row = [1 if tactic in tactics else 0 for tactic in mitre_tactics]
        encoded_df.loc[index] = encoded_row

    return encoded_df


def preprocess_citations(df):
    df['Description'] = df['Description'].apply(remove_citations)
    df.to_csv('./data/MITRE_Tactic_and_Techniques_Descriptions.csv', index=False)


def is_url_in_list(url, url_list_str):
    # Convert single quotes to double quotes for JSON parsing
    url_list_str = url_list_str.replace("'", '"')
    url_list = ast.literal_eval(url_list_str)
    return url in url_list


def process_procedure_urls(df):
    df['Retrieved Procedure URLs'] = df['Retrieved Procedure URLs'].apply(lambda x: x.replace("'", '"'))
    df_match = df[df.apply(lambda row: is_url_in_list(row['Procedure URL'], row['Retrieved Procedure URLs']), axis=1)]
    df_not_match = df[~df.apply(lambda row: is_url_in_list(row['Procedure URL'], row['Retrieved Procedure URLs']), axis=1)]

    # Exporting the DataFrames to CSV files
    df_match.to_csv('./data/procedures_matched_urls.csv', index=False)
    df_not_match.to_csv('./data/procedures_not_matched_urls.csv', index=False)

    # Concatenate the two DataFrames
    df['Label'] = df.apply(lambda row: 'matched' if is_url_in_list(row['Procedure URL'], row['Retrieved Procedure URLs']) else 'not_matched', axis=1)
    df_combined = df
    df_combined.to_csv('./data/procedures_similarity.csv', index=False)


def split_tactics_column(df):
    df['Tactic(s)'] = df['Tactic(s)'].apply(lambda x: ast.literal_eval(x.replace('nan', 'None')))

    # Determine the maximum list length to know how many columns to create
    max_len = df['Tactic(s)'].str.len().max()

    # Create new columns for each possible item in the lists
    for i in range(max_len):
        column_name = f'Tactic{i+1}'
        df[column_name] = df['Tactic(s)'].apply(lambda x: x[i] if i < len(x) and x[i] is not None else None)

    # Drop the original 'Tactic(s)' column if no longer needed
    df.drop('Tactic(s)', axis=1, inplace=True)
    df.to_csv('./data/procedures_similarity_main.csv', index=False)


if __name__ == '__main__':
    # Uncomment these lines if preprocessing the initial data
    # df = pd.read_csv("./data/MITRE_Tactic_and_Techniques_Descriptions.csv", encoding='ISO-8859-1')
    # preprocess_citations(df)

    df = pd.read_csv("./data/MITRE_Procedures.csv")
    encoded_df = convert_tactics_to_binary(df)
    encoded_df.to_csv('./data/MITRE_Procedures_encoded.csv', index=False)

    #df = pd.read_csv('./data/MITRE_Procedures.csv')
    #process_procedure_urls(df)
    #df = pd.read_csv('./data/procedures_similarity.csv')
    #split_tactics_column(df)