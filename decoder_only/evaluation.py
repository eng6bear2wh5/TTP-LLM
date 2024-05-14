import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import argparse

parser = argparse.ArgumentParser(description="MITRE Evaluation")
parser.add_argument('--encoded_file_path', type=str, default='./results/preds_gpt-3.5-turbo-1106_all_urls_encoded.csv', help='Specify the path to the encoded file for evaluation')
args = parser.parse_args()

def eval(preds, labels):
    mitre_tactics=['collection','command and control','credential access','defense evasion','discovery','execution','exfiltration','impact','initial access','lateral movement','persistence','privilege escalation','reconnaissance','resource development']
    report = classification_report(labels, preds,target_names=mitre_tactics)
    acc = accuracy_score(labels, preds)
    return report, acc

if __name__=="__main__":
    # TO RUN: python evaluation.py --encoded_file_path ./Results/preds_gpt-3.5-turbo-1106_all_urls_encoded.csv
    labels_df = pd.read_csv('./data/MITRE_Procedures_encoded.csv')
    labels = labels_df.values
    preds_df = pd.read_csv(args.encoded_file_path)
    preds = preds_df.values
    report, acc = eval(preds, labels)
    print(report)
    print(acc)
