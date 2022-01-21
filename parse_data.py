import argparse
from s2vt.feature_extractor import parse_features_from_txt

parser = argparse.ArgumentParser(description='Parse CNN Features from .txt File')
parser.add_argument('--feature-file', help='File path to feature file', required=True)
parser.add_argument('--output-dir', help='Path for output directory', required=True)

args = parser.parse_args()
feature_file = args.feature_file
output_dir = args.output_dir

parse_features_from_txt(feature_file, output_dir)
