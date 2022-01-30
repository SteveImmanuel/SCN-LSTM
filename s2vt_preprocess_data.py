import argparse
from s2vt.feature_extractor import extract_features

parser = argparse.ArgumentParser(description='Extract features using VGG Model')
parser.add_argument('--annotation-path', help='File path to annotation', required=True)
parser.add_argument('--output-dir', help='Path for output directory', required=True)
parser.add_argument(
    '--dataset-data-dir',
    help='Directory containing subdirectories containing all video frames',
    required=True,
)
args = parser.parse_args()
annotation_path = args.annotation_path
output_dir = args.output_dir
dataset_data_dir = args.dataset_data_dir

extract_features(annotation_path, dataset_data_dir, output_dir)
