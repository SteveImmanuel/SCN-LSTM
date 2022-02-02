import cv2
import os
import shutil
from typing import Tuple, Dict, List
from constant import *


def generate_epsilon(total_epoch: int) -> List[float]:
    """Generate sampling probability for each epoch for SAVC model

    Args:
        total_epoch (int):

    Returns:
        List:
    """
    res = []

    threshold = 50
    for i in range(threshold):
        res.append(1.0)

    for i in range(threshold, total_epoch):
        res.append(1 - (i - threshold + 1) / (total_epoch - threshold + 1) * (1 - 0.4))
    return res


def count_word_occurence(annotation_file: str) -> Dict:
    """Count all word occurence in annotation_file to build tags.

    Args:
        annotation_file (str): 

    Returns:
        Dict: 
    """
    res = {}
    with open(annotation_file, 'r') as annot:
        line = annot.readline()
        while line:
            line = line.strip('\n')
            tokens = line.split(' ')[1:]
            for token in tokens:
                if token in res:
                    res[token] += 1
                else:
                    res[token] = 1
            line = annot.readline()

    return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))


def build_tags(annotation_file: str, num_tags: int = SEMANTIC_SIZE, reverse_key: bool = False) -> Dict:
    """Build tag dictionary consisting of top num_tags words in terms of occurence
    in annotation_file

    Args:
        annotation_file (str): 
        num_tags (int, optional): . Defaults to 750.

    Returns:
        Dict: 
    """
    tags = count_word_occurence(annotation_file)
    for word in BLACKLIST_WORDS:
        tags.pop(word, None)

    tags = list(tags.items())[:num_tags]
    idx = 0

    res = {}
    for tag in tags:
        res[idx] = tag
        idx += 1

    if reverse_key:
        return {val: key for key, val in res.items()}
    return res


def build_video_dict(annotation_file: str, reverse_key: bool = False) -> Dict:
    """Create video index mapping

    Args:
        annotation_file (str): 

    Returns:
        Dict: 
    """
    video_dict = {}
    with open(annotation_file, 'r') as annot:
        line = annot.readline()
        idx = 0

        while line:
            line = line.strip('\n')
            tokens = line.split(' ')
            video_name = tokens[0]

            if video_name not in video_dict:
                video_dict[video_name] = idx
                idx += 1
            line = annot.readline()

    if reverse_key:
        return {val: key for key, val in video_dict.items()}

    return video_dict


def build_vocab(annotation_file: str) -> Tuple[Dict, Dict, Dict]:
    """Build vocab for annotation_file

    Args:
        annotation_file (str): 
    Returns:
        (word_to_idx dict, idx_to_word dict, video_mapping dict)
    """

    video_mapping = {}
    word_to_idx = {PAD_TAG: 0}
    # start from 1 so that index 0 is reserved for padding
    idx = 1

    with open(annotation_file, 'r') as annot:
        line = annot.readline()
        while line:
            line = line.strip('\n')
            tokens = line.split(' ')
            video_name = tokens[0]
            video_annot = [x.lower() for x in tokens[1:]]
            video_annot = [BOS_TAG] + video_annot + [EOS_TAG]

            for word in video_annot:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1

            if video_name in video_mapping:
                video_mapping[video_name].append(video_annot)
            else:
                video_mapping[video_name] = [video_annot]
            line = annot.readline()

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return (word_to_idx, idx_to_word, video_mapping)


def annotation_to_idx(annotation: List[str], word_to_idx: Dict) -> List[int]:
    """Converts str annotation into integer indexes

    Args:
        annotation (List[str]): 

    Returns:
        List[int]: 
    """
    return [word_to_idx[x] if x in word_to_idx else -1 for x in annotation]


def idx_to_annotation(idxs: List[int], idx_to_word: Dict) -> List[str]:
    """Converts integer indexes into annotation

    Args:
        annotation (List[str]): 

    Returns:
        List[int]: 
    """
    return [idx_to_word[x] if x in idx_to_word else UNKNOWN_TAG for x in idxs]


def video_to_frames(root_path: str = '.', output_dim: Tuple = (224, 224)) -> None:
    """Converts all videos in root_path to sequence of images to each own directory

    Args:
        root_path (str, optional): . Defaults to '.'.
        output_dim (Tuple, optional): . Defaults to (224, 224).
    """
    allowed_ext = ['.avi', '.mp4']

    all_videos = [video for video in os.listdir(root_path) if os.path.splitext(video)[-1] in allowed_ext]
    total_videos = len(all_videos)
    print(f'Found {total_videos} videos')

    for idx, video in enumerate(all_videos):
        print(f'Processing {idx}/{total_videos}', video, end='\r')

        vid_cap = cv2.VideoCapture(f'{root_path}/{video}')
        out_dir = f'{root_path}/{os.path.splitext(video)[0]}'
        os.makedirs(out_dir, exist_ok=True)

        count = 1
        success, image = vid_cap.read()
        while (success):
            image = cv2.resize(image, output_dim)
            cv2.imwrite(f'{out_dir}/{count:03}.jpg', image)
            success, image = vid_cap.read()
            count += 1


def frames_to_video(root_path: str = '.') -> None:
    """Converts all sequence of frames inside each subdirectory in root_path
    to videos. This is the reverse of video_to_frames function

    Args:
        root_path (str, optional): . Defaults to '.'.
    """
    allowed_ext = ['.tif', '.jpg', '.jpeg', '.png']

    all_videos = [video for video in next(os.walk(root_path))][1]
    print(f'Found {len(all_videos)} videos')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24

    for video in all_videos:
        print('Processing', video)
        all_frames = sorted(
            [frame for frame in os.listdir(f'{root_path}/{video}') if os.path.splitext(frame)[-1] in allowed_ext])

        if len(all_frames) > 0:
            temp_frame = cv2.imread(f'{root_path}/{video}/{all_frames[0]}')
            height, width, channel = temp_frame.shape
            video_writer = cv2.VideoWriter(f'{root_path}/{video}.mp4', fourcc, fps, (width, height))

            for frame_path in all_frames:
                frame = cv2.imread(f'{root_path}/{video}/{frame_path}')
                video_writer.write(frame)
            video_writer.release()


def split_train_val_test(root_path: str = '.') -> None:
    """Split data into traininig, validation, and test
    only call this function after all videos has been converted into frames to each own subdirectory
    by calling video_to_frames. Video 1-1200 will be for training, video 1201-1300 will be for validation,
    and video 1301-1970 will be for testing 

    Args:
        root_path (str, optional): . Defaults to '.'.
    """
    videos = sorted(os.listdir(root_path))
    subdir = {'train_val': [0, 1299], 'train': [0, 1199], 'validation': [1200, 1299], 'testing': [1300, 1969]}

    for subdir_name, index_range in subdir.items():
        os.makedirs(os.path.join(root_path, subdir_name), exist_ok=True)
        for i in range(index_range[0], index_range[1] + 1):
            source_path = os.path.join(root_path, videos[i])
            dest_path = os.path.join(root_path, subdir_name, videos[i])

            if subdir_name == 'train_val':
                shutil.copy(source_path, dest_path)
            else:
                shutil.move(source_path, dest_path)


if __name__ == '__main__':
    cnn2d_model = ['regnetx32', 'vgg']
    cnn3d_model = ['shufflenetv2', 'shufflenet']

    for model_2d in cnn2d_model:
        for model_3d in cnn3d_model:
            split_train_val_test(f'D:/ML Dataset/MSVD/features/{model_2d}_{model_3d}/semantics')
