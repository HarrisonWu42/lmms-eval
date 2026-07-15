import os


def get_cache_dir(config, sub_dir="videos"):
    dataset_path = config.get("dataset_path")
    if dataset_path:
        dataset_path = os.path.expanduser(dataset_path)
        local_cache_dir = os.path.join(dataset_path, sub_dir)
        if os.path.isdir(local_cache_dir):
            return local_cache_dir

    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface/")
    cache_dir = config["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(os.path.expanduser(hf_home), cache_dir)
    cache_dir = os.path.join(cache_dir, sub_dir)
    return cache_dir


def _get_video_file(prefix: str, video_name: str, suffix: str):
    if not isinstance(video_name, str):
        video_name = str(video_name)
    if not video_name.endswith(suffix):
        video_name = f"{video_name}.{suffix}"
    video_path = os.path.join(prefix, video_name)
    return video_path


def get_video(prefix: str, video_name: str, suffix: str = "mp4"):
    tried = [os.path.abspath(_get_video_file(prefix, video_name, suffix)), os.path.abspath(_get_video_file(prefix, video_name, suffix.upper())), os.path.abspath(_get_video_file(prefix, video_name, suffix.lower()))]
    for video_path in tried:
        if os.path.exists(video_path):
            return video_path
    raise FileNotFoundError(f"Tried both {tried} but none of them exist, please check")
