import pathlib

def makedir_if_not_exists(file_path, is_path_dir=True):
    path = pathlib.Path(file_path)
    if is_path_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)