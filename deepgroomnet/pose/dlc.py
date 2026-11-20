import numpy as np


def analyze_keypoints(video, project_config, dest_folder, videotype="mp4"):
    try:
        import deeplabcut
    except ImportError as error:
        raise RuntimeError("DeepLabCut is required for keypoint extraction; "
                           "install it in a separate environment") from error
    deeplabcut.analyze_videos(project_config, [str(video)], videotype=videotype,
                              save_as_csv=True, destfolder=str(dest_folder))
    return dest_folder


def keypoints_to_array(csv_path, n_frames=None):
    import pandas as pd
    header = pd.read_csv(csv_path, header=None, nrows=3)
    bodyparts = list(header.iloc[1, 1:])
    coords = list(header.iloc[2, 1:])
    data = pd.read_csv(csv_path, skiprows=3, header=None)
    data = data.iloc[:, 1:1 + len(bodyparts)]
    array = data.to_numpy(np.float32)
    if n_frames is not None:
        if len(array) < n_frames:
            pad = np.full((n_frames - len(array), array.shape[1]), np.nan, np.float32)
            array = np.vstack([array, pad])
        array = array[:n_frames]
    return array, list(zip(bodyparts, coords))
