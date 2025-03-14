import cv2
import numpy as np
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_base_dir



def load_video(
    filename,
    every_n_frames=None,
    specific_frames=None,
    to_rgb=True,
    rescale=None,
    inc_pil=False,
    max_frames=None,
):
    """Loads a video.
    Called by:

    1) The finding faces algorithm where it pulls a frame every FACE_FRAMES frames up to MAX_FRAMES_TO_LOAD at a scale of FACEDETECTION_DOWNSAMPLE, and then half that if there's a CUDA memory error.

    2) The inference loop where it pulls EVERY frame up to a certain amount which it the last needed frame for each face for that video
    """

    assert (
        every_n_frames or specific_frames
    ), "Must supply either every n_frames or specific_frames"
    assert bool(every_n_frames) != bool(
        specific_frames
    ), "Supply either 'every_n_frames' or 'specific_frames', not both"

    cap = cv2.VideoCapture(filename)
    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if rescale:
        rescale = rescale * 1920.0 / np.max((width_in, height_in))

    width_out = int(width_in * rescale) if rescale else width_in
    height_out = int(height_in * rescale) if rescale else height_in
    
    print(f"Loading {filename} with {n_frames_in} frames at {width_in}x{height_in} rescaled to {width_out}x{height_out}")

    if max_frames:
        n_frames_in = min(n_frames_in, max_frames)

    if every_n_frames:
        specific_frames = list(range(0, n_frames_in, every_n_frames))

    n_frames_out = len(specific_frames)

    out_pil = []

    out_video = np.zeros((n_frames_out, height_out, width_out, 3), np.dtype("uint8"))

    i_frame_in = 0
    i_frame_out = 0
    ret = True

    while i_frame_in < n_frames_in and ret:

        try:
            try:

                if every_n_frames == 1:
                    ret, frame_in = cap.read()  # Faster if reading all frames
                else:
                    ret = cap.grab()

                    if i_frame_in not in specific_frames:
                        i_frame_in += 1
                        continue

                    ret, frame_in = cap.retrieve()
                    
                if not ret or frame_in is None:
                    print(f"Warning: Empty frame at {i_frame_in}, using black frame.")
                    frame_in = np.zeros((height_out, width_out, 3), dtype=np.uint8)
                if rescale:
                    frame_in = cv2.resize(frame_in, (width_out, height_out))
                if to_rgb:
                    frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                if frame_in.shape != (height_out, width_out, 3):
                    print(f"Error: Frame {i_frame_in} has unexpected shape {frame_in.shape}, skipping.")
                    frame_in = np.zeros((height_out, width_out, 3), dtype=np.uint8)

            except Exception as e:
                print(
                    f"Error for frame {i_frame_in} for video {filename}: {e}; using 0s"
                )
                frame_in = np.zeros((height_out, width_out, 3))

            out_video[i_frame_out] = frame_in
            i_frame_out += 1

            if inc_pil:
                try:  # https://www.kaggle.com/zaharch/public-test-errors
                    pil_img = Image.fromarray(frame_in)
                except Exception as e:
                    print(
                        f"Using a blank frame for video {filename} frame {i_frame_in} as error {e}"
                    )
                    pil_img = Image.fromarray(
                        np.zeros((224, 224, 3), dtype=np.uint8)
                    )  # Use a blank frame
                out_pil.append(pil_img)

            i_frame_in += 1

        except Exception as e:
            print(f"Error for file {filename}: {e}")

    cap.release()
    
    if inc_pil:
        return out_video, out_pil, rescale
    else:
        return out_video, rescale
