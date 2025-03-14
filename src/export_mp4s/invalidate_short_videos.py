import shutil
import multiprocessing
import os
import subprocess
from glob import glob
from tqdm import tqdm
"""This script just renames files that have < 64 frames (and so can't be used to train 3D CNNs)
from *.mp4 to *.mp4.short"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
PATH = os.path.join(PARENT_DIR, "data", "face_videos_by_part")
MIN_FRAMES = 64

def check_and_move_video(videopath):
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=nb_frames",  
            "-of", "default=noprint_wrappers=1:nokey=1", 
            videopath
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        total_frames = int(result.stdout.strip())
        print(f"Video {videopath} has {total_frames} frames")
        if total_frames < MIN_FRAMES:
            print(f'Moving {videopath}')
            shutil.move(videopath, videopath + '.short')
    except Exception as e:
        print(f"Error reading {videopath}: {e}")

if __name__ == "__main__":
    N_WORKERS = multiprocessing.cpu_count()
    
    videopaths = glob(os.path.join(PATH, '**', '*.mp4'), recursive=True)
    
    if not videopaths:
        print(f"No video files found in {PATH}. Please check the directory.")
    else:
        with multiprocessing.Pool(N_WORKERS) as p:
            for _ in tqdm(p.imap(check_and_move_video, iter(videopaths)), total=len(videopaths)):
                pass

