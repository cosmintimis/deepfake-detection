import os
import pathlib

def get_base_dir():
    return str(pathlib.Path(__file__).resolve().parent)