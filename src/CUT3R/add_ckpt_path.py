import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    repo_src_path = path.join(path.dirname(path.abspath(__file__)), "src")
    if repo_src_path not in sys.path:
        sys.path.insert(0, repo_src_path)

    HERE_PATH = os.path.dirname(os.path.abspath(ckpt))
    # workaround for sibling import
    if HERE_PATH not in sys.path:
        sys.path.insert(0, HERE_PATH)
