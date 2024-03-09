#!/usr/local/bin/python3

import os
import shutil

dirs = ['simulate','format','train','estimate','plot','empirical']
for d in dirs:
    test_dir = f'./workspace/test/{d}'
    valid_dir = f'./workspace/valid/{d}'
    if os.path.exists(valid_dir):
        print(f'Remove {valid_dir}')
        shutil.rmtree(valid_dir)
    if os.path.exists(test_dir):
        print(f'Copy {test_dir} to {valid_dir}')
        shutil.copytree(test_dir, valid_dir)
