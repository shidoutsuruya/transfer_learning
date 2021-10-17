import numpy as np
import os
from pyhanlp import *
root=r'D:\Python_data\IMDB'
negative_files_path=os.path.join(root,'train','neg')
positive_files_path=os.path.join(root,'train','pos')
unsupervised_files_path=os.path.join(root,'train','unsup')
def load_data(dir_path):
    lst=[]
    files=os.listdir(dir_path)
    for file in files:
        with open(os.path.join(dir_path,file),encoding='utf-8') as f:
            lst.append(f.read())
    return lst
negative_data=load_data(negative_files_path)
positive_data=load_data(positive_files_path)
unsupervised_data=os.path.join(unsupervised_files_path)
