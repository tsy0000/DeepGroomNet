import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import train
sys.argv = ['train.py'] + sys.argv[1:]
if __name__ == '__main__':
    train.main()
