import os
import sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'config.yaml'), encoding='utf-8'))
from data import openfield_meta, homecage_meta
if __name__ == '__main__':
    openfield_meta.build_openfield_meta(cfg['DATA']['openfield_roots'], os.path.join(cfg['DATA']['meta_root'], 'openfield_meta.csv'))
    homecage_meta.build_homecage_meta(cfg['DATA']['homecage_root'], os.path.join(cfg['DATA']['meta_root'], 'homecage_meta.csv'))
