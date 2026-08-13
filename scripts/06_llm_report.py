import argparse
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import yaml
from llm.report import generate_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--pred', required=True, help='infer.py 输出的帧级预测 CSV')
    ap.add_argument('--out', default=None, help='报告输出路径（默认 <pred>_report.md）')
    ap.add_argument('--language', default=None, choices=['zh', 'en'])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    out = args.out or os.path.join(ROOT, 'outputs', 'reports', os.path.basename(os.path.splitext(args.pred)[0]) + '_report.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    generate_report(cfg, args.pred, out_md=out, language=args.language)
    if os.path.exists(out):
        print(f'[06] report saved -> {out}')
if __name__ == '__main__':
    main()
