import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet.pose.kpms import apply_kpms, fit_kpms


def main():
    parser = argparse.ArgumentParser(description="Fit or apply a Keypoint-MoSeq model")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--keypoint-csv", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--latent-dim", type=int, default=10)
    fit.add_argument("--num-states", type=int, default=100)
    fit.add_argument("--kappa", type=float, default=1e7)
    fit.add_argument("--ar-iters", type=int, default=50)
    fit.add_argument("--full-iters", type=int, default=500)
    fit.add_argument("--seed", type=int, default=0)
    apply_ = sub.add_parser("apply")
    apply_.add_argument("--project-dir", required=True)
    apply_.add_argument("--model-name", required=True)
    apply_.add_argument("--keypoint-csv", required=True)
    apply_.add_argument("--out-npy", required=True)
    apply_.add_argument("--apply-iters", type=int, default=200)
    args = parser.parse_args()

    if args.command == "fit":
        info = fit_kpms(args.keypoint_csv, args.out, latent_dim=args.latent_dim,
                        num_states=args.num_states, kappa=args.kappa,
                        ar_iters=args.ar_iters, full_iters=args.full_iters,
                        seed=args.seed)
    else:
        apply_kpms(args.project_dir, args.model_name, args.keypoint_csv,
                   args.out_npy, apply_iters=args.apply_iters)
        info = {"syllables": args.out_npy}
    print(info)


if __name__ == "__main__":
    main()
