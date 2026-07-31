#!/usr/bin/env python3
import argparse

from common import get_device, load_lm, str2dtype
from qmamba.esr.trainable import describe_mixer_slices, get_layers, get_mixer, mamba2_inproj_slices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="state-spaces/mamba2-370m")
    p.add_argument("--loader", type=str, default="mamba_ssm")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    model = load_lm(args.model, args.loader, str2dtype(args.dtype), get_device(args.device))
    layers = get_layers(model)
    print(f"num layers: {len(layers)}")
    mixer = get_mixer(layers[0])
    print("mixer attrs:")
    for k in ["d_model", "d_ssm", "d_state", "nheads", "headdim", "ngroups", "rmsnorm", "activation"]:
        print(f"  {k}: {getattr(mixer, k, None)}")
    print("in_proj slices:")
    for k, v in describe_mixer_slices(mixer).items():
        print(f"  {k}: {v}")
    s = mamba2_inproj_slices(mixer)
    print("train rows B/C/dt:", s.train_bcdt_rows[:10].tolist(), "...", s.train_bcdt_rows[-10:].tolist())


if __name__ == "__main__":
    main()
