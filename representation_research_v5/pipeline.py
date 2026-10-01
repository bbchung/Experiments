"""Run the bounded frozen comparison in one process, sharing artifact hashes."""

import argparse

from . import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    profile = runner.contract.load_profile(args.profile)
    path = runner.output(profile) / "frozen-contract.yaml"
    if not path.exists():
        print("PREFLIGHT START", flush=True)
        receipt = runner.preflight(profile)
        print({"preflight": receipt["passed"], "entries": len(receipt["entries"])}, flush=True)
        frozen = runner.freeze(profile)
        print({"frozen_identity": frozen["identity"]}, flush=True)
    else:
        print({"resuming_verified_contract": runner.contract.verify(path)["identity"]}, flush=True)
    runner.run(profile)
    print(runner.read_yaml(runner.output(profile) / "completed.yaml"), flush=True)


if __name__ == "__main__":
    main()
