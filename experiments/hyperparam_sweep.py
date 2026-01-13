import wandb
from fcr_run import main_fcr
import argparse

def run_sweep():
    # Parse (unknown) arguments, which define the name of the run
    parser = argparse.ArgumentParser()
    known, unknown = parser.parse_known_args()
    # Join key-value pairs into a clean string, build intermediate dictionary
    config = {}
    for i in range(0, len(unknown)):
        key = unknown[i].replace("-","").split("=")[0]
        val = unknown[i].replace("-","").split("=")[1]
        config[key] = val
    
    # Determine the experiment directory
    args = {}
    args["experiment"] = config["directory"]

    # Determine the subdirectory
    del config["directory"]
    param_string = "_".join(f"{k}_{v}" for k, v in config.items())
    print(f"Parameter string: {param_string}")

    args["description"] = param_string

    main_fcr(args)

if __name__ == "__main__":
    run_sweep()