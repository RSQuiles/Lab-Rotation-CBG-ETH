from fcr import FCR_sim
from fcr import fetch_latest
import os
import argparse
import json
import socket
import subprocess
import warnings
warnings.filterwarnings("ignore")

def main_fcr(arguments=None):
    # USE CASE 1: Standard training
    if arguments is None:
        # Parse arguments from the sbatch script
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", type=str, help="Name of the experiment")
        parser.add_argument("--description", type=str, help="More detailed description of the experiment")
        parser.add_argument("--parameters", nargs=argparse.REMAINDER, help="List of key-value pairs that define user-input parameters")
        parser.add_argument("--parallel", action="store_true", help="Enable parallel training")
        print("Parsing arguments from command line...")
        args = parser.parse_args()

        experiment = args.experiment
        description = args.description
        parameters = args.parameters
        parallel = args.parallel

    # USE CASE 2: Hyperparameter sweep
    else:
        # Read arguments from sweep file
        experiment = arguments["experiment"]
        description = arguments["description"]
        parameters = ["sweep=True"]
        parallel = False

    """
    # Set up envitonment variables for distributed training if parallel is enabled
    if parallel:
        # --- MASTER_PORT ---
        sock = socket.socket()
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
        sock.close()
        os.environ["MASTER_PORT"] = str(master_port)

        # --- WORLD_SIZE ---
        print(f"SLURM_NNODES: {os.environ.get('SLURM_NNODES', None)}")
        slurm_nnodes = int(os.environ.get("SLURM_NNODES", 1))
        print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS', None)}")
        slurm_ntasks_per_node = int(os.environ.get("SLURM_NTASKS", 1))
        world_size = (slurm_nnodes * slurm_ntasks_per_node) - 1
        os.environ["WORLD_SIZE"] = str(world_size)

        # --- MASTER_ADDR ---
        # Use `scontrol show hostnames` to get the first node in the allocation
        try:
            master_addr = (
                subprocess.check_output(
                    ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
                )
                .decode()
                .splitlines()[0]
            )
        except Exception:
            # fallback: use localhost if not in SLURM context
            master_addr = "127.0.0.1"

        os.environ["MASTER_ADDR"] = master_addr

        print(f"MASTER_ADDR={master_addr}")
        print(f"MASTER_PORT={master_port}")
        print(f"WORLD_SIZE={world_size}")
    """

    # Parse key=value pairs into a dictionary
    params = {}
    # Some parameters can be inputted directly
    params["resume"] = False

    if parameters:
        for kv in parameters:
            if "=" not in kv:
                raise ValueError(f"Invalid parameter format: {kv}. Use key=value.")
            k, v = kv.split("=", 1)
            # try to cast numbers automatically
            try:
                v = eval(v, {"__builtins__":{}})
            except Exception:
                pass
            params[k] = v

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, experiment, "config.json")

    # Use the experiment and description to define the artifact_path
    params["artifact_path"] = str(os.path.join(script_dir, experiment, description))
    params["name"] = experiment
    params["experiment"] = description
    params["parallel"] = parallel

    # Load state dict if resuming training
    if params["resume"]:
        latest_checkpoint = fetch_latest(params["artifact_path"])
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = FCR_sim(model_path = latest_checkpoint)
        state_dict = model.state_dict
        epoch_str = latest_checkpoint.split("_epoch=")[1].split(".pt")[0]
        epoch = int(epoch_str)
        model.arguments["epoch"] = epoch

        with open(config_path, "r") as f:
            config_args = json.load(f)
        
        # Update model parameters if changed in the config file
        model.arguments["max_epochs"] = config_args.get("max_epochs", model.arguments["max_epochs"])
    
    else:
        model = FCR_sim(config_path=config_path, parameters=params)
        state_dict = None

    model.train_fcr(parallel=parallel, state_dict=state_dict)

if __name__ == "__main__":
    main_fcr()
