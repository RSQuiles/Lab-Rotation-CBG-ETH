# Read input arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <drug fingerprint name>"
    exit 1
fi

FP="$1"

# Determine fingerprint size based on name
if [ "$FP" == "morgan_1024" ]; then
    FP_SIZE=1024
elif [ "$FP" == "morgan_512" ]; then
    FP_SIZE=512
elif [ "$FP" == "MACCS" ]; then
    FP_SIZE=167
else
    echo "Unknown fingerprint name: $FP. Supported names are morgan_512, morgan_1024, MACCS."
    exit 1
fi

# Queue all experiments for the specified fingerprint
# PC embeddings
echo "Submitting PC embeddings for fingerprint: $FP"
for pcs in 10 25 50; do
    name="raw_data_pcs_${pcs}.npy"
    bash launch_runs.sh "$name" "$pcs" "$FP_SIZE" "$FP" "pcs_${pcs}"
done

# HVG embeddings
echo "Submitting HVG embeddings for fingerprint: $FP"
for n_hvgs in 1000 3000 5000; do
    name="raw_data_hvg_${n_hvgs}.npy"
    bash launch_runs.sh "$name" "$n_hvgs" "$FP_SIZE" "$FP" "hvgs_${n_hvgs}"
done

# piSCVI embeddings
echo "Submitting piscVI embeddings for fingerprint: $FP"
name="raw_data_piscvi.npy"
bash launch_runs.sh "$name" 100 "$FP_SIZE" "$FP" "piscvi"