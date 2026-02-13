#!/bin/bash
# Runs multiple combinations of supercells and backend (n_steps can be changed)

cd "$(dirname "$0")"

N_STEPS=1000
SUPERCELLS=("1,1,1" "2,2,2" "3,3,3" "4,4,4" "5,5,5")
BACKENDS=(python numpy fortran cpp_openmp)

RESULTS_FILE="benchmark_results.csv"
echo "backend,supercell,time" > "$RESULTS_FILE"

for supercell in "${SUPERCELLS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        sc_name=$(echo "$supercell" | cut -d',' -f1)
        output_name="Ar_nvt_steps${N_STEPS}_sc${sc_name}_${backend}"
        
        echo "Running: n_steps=$N_STEPS, supercell=[$supercell], backend=$backend"
        
        cat > temp_nvt.yaml << EOF
input_file: Ar.xyz
output_file: ${output_name}

ensemble: nvt
temperature: 1.2
tau: 10.0

dt: 0.001
n_steps: ${N_STEPS}
log_every: 100
traj_every: 1000

r_cut: 2.5
r_skin: 0.3

sigma: 0.9
epsilon: 0.6

box: [3.419952, 3.419952, 3.419952]
supercell: [${supercell}]

backend: ${backend}
EOF
        
        python -m minimd temp_nvt.yaml
        
        time_taken=$(grep "# Total" "${output_name}.log" | awk '{print $5}')
        echo "${backend},${sc_name},${time_taken}" >> "$RESULTS_FILE"
        
        echo "Completed: $output_name (${time_taken}s)"
        echo "---"
    done
done

rm -f temp_nvt.yaml
rm -f Ar_nvt*

echo "All benchmarks completed!"
echo "Results saved to $RESULTS_FILE"

python plot_benchmarks.py
