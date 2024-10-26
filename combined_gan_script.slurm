#!/bin/bash
#SBATCH --job-name=python_test_job    # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --output=text_output_gan.txt  # Output file
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --time=24:00:00               # Time limit (hh:mm:ss)
#SBATCH --mem=64G                     # Memory allocation (64 GB)
#SBATCH --gres=gpu:1                  # Request 1 GPU (You have 4 A6000s, adjust as needed)
#SBATCH --cpus-per-task=16            # Number of CPU cores per task

# Activate the virtual environment
source /home/RSGIS-st121616/tony_env/bin/activate  # Use absolute path to virtual env

# Email notifications
#SBATCH --mail-user=akkarapon.chaiyana@gmail.com,67130749@dpu.ac.th  # Email addresses
#SBATCH --mail-type=END,FAIL         # Send email when job ends or fails

# Load Python module if necessary (only if you need system Python)
# module load python

# Run the Python script
srun python gan.py
srun python generate_inference.py
srun python mosaic_inference.py

# Deactivate the virtual environment
# deactivate
