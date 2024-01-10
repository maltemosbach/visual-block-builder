#!/bin/bash

CONDA_ENV_NAME=$(head -1 ./conda_env.yml | cut -f2 -d' ')
MUJOCO_PATH="$HOME/.mujoco/mujoco210"
MUJOCO_LD_PATHS=("$HOME/.mujoco/mujoco210/bin" "/usr/lib/nvidia")

# Check if MuJoCo is installed.
if [ ! -d $MUJOCO_PATH ]; then
    echo "MuJoCo installation not found. Installing MuJoCo 2.1.0."
    wget "https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
    mkdir -p "$HOME/.mujoco"
    tar -xf mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"
    rm mujoco210-linux-x86_64.tar.gz
fi

# Check environment variables needed by MuJoCo.
source $HOME/.bashrc
ENVIRONMENT_VARIABLES_ADDED=false
for path in "${MUJOCO_LD_PATHS[@]}"; do
    if [[ -z "$LD_LIBRARY_PATH" ]] || [[ $LD_LIBRARY_PATH != *"$path"* ]]; then
        if [ "$ENVIRONMENT_VARIABLES_ADDED" = false ]; then
            echo "# >>> The following lines were added by visual-block-builder/create_conda_env.sh >>>" >> $HOME/.bashrc
            ENVIRONMENT_VARIABLES_ADDED=true
        fi
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$path" >> $HOME/.bashrc
    fi
done
if [[ -z "$MUJOCO_PY_MUJOCO_PATH" ]] || [[ $MUJOCO_PY_MUJOCO_PATH != "$HOME/.mujoco/mujoco210" ]]; then
    if [ "$ENVIRONMENT_VARIABLES_ADDED" = false ]; then
            echo "# >>> The following lines were added by visual-block-builder/create_conda_env.sh >>>" >> $HOME/.bashrc
            ENVIRONMENT_VARIABLES_ADDED=true
    fi
    echo "export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210" >> $HOME/.bashrc
fi

if [ "$ENVIRONMENT_VARIABLES_ADDED"=true ]; then
    echo "# <<< End of lines added by visual-block-builder/create_conda_env.sh <<<" >> $HOME/.bashrc
fi
source $HOME/.bashrc

# Source conda.
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# Deactivate environment to be created, if it is active.
ACTIVE_ENV_NAME="$(basename ${CONDA_PREFIX})"
if [ "${CONDA_ENV_NAME}" = "${ACTIVE_ENV_NAME}" ]; then
    conda deactivate
fi

# Remove existing version of this environment.
conda remove -y -n "${CONDA_ENV_NAME}" --all

# Remove existing version of fetch-block-construction package.
rm -rf ./src/fetch-block-construction

# Create environment from YAML.
conda env create -f ./conda_env.yml
if [ $? -ne 0 ]; then
    echo "Failed to create $CONDA_ENV_NAME conda environment."
    exit 1
fi

# Activate environment.
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "Failed to activate $CONDA_ENV_NAME conda environment."
    exit 1
fi

# Add additional assets.
cp ./visual_block_builder/assets/simplified_robot.xml ./src/fetch-block-construction/fetch_block_construction/envs/robotics/assets/fetch/
cp ./visual_block_builder/assets/robot.xml ./src/fetch-block-construction/fetch_block_construction/envs/robotics/assets/fetch/

# Install the Python package.
pip install -e .
