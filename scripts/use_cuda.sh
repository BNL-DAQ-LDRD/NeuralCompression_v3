#!/bin/bash

# location of the cuda toolkit folder(s)
# if this is not the folder in your system, change it.
CUDA_ROOT="/usr/local"

# == Get the highest version ============================================
# Find all directories in /usr/local/ that start with "cuda-"
cuda_dirs=$(ls -d "$CUDA_ROOT"/cuda-*)

versions=()

for dir in $cuda_dirs; do
  # Extract the version number from the directory name
  version=$(basename "$dir" | sed 's/cuda-//')
  versions+=("$version")
done

# Sort the versions and find the highest one
highest_version=$(printf '%s\n' "${versions[@]}" | sort -V | tail -n1)

# Output the highest version
echo "The highest CUDA version is: $highest_version"
# =======================================================================


# == Check if a parameter is provided ===================================
# if not, used the highest available
if [ -z "$1" ];
then
    echo -e "A version number is not provided"
    echo -e "The highest version $highest_version will be used."
    CUDA_VERSION="$highest_version"
else
    CUDA_VERSION=$1
fi
# =======================================================================

# == Check if the directory exists ======================================
# if so, go ahead to change the nvcc version
CUDA_DIR=/usr/local/cuda-$CUDA_VERSION

if [ ! -d "$CUDA_DIR" ]; then
    echo "Error: CUDA directory $CUDA_DIR does not exist."
else
    # Set CUDA environment variables
    export CUDA_HOME=$CUDA_DIR
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    # Verify the setup
    echo "CUDA version set to $CUDA_VERSION"
fi
# =======================================================================
