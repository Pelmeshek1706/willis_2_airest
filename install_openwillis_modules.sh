set -e

# List of subpackages to install in editable mode
PACKAGES=(
    "./openwillis/openwillis-speech"
    "./openwillis/openwillis-transcribe"
    "./openwillis/openwillis-voice"
)

for pkg in "${PACKAGES[@]}"; do
    echo "Installing ${pkg} in editable mode..."
    pip3 install -e "${pkg}"
done

echo "All subpackages installed successfully!"

# Installing setuptools-rust with trusted host options
echo "Installing setuptools-rust with trusted host options..."
pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org setuptools-rust

# Installing additional dependencies from requirements.txt if it exists
if [ -f "./requirements.txt" ]; then
    echo "Installing additional requirements from requirements.txt..."
    pip3 install -r "./requirements.txt"
fi
