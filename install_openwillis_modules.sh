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