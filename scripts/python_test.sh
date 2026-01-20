set -e

# use pushd and popd to return to the original directory
pushd "$(dirname "$0")/.."
pytest --cov=torch_admp tests --cov-report=term-missing
docstr-coverage src/torch_admp/ \
	--skip-private \
	--skip-property \
	--accept-empty \
	--exclude=".*/_version.py"
popd
