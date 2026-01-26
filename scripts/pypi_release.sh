set -e

# use pushd and popd to return to the original directory
pushd "$(dirname "$0")/.."
# clear dist/ if exist
if [ -d "dist" ]; then
    rm -rf dist/*
    echo "Cleared dist/ directory"
fi
python -m build
twine check dist/*
# ask for confirmation
read -p "Are you sure you want to upload the package to PyPI? (y/n) " -n 1 -r
echo # move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo "Upload cancelled."
	exit 1
fi
twine upload -r testpypi dist/* --verbose
twine upload dist/* --verbose
popd
