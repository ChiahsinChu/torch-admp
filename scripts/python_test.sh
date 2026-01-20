set -e

# use pushd and popd to return to the original directory
pushd "$(dirname "$0")/.."
mkdir -p docs/reports
mkdir -p docs/badges
pytest --cov=torch_admp tests \
	--cov-report=term-missing \
	--cov-report html:docs/reports/coverage
docstr-coverage src/torch_admp/ \
	--skip-private \
	--skip-property \
	--accept-empty \
	--exclude=".*/_version.py"
popd
