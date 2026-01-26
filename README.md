# ADMP (Automatic Differentiable Multipolar Polarizable) in PyTorch backend

[![codecov](https://codecov.io/gh/ChiahsinChu/torch-admp/graph/badge.svg?token=aBlN5QoejV)](https://codecov.io/gh/ChiahsinChu/torch-admp)
[![docscov](./docs/badges/docstring-coverage.svg)](https://pypi.org/project/docstr-coverage/)
[![PyPI version](https://img.shields.io/pypi/v/torch-admp)](https://pypi.org/project/torch-admp/)

> torch version of ADMP is initialized by [Zheng Cheng](https://github.com/zhengcheng233/dmff_torch) (AISI).

This package implements the PME method (for monopoles) and the QEq method in [DMFF](https://github.com/deepmodeling/DMFF) with PyTorch, allowing not only GPU-accelerated calculation of PME/QEq methods but also further customization and extension of other PyTorch-based models.

## Installation

This package can be installed by:

```bash
pip install torch-admp
```

For the unit tests, you can install the package from source with the following command:

```bash
git clone https://github.com/ChiahsinChu/torch-admp
pip install torch-admp[test,vesin]
pip install "DMFF @ git+https://github.com/ChiahsinChu/DMFF.git@devel"
```

## [Documentation](https://chiahsinchu.github.io/torch-admp/)

## License

This project is licensed under the LGPL-3.0-or-later license. See the [LICENSE](LICENSE) file for details.

## Citation

If you use torch-admp in your research, please refer to the full citation details in the [CITATION.cff](CITATION.cff) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/ChiahsinChu/torch-admp).
