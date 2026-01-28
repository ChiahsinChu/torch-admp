# SPDX-License-Identifier: LGPL-3.0-or-later
"""
JIT compilation example for QEq calculations.

This example demonstrates how to use JIT (Just-In-Time) compilation
to optimize QEq calculations for better performance.
"""

import time

import jax.numpy as jnp
import numpy as np
import openmm.app as app
import openmm.unit as unit
import torch
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from scipy import constants

from torch_admp import env
from torch_admp.nblist import TorchNeighborList
from torch_admp.qeq import QEqForceModule, pgrad_optimize
from torch_admp.utils import calc_grads

# Set default device and precision
torch.set_default_device(env.DEVICE)
torch.set_default_dtype(env.GLOBAL_PT_FLOAT_PRECISION)


def load_test_data():
    """
    Load test data from PDB and XML files.

    Returns
    -------
    dict
        Dictionary containing molecular system data.
    """
    xml = XMLIO()
    xml.loadXML("qeq.xml")
    res = xml.parseResidues()
    ffinfo = xml.parseXML()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = np.array([a["type"] for a in res[0]["particles"]])

    pdb = app.PDBFile("qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = jnp.array(positions)
    a, b, c = dmfftop.getPeriodicBoxVectors()

    n_atoms = dmfftop.getNumAtoms()
    eta = np.zeros([n_atoms])
    chi = np.zeros([n_atoms])
    hardness = np.zeros([n_atoms])
    for _data in ffinfo["Forces"]["ADMPQeqForce"]["node"]:
        eta[types == _data["attrib"]["type"]] = float(_data["attrib"]["eta"])
        chi[types == _data["attrib"]["type"]] = float(_data["attrib"]["chi"])
        hardness[types == _data["attrib"]["type"]] = float(_data["attrib"]["J"])

    # Convert energy units from kJ/mol to eV/particle
    j2ev = constants.physical_constants["joule-electron volt relationship"][0]
    energy_coeff = j2ev * constants.kilo / constants.Avogadro

    data_dict = {
        "n_atoms": n_atoms,
        "position": np.array(positions),
        "box": np.array([a._value, b._value, c._value]) * 10.0,
        "chi": chi * energy_coeff,
        "hardness": hardness * energy_coeff,
        "eta": eta,
        "charge": charges,
    }
    return data_dict


def benchmark_jit_vs_regular():
    """
    Benchmark JIT-compiled vs regular QEq calculations.
    """
    print("Benchmarking JIT-compiled vs regular QEq calculations...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create regular QEq module
    regular_module = QEqForceModule(rcut=rcut, ethresh=ethresh)

    # Create JIT-compiled QEq module
    jit_module = torch.jit.script(QEqForceModule(rcut=rcut, ethresh=ethresh))

    # Benchmark regular module
    print("\nRegular QEq module:")
    start_time = time.time()
    energy_regular, q_opt_regular = regular_module.solve_pgrad(
        charges,
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        constraint_vals,
        method="lbfgs",
    )
    regular_time = time.time() - start_time

    print(f"  Energy: {energy_regular.item():.6f} eV")
    print(f"  Total charge: {q_opt_regular.sum().item():.6f} e")
    print(f"  Converged in: {regular_module.converge_iter} iterations")
    print(f"  Execution time: {regular_time:.4f} seconds")

    # Benchmark JIT module using pgrad_optimize function
    print("\nJIT-compiled QEq module:")
    from torch_admp.utils import vector_projection_coeff_matrix

    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    start_time = time.time()
    energy_jit, q_opt_jit = pgrad_optimize(
        jit_module,
        charges,
        positions,
        box,
        chi,
        hardness,
        eta,
        pairs,
        ds,
        buffer_scales,
        constraint_matrix,
        constraint_vals,
        coeff_matrix,
        method="lbfgs",
    )
    jit_time = time.time() - start_time

    print(f"  Energy: {energy_jit.item():.6f} eV")
    print(f"  Total charge: {q_opt_jit.sum().item():.6f} e")
    print(f"  Converged in: {jit_module.converge_iter} iterations")
    print(f"  Execution time: {jit_time:.4f} seconds")

    # Compare results
    energy_diff = abs(energy_regular.item() - energy_jit.item())
    charge_diff = torch.norm(q_opt_regular - q_opt_jit).item()
    speedup = regular_time / jit_time

    print(f"\nComparison:")
    print(f"  Energy difference: {energy_diff:.8f} eV")
    print(f"  Charge difference: {charge_diff:.8f} e")
    print(f"  Speedup: {speedup:.2f}x")

    if energy_diff < 1e-6 and charge_diff < 1e-6:
        print("  ✓ JIT and regular modules give consistent results!")
    else:
        print("  ⚠ Warning: JIT and regular modules show significant differences")


def test_jit_with_different_methods():
    """
    Test JIT compilation with different optimization methods.
    """
    print("\nTesting JIT compilation with different optimization methods...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create JIT-compiled QEq module
    jit_module = torch.jit.script(QEqForceModule(rcut=rcut, ethresh=ethresh))
    from torch_admp.utils import vector_projection_coeff_matrix

    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    # Test different optimization methods
    methods = ["lbfgs", "quadratic"]

    print("\nMethod | Energy (eV) | Total Charge (e) | Iterations | Time (s)")
    print("-" * 65)

    for method in methods:
        # Benchmark JIT module
        start_time = time.time()
        energy, q_opt = pgrad_optimize(
            jit_module,
            charges,
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
            coeff_matrix,
            method=method,
        )
        execution_time = time.time() - start_time

        print(
            f"{method:7s} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {jit_module.converge_iter:10d} | {execution_time:8.4f}"
        )


def test_jit_for_batch_processing():
    """
    Test JIT compilation for batch processing.
    """
    print("\nTesting JIT compilation for batch processing...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5
    n_configs = 5

    # Load molecular data
    data_dict = load_test_data()

    # Generate multiple configurations
    base_positions = data_dict["position"]
    rng = np.random.default_rng(seed=42)
    batch_positions = []
    for i in range(n_configs):
        displacement = rng.uniform(-0.1, 0.1, base_positions.shape)
        config_positions = base_positions + displacement
        batch_positions.append(config_positions)

    # Convert other data to PyTorch tensors
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create JIT-compiled QEq module
    jit_module = torch.jit.script(QEqForceModule(rcut=rcut, ethresh=ethresh))
    from torch_admp.utils import vector_projection_coeff_matrix

    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    # Process batch with JIT
    print(f"\nProcessing {n_configs} configurations with JIT...")
    print("Config | Energy (eV) | Total Charge (e) | Iterations | Time (s)")
    print("-" * 65)

    total_time = 0
    for i in range(n_configs):
        # Convert current configuration to tensor
        positions = torch.tensor(batch_positions[i], requires_grad=True)
        charges = torch.zeros(data_dict["n_atoms"], dtype=torch.float64)

        # Calculate neighbor list for current configuration
        nblist = TorchNeighborList(cutoff=rcut)
        pairs = nblist(positions, box)
        ds = nblist.get_ds()
        buffer_scales = nblist.get_buffer_scales()

        # Benchmark JIT module
        start_time = time.time()
        energy, q_opt = pgrad_optimize(
            jit_module,
            charges,
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
            coeff_matrix,
            method="lbfgs",
        )
        execution_time = time.time() - start_time
        total_time += execution_time

        print(
            f"{i:5d} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {jit_module.converge_iter:10d} | {execution_time:8.4f}"
        )

    print(f"\nTotal JIT processing time: {total_time:.4f} seconds")
    print(f"Average time per configuration: {total_time/n_configs:.4f} seconds")


def test_jit_warmup():
    """
    Test JIT warmup effects on performance.
    """
    print("\nTesting JIT warmup effects...")

    # Set up QEq parameters
    rcut = 8.0
    ethresh = 1e-5

    # Load molecular data
    data_dict = load_test_data()

    # Convert data to PyTorch tensors
    positions = torch.tensor(
        data_dict["position"],
        requires_grad=True,
    )
    box = torch.tensor(
        data_dict["box"],
        requires_grad=False,
    )
    chi = torch.tensor(
        data_dict["chi"],
        requires_grad=False,
    )
    hardness = torch.tensor(
        data_dict["hardness"],
        requires_grad=False,
    )
    eta = torch.tensor(
        data_dict["eta"],
        requires_grad=False,
    )
    charges = torch.tensor(
        data_dict["charge"],
        requires_grad=False,
    )

    # Calculate neighbor list
    nblist = TorchNeighborList(cutoff=rcut)
    pairs = nblist(positions, box)
    ds = nblist.get_ds()
    buffer_scales = nblist.get_buffer_scales()

    # Set up charge constraints
    constraint_matrix = torch.ones([1, data_dict["n_atoms"]], dtype=torch.float64)
    constraint_vals = torch.zeros(1, dtype=torch.float64)

    # Create JIT-compiled QEq module
    jit_module = torch.jit.script(QEqForceModule(rcut=rcut, ethresh=ethresh))
    from torch_admp.utils import vector_projection_coeff_matrix

    coeff_matrix = vector_projection_coeff_matrix(constraint_matrix)

    # Test JIT warmup
    n_runs = 5
    print(f"\nRunning {n_runs} JIT executions to test warmup...")
    print("Run | Energy (eV) | Total Charge (e) | Iterations | Time (s)")
    print("-" * 65)

    for run in range(n_runs):
        # Benchmark JIT module
        start_time = time.time()
        energy, q_opt = pgrad_optimize(
            jit_module,
            charges,
            positions,
            box,
            chi,
            hardness,
            eta,
            pairs,
            ds,
            buffer_scales,
            constraint_matrix,
            constraint_vals,
            coeff_matrix,
            method="lbfgs",
        )
        execution_time = time.time() - start_time

        print(
            f"{run:3d} | {energy.item():10.6f} | {q_opt.sum().item():14.6f} | {jit_module.converge_iter:10d} | {execution_time:8.4f}"
        )

    print("\nNote: First run may be slower due to JIT compilation overhead.")


def main():
    """
    Main function demonstrating JIT compilation for QEq.
    """
    print("Running QEq JIT compilation example...")

    # Test JIT compilation
    benchmark_jit_vs_regular()
    test_jit_with_different_methods()
    test_jit_for_batch_processing()
    test_jit_warmup()

    print("\n✓ All JIT compilation tests completed successfully!")


if __name__ == "__main__":
    main()
