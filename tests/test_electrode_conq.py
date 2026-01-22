# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for polarizable electrode functionality in torch-admp.

This module contains tests to verify the correctness of calculations
with polarizable electrode under constant charge (CONQ) conditions
with comparisons against LAMMPS reference data.
"""


# class TestConqInterface2DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 2D interface at zero charge.

#     Tests constant charge electrode simulation for a 2D interface system
#     at zero charge condition with slab correction.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 2D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 2D interface
#         system at zero charge condition with slab correction.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_2d_pzc/dump.lammpstrj"
#         )
#         self.ref_energy = -1943.6576
#         self.slab_corr = True
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         self.input_data = setup_from_lammps(
#             len(self.atoms),
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(216),
#                     value=0.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#             ],
#         )


# class TestConqInterface2DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 2D interface with bias.

#     Tests constant charge electrode simulation for a 2D interface system
#     with applied bias potential and slab correction.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 2D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 2D interface
#         system with applied bias potential and slab correction.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_2d_bias/dump.lammpstrj"
#         )
#         self.ref_energy = -900.46651
#         self.slab_corr = True
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         self.input_data = setup_from_lammps(
#             len(self.atoms),
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(108),
#                     value=-10.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(108, 216),
#                     value=10.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#             ],
#         )


# class TestConqInterface2DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 2D interface with EDL.

#     Tests constant charge electrode simulation for a 2D interface system
#     with electrical double layer (EDL) formation.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 2D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 2D interface
#         system with electrical double layer (EDL) formation.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_2d_edl/dump.lammpstrj"
#         )
#         self.ref_energy = -1114.9378
#         self.slab_corr = True
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         self.input_data = setup_from_lammps(
#             len(self.atoms),
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(216),
#                     value=-10.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#             ],
#         )


# class TestConqInterface3DEDL(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 3D interface with EDL.

#     Tests constant charge electrode simulation for a 3D interface system
#     with electrical double layer (EDL) formation.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 3D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 3D interface
#         system with electrical double layer (EDL) formation.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_3d_edl/dump.lammpstrj"
#         )
#         self.ref_energy = -1114.9377
#         self.slab_corr = False
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         self.input_data = setup_from_lammps(
#             len(self.atoms),
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(216),
#                     value=-10.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#             ],
#         )


# class TestConqInterface3DBIAS(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 3D interface with bias.

#     Tests constant charge electrode simulation for a 3D interface system
#     with applied bias potential.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 3D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 3D interface
#         system with applied bias potential.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_3d_bias/dump.lammpstrj"
#         )
#         self.ref_energy = -1648.7002
#         self.slab_corr = False
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         with self.assertRaises(AttributeError) as context:
#             self.input_data = setup_from_lammps(
#                 len(self.atoms),
#                 [
#                     LAMMPSElectrodeConstraint(
#                         indices=np.arange(108),
#                         value=-10.0,
#                         mode="conq",
#                         eta=1.6,
#                         ffield=True,
#                     ),
#                     LAMMPSElectrodeConstraint(
#                         indices=np.arange(108, 216),
#                         value=10.0,
#                         mode="conq",
#                         eta=1.6,
#                         ffield=True,
#                     ),
#                 ],
#             )

#         self.assertIn(
#             "ffield with conq has not been implemented yet",
#             str(context.exception),
#         )

#     def test(self):
#         pass


# class TestConqInterface3DPZC(LAMMPSReferenceDataTest, unittest.TestCase):
#     """Test constant charge simulation for 3D interface at zero charge.

#     Tests constant charge electrode simulation for a 3D interface system
#     at zero charge condition.
#     """

#     def setUp(self) -> None:
#         """Set up test data for 3D interface constant charge simulation.

#         Loads atomic positions and sets up electrode constraints for a 3D interface
#         system at zero charge condition.
#         """
#         self.atoms = io.read(
#             Path(__file__).parent / "data/lmp_conq_interface_3d_pzc/dump.lammpstrj"
#         )
#         self.ref_energy = -1943.6583
#         self.slab_corr = False
#         # mask, eta, chi, hardness, constraint_matrix, constraint_vals, ffield_electrode_mask, ffield_potential
#         self.input_data = setup_from_lammps(
#             len(self.atoms),
#             [
#                 LAMMPSElectrodeConstraint(
#                     indices=np.arange(216),
#                     value=0.0,
#                     mode="conq",
#                     eta=1.6,
#                     ffield=False,
#                 ),
#             ],
#         )
