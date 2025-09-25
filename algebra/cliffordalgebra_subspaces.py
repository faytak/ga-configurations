import functools
import numpy as np
import torch

from .cliffordalgebraex import CliffordAlgebraQT


class CliffordAlgebraSubspaces(CliffordAlgebraQT):
    """
    Extends CliffordAlgebraQT to enable working with different subspaces in geometric algebra (GA):

    1. Parity subspaces (even/odd subspaces) -For Group P equivariance
    2. Quaternion Type (QT) direct sum subspaces:
       - QT01 and QT23 subspaces (direct sums of QT 0, 1 and 2, 3 respectively) - For Group A equivariance  
       - QT03 and QT12 subspaces (direct sums of QT 1, 2 and 0, 3 respectively) - For Group B equivariance
    3. Triple subspaces (grades modulo 3) - For step size 3 weight sharing
    
    The class provides:
    - Dimension calculations for each subspace partition
    - Permutation mappings for weight reordering
    - Geometric product path definitions for each partition
    - Operations of projection of elements onto the subspaces
    - Norm calculations with respect to each subspace
    - Auxiliary index mappings
    
    This enables the creation of geometric product layers that are equivariant with respect
    to different groups (P, A, B) operating with these subspaces.
    """

    def __init__(self, metric):
        super().__init__(metric)

        # Dimensions of the subspaces of even and odd subspaces
        self.dim_even, self.dim_odd = int(2**(self.dim - 1)), int(2**(self.dim - 1))

        # Dimensions of the direct sums of quaternion types (QT) subspaces
        self.dim_01 = self.dim_0 + self.dim_1
        self.dim_23 = self.dim_2 + self.dim_3
        self.dim_03 = self.dim_0 + self.dim_3
        self.dim_12 = self.dim_1 + self.dim_2

        #  Dimensions of the direct sums of subspaces of fixed grades with step size 3
        self.dim_triple0, self.dim_triple1, self.dim_triple2 = 0, 0, 0
        for grade in range(self.n_subspaces):
            if grade % 3 == 0:
                self.dim_triple0 += self.subspaces[grade].item()
            if grade % 3 == 1:
                self.dim_triple1 += self.subspaces[grade].item()
            if grade % 3 == 2:
                self.dim_triple2 += self.subspaces[grade].item()


    # ============================================================
    # helper: weight permutations
    # ============================================================
    def _make_weights_permutation(self, assign_partition, partition_offsets):
        """
        Construct a permutation array for reordering weights into partitions.

        assign_partition: function mapping grade -> partition_id (int)
        partition_offsets: list of cumulative starting offsets for each partition
        """
        counters = [0] * len(partition_offsets)
        arange = np.arange(2 ** self.dim)
        permutation = []

        for grade in range(self.n_subspaces):
            part_id = assign_partition(grade)
            offset = partition_offsets[part_id]
            for _ in range(self.subspaces[grade]):
                permutation.append(offset + arange[counters[part_id]])
                counters[part_id] += 1

        return permutation


    # ============================================================
    # actual permutations
    # ============================================================
    @functools.cached_property
    def parity_weights_permutation(self):
        return self._make_weights_permutation(
            assign_partition=lambda grade: grade % 2,
            partition_offsets=[0, self.dim_even]
        )

    @functools.cached_property
    def qt01_23_weights_permutation(self):
        return self._make_weights_permutation(
            assign_partition=lambda grade: 0 if grade % 4 in (0, 1) else 1,
            partition_offsets=[0, self.dim_01]
        )

    @functools.cached_property
    def qt03_12_weights_permutation(self):
        return self._make_weights_permutation(
            assign_partition=lambda grade: 0 if grade % 4 in (0, 3) else 1,
            partition_offsets=[0, self.dim_03]
        )


    # ============================================================
    # geometric product paths (partition-based)
    # ============================================================

    @functools.cached_property
    def parity_geometric_product_paths(self):
        # Sum up the results of multiplications (mod 2): [2, n+1, n+1]
        parity_results = torch.zeros((2, self.dim + 1, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            parity_results[grade % 2, :, :] += self.geometric_product_paths[grade, :, :]

        # Sum up the rows in multiplication table (mod 2): [2, 2, n+1]
        parity_sum_rows = torch.zeros((2, 2, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            parity_sum_rows[:, grade % 2, :] += parity_results[:, grade, :]

        # Sum up the columns in multiplication table (mod 2): [2, 2, 2]
        parity_sum_cols = torch.zeros((2, 2, 2), dtype=bool)
        for grade in range(self.dim + 1):
            parity_sum_cols[:, :, grade % 2] += parity_sum_rows[:, :, grade]

        return parity_sum_cols

    @functools.cached_property
    def qt01_23_geometric_product_paths(self): # same output for qt03_12
        self.qt_geometric_product_paths
        # Sum up the results of multiplications: [2, 4, 4]
        qt01_23_results = torch.zeros((2, 4, 4), dtype=bool)
        qt01_23_results[0, :, :] += (self.qt_geometric_product_paths[0, :, :] + self.qt_geometric_product_paths[1, :, :])
        qt01_23_results[1, :, :] += (self.qt_geometric_product_paths[2, :, :] + self.qt_geometric_product_paths[3, :, :])

        # Sum up the rows in multiplication table: [2, 2, 4]
        qt01_23_sum_rows = torch.zeros((2, 2, 4), dtype=bool)
        qt01_23_sum_rows[:, 0, :] += (qt01_23_results[:, 0, :] + qt01_23_results[:, 1, :])
        qt01_23_sum_rows[:, 1, :] += (qt01_23_results[:, 2, :] + qt01_23_results[:, 3, :])

        # Sum up the columns in multiplication table: [2, 2, 2]
        qt01_23_sum_cols = torch.zeros((2, 2, 2), dtype=bool)
        qt01_23_sum_cols[:, :, 0] += (qt01_23_sum_rows[:, :, 0] + qt01_23_sum_rows[:, :, 1])
        qt01_23_sum_cols[:, :, 1] += (qt01_23_sum_rows[:, :, 2] + qt01_23_sum_rows[:, :, 3])

        return qt01_23_sum_cols


    # ============================================================
    # slice partitions
    # ============================================================

    @functools.cached_property
    def parity_to_list(self):
        """
        Get list of 2 lists with ids of basis elements for even and odd subspaces in slices
        """
        return [self.grade_to_slice[::2], self.grade_to_slice[1::2]]

    @functools.cached_property
    def qt01_23_to_list(self):
        """
        Get list of 2 lists with ids of basis elements for QT01 and QT23 subspaces in slices
        """
        return [self.grade_to_slice[::4] + self.grade_to_slice[1::4], self.grade_to_slice[2::4] + self.grade_to_slice[3::4]]

    @functools.cached_property
    def qt03_12_to_list(self):
        """
        Get list of 2 lists with ids of basis elements for QT03 and QT12 subspaces in slices
        """
        return [self.grade_to_slice[::4] + self.grade_to_slice[3::4], self.grade_to_slice[1::4] + self.grade_to_slice[2::4]]


    # ============================================================
    # helper: convertation of list-of-slices to list-of-index-tensors
    # ============================================================
    def _slices_to_index_tensors(self, slice_groups):
        """
        Convert list of slice groups into list of torch index tensors.
        """
        result = []
        for slices in slice_groups:
            indices = []
            for s in slices:
                indices.extend(range(s.start.item(), s.stop.item()))
            result.append(torch.tensor(indices))
        return result

    # ============================================================
    # index lists
    # ============================================================
    @functools.cached_property
    def qt01_23_to_index(self):
        """ List of tensors with ids of basis elements for QT01 or QT23 """
        return self._slices_to_index_tensors(self.qt01_23_to_list)

    @functools.cached_property
    def qt03_12_to_index(self):
        """ List of tensors with ids of basis elements for QT03 or QT12 """
        return self._slices_to_index_tensors(self.qt03_12_to_list)

    @functools.cached_property
    def parity_to_index(self):
        """ List of tensors with ids of basis elements for even/odd """
        return self._slices_to_index_tensors(self.parity_to_list)



    # ============================================================
    # generic helper: projection
    # ============================================================
    def _project_from_slices(self, mv: torch.Tensor, slice_groups, idx: int) -> torch.Tensor:
        """
        Generic projection of a multivector onto a chosen subspace partition.

        mv: input multivector tensor
        slice_groups: e.g. self.parity_to_list, self.qt01_23_to_list, ...
        idx: which partition to pick
        """
        slices = slice_groups[idx]
        return torch.cat([mv[..., slice_] for slice_ in slices], dim=2)

    # ============================================================
    # projections
    # ============================================================
    def get_parity(self, mv: torch.Tensor, parity: int) -> torch.Tensor:
        """
        Project a multivector onto a subspace of fixed parity (even=0 or odd=1).
        """
        return self._project_from_slices(mv, self.parity_to_list, parity)

    def get_qt03_12(self, mv: torch.Tensor, qt_sum: int) -> torch.Tensor:
        """
        Project a multivector onto QT03 (0) or QT12 (1).
        """
        return self._project_from_slices(mv, self.qt03_12_to_list, qt_sum)

    def get_qt01_23(self, mv: torch.Tensor, qt_sum: int) -> torch.Tensor:
        """
        Project a multivector onto QT01 (0) or QT23 (1).
        """
        return self._project_from_slices(mv, self.qt01_23_to_list, qt_sum)

    # ============================================================
    # norm functions with reversion, clifford conjugation, and grade involution
    # ============================================================

    def norm_reversion(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q(mv, blades=blades))

    def norm_clifford_conjugation(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q_clifford_conjugation(mv, blades=blades))

    def norm_grade_involution(self, mv, blades=None):
        return self._smooth_abs_sqrt(self.q_grade_involution(mv, blades=blades))

    def q_clifford_conjugation(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
        return self.b_clifford_conjugation(mv, mv, blades=blades)

    def b_clifford_conjugation(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            gamma_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            gamma_blades = None

        return self.geometric_product(
            self.gamma(x, blades=gamma_blades),
            y,
            blades=blades,
        )

    def q_grade_involution(self, mv, blades=None):
        if blades is not None:
            blades = (blades, blades)
        return self.b_grade_involution(mv, mv, blades=blades)

    def b_grade_involution(self, x, y, blades=None):
        if blades is not None:
            assert len(blades) == 2
            alpha_blades = blades[0]
            blades = (
                blades[0],
                torch.tensor([0]),
                blades[1],
            )
        else:
            blades = torch.tensor(range(self.n_blades))
            blades = (
                blades,
                torch.tensor([0]),
                blades,
            )
            alpha_blades = None

        return self.geometric_product(
            self.alpha(x, blades=alpha_blades),
            y,
            blades=blades,
        )


    # ============================================================
    # generic helper: norms
    # ============================================================
    def _norms_from_partition(self, mv, projector, index_list, norm_function):
        """
        Compute norms for a given partitioning scheme using given norm function.

        projector: function (mv, idx) -> projected tensor
        index_list: list of index tensors for each partition
        norm_function: function(mv, blades) -> norm of an element
        """
        return [
            norm_function(projector(mv, idx), blades=index_list[idx])
            for idx in range(2)
        ]

    # ============================================================
    # norms
    # ============================================================
    def norms_parity(self, mv):
        return self._norms_from_partition(
            mv, self.get_parity, self.parity_to_index, self.norm_grade_involution
        )

    def norms_qt01_23(self, mv):
        return self._norms_from_partition(
            mv, self.get_qt01_23, self.qt01_23_to_index, self.norm_reversion
        )

    def norms_qt03_12(self, mv):
        return self._norms_from_partition(
            mv, self.get_qt03_12, self.qt03_12_to_index, self.norm_clifford_conjugation
        )
    

    # ============================================================
    # generic helper: squared norms
    # ============================================================

    def _qs_generic(self, mv, projector, index_getter, num_partitions):
        """
        Generic function for computing squared norms for subspaces partitions elements (except grade-0 part).
        """
        temporary_mv = mv.clone()
        temporary_mv[..., 0] = 0  # Zero out the scalar component
        return [self.q(projector(temporary_mv, idx), blades=index_getter[idx])
                for idx in range(num_partitions)]

    # ============================================================
    # squared norms
    # ============================================================

    def qt_qs(self, mv):
        """Compute squared norms for quaternion type (QT) subspaces."""
        return self._qs_generic(mv, self.get_qt, self.qt_to_index, 4)

    def qt01_23_qs(self, mv):
        """Compute squared norms for QT01 and QT23 direct sum subspaces."""
        return self._qs_generic(mv, self.get_qt01_23, self.qt01_23_to_index, 2)

    def qt03_12_qs(self, mv):
        """Compute squared norms for QT03 and QT12 direct sum subspaces."""
        return self._qs_generic(mv, self.get_qt03_12, self.qt03_12_to_index, 2)

    def parity_qs(self, mv):
        """Compute squared norms for parity (even/odd) subspaces."""
        return self._qs_generic(mv, self.get_parity, self.parity_to_index, 2)


    # ============================================================
    # functional for weight sharing with step size 3
    # ============================================================

    @functools.cached_property
    def triples_weights_permutation(self):
        triple0, triple1, triple2 = 0, 0, 0
        arange = np.arange(2 ** self.dim)
        permutation = []

        for grade in range(self.n_subspaces):
            for elem in range(self.subspaces[grade]):
                if grade % 3 == 0:
                    permutation.append(arange[triple0])
                    triple0 += 1
                elif grade % 3 == 1:
                    permutation.append(self.dim_triple0 + arange[triple1])
                    triple1 += 1
                else:
                    permutation.append(self.dim_triple0 + self.dim_triple1 + arange[triple2])
                    triple2 += 1

        return permutation


    @functools.cached_property
    def triples_geometric_product_paths(self):
        # Sum up the results of multiplications (mod 3): [3, n+1, n+1]
        parity_results = torch.zeros((3, self.dim + 1, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            parity_results[grade % 3, :, :] += self.geometric_product_paths[grade, :, :]

        # Sum up the rows in multiplication table (mod 3): [3, 3, n+1]
        parity_sum_rows = torch.zeros((3, 3, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            parity_sum_rows[:, grade % 3, :] += parity_results[:, grade, :]

        # Sum up the columns in multiplication table (mod 3): [3, 3, 3]
        parity_sum_cols = torch.zeros((3, 3, 3), dtype=bool)
        for grade in range(self.dim + 1):
            parity_sum_cols[:, :, grade % 3] += parity_sum_rows[:, :, grade]

        return parity_sum_cols


    @functools.cached_property
    def triple_to_list(self):
        """
        Get list of 2 lists with ids of basis elements for even and odd subspaces in slices
        """
        return [self.grade_to_slice[::3], self.grade_to_slice[1::3], self.grade_to_slice[2::3]]


    def get_triple(self, mv: torch.Tensor, triple: int) -> torch.Tensor:
        """
        Project a multivector onto a subspaces of fixed parity (even or odd)
        """
        triple_list = self.triple_to_list[triple]
        indices = [(s.start, s.stop) for s in triple_list]
        new_slices = [slice(start.item(), stop.item()) for start, stop in indices]
        projection = []
        for slice_ in new_slices:
            projection.append(mv[..., slice_])
        return torch.cat(projection, dim=2)


    @functools.cached_property
    def triple_to_index(self):
        """
        Get list of tensors with ids of basis elements for even and odd subspaces
        """
        result = []
        for slices in self.triple_to_list:
            indices = []
            for s in slices:
                start = s.start.item()
                stop = s.stop.item()
                indices.extend(range(start, stop))
            result.append(torch.tensor(indices))
        return result


    def norms_triple(self, mv):
        return [
                self.norm(self.get_triple(mv, triple), blades=self.triple_to_index[triple])
                for triple in range(3)
            ]
    

    def triple_qs(self, mv):
        """Compute squared norms for triple subspaces (grades modulo 3)."""
        return self._qs_generic(mv, self.get_triple, self.triple_to_index, 3)

    


