from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rocket.coordinates import (
    align_tensors,
    calculate_mse_loss_per_residue,
    construct_SO3,
    decompose_SO3,
    extract_allatoms,
    extract_atoms_and_backbone,
    extract_bfactors,
    find_rigidbody_matrix_lbfgs_quat,
    fractionalize_torch,
    kabsch_align_matrices,
    pose_train_lbfgs_quat,
    quaternions_to_SO3,
    rigidbody_refine_quat,
    select_CA_elements,
    select_CA_from_craname,
    weighted_kabsch,
    write_pdb_with_positions,
)


@pytest.fixture
def mock_setup():
    """Create mock data for rigid body refinement tests."""
    # Create mock coordinates
    xyz = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )

    # Create mock CRA names
    cra_name = ["A-0-GLY-CA", "A-1-ALA-CA", "A-2-LEU-CA", "A-3-VAL-CA", "A-4-PHE-CA"]

    # Create mock LLGloss
    llgloss = MagicMock()
    llgloss.device = "cpu"

    return xyz, cra_name, llgloss


@patch("rocket.coordinates.find_rigidbody_matrix_lbfgs_quat")
def test_rigidbody_refine_quat_single_domain(mock_find_rigidbody, mock_setup):
    """Test rigidbody_refine_quat with a single domain."""
    xyz, cra_name, llgloss = mock_setup

    # Mock the quaternions and translation vectors returned by find_rigidbody_matrix_lbfgs_quat  # noqa: E501
    trans_vecs = [torch.tensor([0.1, 0.2, 0.3])]
    qs = [torch.tensor([0.9, 0.1, 0.1, 0.1])]
    mock_find_rigidbody.return_value = (trans_vecs, qs, [10.0, 9.0, 8.0])  # Loss track

    # Call the function
    result_xyz, loss_track = rigidbody_refine_quat(
        xyz=xyz, llgloss=llgloss, cra_name=cra_name, lbfgs=True, verbose=False
    )

    # Verify find_rigidbody_matrix_lbfgs_quat was called correctly
    mock_find_rigidbody.assert_called_once()
    args, kwargs = mock_find_rigidbody.call_args
    assert args[0] == llgloss
    assert torch.equal(args[1].detach(), xyz.detach())
    assert args[2] == llgloss.device
    # The domain_bools would be one array of all True
    assert len(args[3]) == 1
    assert np.all(args[3][0])

    # Check that result has correct shape
    assert result_xyz.shape == xyz.shape
    assert loss_track == [10.0, 9.0, 8.0]

    # Check that the transformation is applied correctly
    propose_rmcom = xyz - torch.mean(xyz, dim=0)
    propose_com = torch.mean(xyz, dim=0)
    expected_rotation = quaternions_to_SO3(qs[0]).detach()
    optimized_xyz = (
        torch.matmul(propose_rmcom, expected_rotation)
        + propose_com
        + trans_vecs[0].detach()
    )
    assert torch.allclose(result_xyz, optimized_xyz)


@patch("rocket.coordinates.find_rigidbody_matrix_lbfgs_quat")
def test_rigidbody_refine_quat_multiple_domains(mock_find_rigidbody, mock_setup):
    """Test rigidbody_refine_quat with multiple domains."""
    xyz, cra_name, llgloss = mock_setup

    # Domain segmentation at residue 2 (so residues 1-2 and 3-5 are two domains)
    domain_segs = [3]  # Split between residues 2 and 3

    # Mock the quaternions and translation vectors returned by find_rigidbody_matrix_lbfgs_quat  # noqa: E501
    trans_vecs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])]
    qs = [torch.tensor([0.9, 0.1, 0.1, 0.1]), torch.tensor([0.8, 0.2, 0.2, 0.2])]
    mock_find_rigidbody.return_value = (trans_vecs, qs, [10.0, 9.0])  # Loss track

    # Call the function with domain segmentation
    result_xyz, loss_track = rigidbody_refine_quat(
        xyz=xyz,
        llgloss=llgloss,
        cra_name=cra_name,
        lbfgs=True,
        verbose=False,
        domain_segs=domain_segs,
    )

    # Verify find_rigidbody_matrix_lbfgs_quat was called correctly
    mock_find_rigidbody.assert_called_once()
    args, kwargs = mock_find_rigidbody.call_args

    # There should be 2 domain boolean arrays
    assert len(args[3]) == 2

    # First domain: residues 1-2
    expected_domain1 = np.array([True, True, False, False, False])
    assert np.array_equal(args[3][0], expected_domain1)

    # Second domain: residues 3-5
    expected_domain2 = np.array([False, False, True, True, True])
    assert np.array_equal(args[3][1], expected_domain2)

    # Check that result has correct shape
    assert result_xyz.shape == xyz.shape
    assert loss_track == [10.0, 9.0]

    # Each domain should have coordinates transformed separately
    # So the transformation for first three atoms is different from last two
    domain_bools = args[3]
    optimized_xyz = torch.ones_like(xyz)
    for i in range(2):
        propose_rmcom = xyz[domain_bools[i]] - torch.mean(xyz[domain_bools[i]], dim=0)
        propose_com = torch.mean(xyz[domain_bools[i]], dim=0)
        transform_i = quaternions_to_SO3(qs[i]).detach()
        optimized_xyz[domain_bools[i]] = (
            torch.matmul(propose_rmcom, transform_i)
            + propose_com
            + trans_vecs[i].detach()
        )
    assert torch.allclose(result_xyz, optimized_xyz)


@patch("rocket.coordinates.find_rigidbody_matrix_lbfgs_quat")
def test_rigidbody_refine_quat_with_added_chains(mock_find_rigidbody, mock_setup):
    """Test rigidbody_refine_quat with additional chain parameters."""
    xyz, cra_name, llgloss = mock_setup

    # Mock additional chain parameters
    added_chain_HKL = torch.tensor([1.0, 2.0, 3.0])
    added_chain_asu = torch.tensor([4.0, 5.0, 6.0])

    # Mock return values
    trans_vecs = [torch.tensor([0.1, 0.2, 0.3])]
    qs = [torch.tensor([0.9, 0.1, 0.1, 0.1])]
    mock_find_rigidbody.return_value = (trans_vecs, qs, [8.0, 7.0])

    # Call the function with additional chain parameters
    result_xyz, loss_track = rigidbody_refine_quat(
        xyz=xyz,
        llgloss=llgloss,
        cra_name=cra_name,
        lbfgs=True,
        added_chain_HKL=added_chain_HKL,
        added_chain_asu=added_chain_asu,
        lbfgs_lr=100.0,
        verbose=False,
    )

    # Verify find_rigidbody_matrix_lbfgs_quat was called with correct parameters
    mock_find_rigidbody.assert_called_once()
    args, kwargs = mock_find_rigidbody.call_args
    assert torch.equal(kwargs["added_chain_HKL"], added_chain_HKL)
    assert torch.equal(kwargs["added_chain_asu"], added_chain_asu)
    assert kwargs["lbfgs_lr"] == 100.0
    assert kwargs["verbose"] is False

    # Check results
    assert result_xyz.shape == xyz.shape
    assert loss_track == [8.0, 7.0]


def test_rigidbody_refine_quat_integration():
    """Integration test for rigidbody_refine_quat using actual computation."""
    # Create simple test data
    xyz = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    cra_name = ["A-0-GLY-CA", "A-1-ALA-CA", "A-2-LEU-CA"]

    # Create a mock llgloss that returns a simple loss value
    class MockLLGLoss:
        def __init__(self):
            self.device = torch.device("cpu")
            self.call_count = 0

        def __call__(self, positions, **kwargs):
            # Simple loss function: sum of squared distances from origin
            self.call_count += 1
            return -torch.sum(torch.sum(positions**2, dim=1))

    llgloss = MockLLGLoss()

    # We'll use a patch here to avoid testing the full LBFGS optimization
    with patch("rocket.coordinates.find_rigidbody_matrix_lbfgs_quat") as mock_find:
        # Mock quaternion and translation that makes a simple transformation
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # Identity rotation
        trans = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # Simple translation
        mock_find.return_value = ([trans], [q], [-100.0])

        # Call the function
        optimized_xyz, loss_track = rigidbody_refine_quat(
            xyz=xyz, llgloss=llgloss, cra_name=cra_name, lbfgs=True, verbose=False
        )

        # For an identity rotation and translation of [1,1,1],
        # each point should be shifted by [1,1,1]
        expected_xyz = xyz + torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(optimized_xyz, expected_xyz)
        assert loss_track == [-100.0]


@pytest.fixture
def mock_optimizer():
    """Creates a mock LBFGS optimizer."""
    mock_opt = MagicMock()
    mock_opt.step.return_value = torch.tensor(-10.0)
    return mock_opt


@patch("torch.optim.LBFGS")
def test_find_rigidbody_matrix_lbfgs_quat(mock_lbfgs_class, mock_setup, mock_optimizer):
    """Test find_rigidbody_matrix_lbfgs_quat function."""
    xyz, cra_name, llgloss = mock_setup

    # Make just 2 points for simplicity
    xyz = xyz[:2]
    device = "cpu"

    # Create domain booleans
    domain_bools = [np.array([True, True])]

    # Configure the mock optimizer
    mock_lbfgs_class.return_value = mock_optimizer

    # Call the function
    trans_vecs, qs, loss_track = find_rigidbody_matrix_lbfgs_quat(
        llgloss=llgloss,
        xyz=xyz,
        device=device,
        domain_bools=domain_bools,
        lbfgs_lr=200.0,
        verbose=False,
    )

    # Check that the LBFGS optimizer was created with correct parameters
    mock_lbfgs_class.assert_called_once()
    args, kwargs = mock_lbfgs_class.call_args
    assert len(args[0]) == 2  # qs + trans_vecs
    assert kwargs["lr"] == 200.0
    assert kwargs["line_search_fn"] == "strong_wolfe"

    # Check that the optimizer step was called 15 times (default n_steps)
    assert mock_optimizer.step.call_count == 15

    # Check return values
    assert len(trans_vecs) == 1
    assert len(qs) == 1
    assert len(loss_track) == 15
    assert all(x == -10.0 for x in loss_track)

    # Check that q is initialized as identity quaternion
    assert torch.allclose(qs[0], torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))

    # Check that translation vector is initialized as zeros
    assert torch.allclose(trans_vecs[0], torch.tensor([0.0, 0.0, 0.0], device=device))


def test_pose_train_lbfgs_quat():
    """Test pose_train_lbfgs_quat function with mock optimizer."""

    # Create mock data
    llgloss = MagicMock()
    llgloss.return_value = torch.tensor(-5.0)

    # Create quaternions and translation vectors
    qs = [torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)]
    trans_vecs = [torch.tensor([0.0, 0.0, 0.0], requires_grad=True)]

    # Mock coordinates
    xyz = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )

    # Mock domain booleans
    domain_bools = [np.array([True, True, True])]

    # Use patch to mock the optimizer
    with patch("torch.optim.LBFGS") as mock_lbfgs_class:
        # Create a mock optimizer that returns a specified loss
        mock_optimizer = MagicMock()
        mock_optimizer.step.return_value = torch.tensor(-5.0)
        mock_lbfgs_class.return_value = mock_optimizer

        # Call the function with few steps to speed up test
        loss_track = pose_train_lbfgs_quat(
            llgloss=llgloss,
            qs=qs,
            trans_vecs=trans_vecs,
            xyz=xyz,
            domain_bools=domain_bools,
            lr=100.0,
            n_steps=3,
            verbose=False,
        )

        # Check that optimizer was created with correct parameters
        mock_lbfgs_class.assert_called_once()
        args, kwargs = mock_lbfgs_class.call_args
        assert args[0] == qs + trans_vecs
        assert kwargs["lr"] == 100.0

        # Check that optimizer.step was called the expected number of times
        assert mock_optimizer.step.call_count == 3

        # Check the loss track
        assert len(loss_track) == 3
        assert all(x == -5.0 for x in loss_track)


def test_pose_train_lbfgs_quat_with_added_chains():
    """Test pose_train_lbfgs_quat function properly passes added chains to llgloss."""
    import torch

    from rocket.coordinates import pose_train_lbfgs_quat

    # Track calls to llgloss
    call_args_list = []

    # Create a mock llgloss that records its arguments and returns a tensor that requires gradients  # noqa: E501
    def mock_llgloss_function(
        xyz, added_chain_HKL=None, added_chain_asu=None, **kwargs
    ):
        call_args_list.append({
            "xyz": xyz,
            "added_chain_HKL": added_chain_HKL,
            "added_chain_asu": added_chain_asu,
            "kwargs": kwargs,
        })
        # Create a tensor that requires gradients
        result = torch.tensor(-5.0, requires_grad=True)
        return result

    llgloss = MagicMock(side_effect=mock_llgloss_function)

    # Create quaternions and translation vectors
    qs = [torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)]
    trans_vecs = [torch.tensor([0.0, 0.0, 0.0], requires_grad=True)]

    # Mock coordinates
    xyz = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )

    # Mock domain booleans
    domain_bools = [np.array([True, True, True])]

    # Create added chain tensors
    added_chain_HKL = torch.tensor([1.0, 2.0, 3.0])
    added_chain_asu = torch.tensor([4.0, 5.0, 6.0])

    # Use patch to mock the optimizer with a simpler approach
    with patch("torch.optim.LBFGS") as mock_lbfgs_class:
        # Create a mock optimizer that just returns a fixed loss value when step is called  # noqa: E501
        mock_optimizer = MagicMock()

        # Make step actually call the closure function once and return its value
        def mock_step(closure):
            return closure()

        mock_optimizer.step.side_effect = mock_step
        mock_lbfgs_class.return_value = mock_optimizer

        # Call the function with just 1 step
        loss_track = pose_train_lbfgs_quat(
            llgloss=llgloss,
            qs=qs,
            trans_vecs=trans_vecs,
            xyz=xyz,
            domain_bools=domain_bools,
            lr=100.0,
            n_steps=1,  # Use a single step for simplicity
            added_chain_HKL=added_chain_HKL,
            added_chain_asu=added_chain_asu,
            verbose=False,
        )

        # Verify that llgloss was called at least once
        assert len(call_args_list) > 0

        # Verify that the most recent call included the added_chain parameters
        last_call = call_args_list[-1]
        assert torch.equal(last_call["added_chain_HKL"], added_chain_HKL)
        assert torch.equal(last_call["added_chain_asu"], added_chain_asu)

        # Check the loss track has expected values
        assert len(loss_track) == 1
        assert loss_track[0] == 5.0


def test_quaternions_to_SO3_identity():
    """Test quaternions_to_SO3 with identity quaternion."""

    # Identity quaternion [r,i,j,k] = [1,0,0,0]
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # Apply the function
    rotation_matrix = quaternions_to_SO3(q)

    # Check that the result is the identity matrix
    expected = torch.eye(3)
    assert torch.allclose(rotation_matrix, expected)


def test_quaternions_to_SO3_rotation():
    """Test quaternions_to_SO3 with a 90-degree rotation around z-axis."""

    # Quaternion for 90-degree rotation around z-axis
    # [r,i,j,k] = [cos(45°),0,0,sin(45°)]
    # 45° is half of the desired rotation angle
    import math

    angle = math.pi / 4  # 45 degrees
    q = torch.tensor([math.cos(angle), 0.0, 0.0, math.sin(angle)])

    # Apply the function
    rotation_matrix = quaternions_to_SO3(q)

    # Expected: 90-degree rotation around z-axis
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert torch.allclose(rotation_matrix, expected, atol=1e-6)


def test_quaternions_to_SO3_normalization():
    """Test that quaternions_to_SO3 properly normalizes the quaternion."""
    from rocket.coordinates import quaternions_to_SO3

    # Non-normalized quaternion
    q = torch.tensor([2.0, 0.0, 0.0, 0.0])

    # Apply the function
    rotation_matrix = quaternions_to_SO3(q)

    # Should still produce the identity matrix
    expected = torch.eye(3)
    assert torch.allclose(rotation_matrix, expected)


def test_quaternions_to_SO3_batch():
    """Test quaternions_to_SO3 with batched quaternions."""
    from rocket.coordinates import quaternions_to_SO3

    # Batch of quaternions: identity and 180-degree rotation around x-axis
    qs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # identity
        [0.0, 1.0, 0.0, 0.0],  # 180° around x
    ])

    # Apply the function
    rotation_matrices = quaternions_to_SO3(qs)

    # Expected matrices
    expected = torch.tensor([
        [  # Identity
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [  # 180° around x
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
    ])

    assert torch.allclose(rotation_matrices, expected)


def test_construct_SO3():
    """Test construct_SO3 function."""

    # Test with orthogonal vectors
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])

    R = construct_SO3(v1, v2)

    # Should be identity matrix
    expected = torch.eye(3)
    assert torch.allclose(R, expected)

    # Test with non-orthogonal vectors
    v1 = torch.tensor([1.0, 1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 1.0])

    R = construct_SO3(v1, v2)

    # Check that R is orthogonal (R^T * R = I)
    identity = torch.matmul(R.T, R)
    assert torch.allclose(identity, torch.eye(3), atol=1e-6)

    # Check that det(R) = 1 (proper rotation)
    det = torch.det(R)
    assert torch.isclose(det, torch.tensor(1.0))


def test_decompose_SO3():
    """Test decompose_SO3 function."""

    # Create a rotation matrix
    theta = torch.tensor(0.5)  # some angle
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Decompose it
    v1, v2 = decompose_SO3(R)

    # Reconstruct the rotation matrix
    R_recon = construct_SO3(v1, v2)

    # Check they match
    assert torch.allclose(R, R_recon, atol=1e-6)

    # Test with custom a, b, c parameters
    v1, v2 = decompose_SO3(R, a=2.0, b=1.5, c=0.5)

    # The reconstructed matrix should still be the same
    R_recon = construct_SO3(v1, v2)
    assert torch.allclose(R, R_recon, atol=1e-6)

    # Test exception with c=0
    with pytest.raises(AssertionError):
        decompose_SO3(R, c=0.0)


def test_select_CA_elements():
    """Test select_CA_elements function."""

    # Test with mixed atom types
    cra_names = ["A-1-GLY-CA", "A-1-GLY-N", "A-2-ALA-CA", "A-2-ALA-CB"]

    result = select_CA_elements(cra_names)

    expected = [True, False, True, False]
    assert result == expected


def test_select_CA_from_craname():
    """Test select_CA_from_craname function."""

    # Test with mixed atom types
    cra_names = ["A-1-GLY-CA", "A-1-GLY-N", "A-2-ALA-CA", "A-2-ALA-CB"]

    ca_list, mask = select_CA_from_craname(cra_names)

    expected_list = ["A-1-GLY-CA", "A-2-ALA-CA"]
    expected_mask = [True, False, True, False]

    assert ca_list == expected_list
    assert mask == expected_mask


def test_calculate_mse_loss_per_residue():
    """Test calculate_mse_loss_per_residue function."""

    # Create two tensors with atom coordinates
    tensor1 = torch.tensor([
        [1.0, 1.0, 1.0],  # residue 1
        [2.0, 2.0, 2.0],  # residue 1
        [3.0, 3.0, 3.0],  # residue 2
        [4.0, 4.0, 4.0],  # residue 3
    ])

    tensor2 = torch.tensor([
        [1.1, 1.1, 1.1],  # residue 1
        [2.1, 2.1, 2.1],  # residue 1
        [3.1, 3.1, 3.1],  # residue 2
        [4.1, 4.1, 4.1],  # residue 3
    ])

    # Residue numbers
    residue_numbers = [1, 1, 2, 3]

    # Calculate MSE loss per residue
    mse_losses = calculate_mse_loss_per_residue(tensor1, tensor2, residue_numbers)

    # Expected results:
    # Residue 1: sqrt(mean(sum((1,1,1)-(1.1,1.1,1.1))^2 + sum((2,2,2)-(2.1,2.1,2.1))^2)) = sqrt(3*0.1^2 + 3*0.1^2) = sqrt(0.06)  # noqa: E501
    # Residue 2: sqrt(sum((3,3,3)-(3.1,3.1,3.1))^2) = sqrt(0.03)
    # Residue 3: sqrt(sum((4,4,4)-(4.1,4.1,4.1))^2) = sqrt(0.03)

    expected = [
        np.sqrt(0.06),  # residue 1 with 2 atoms
        np.sqrt(0.03),  # residue 2 with 1 atom
        np.sqrt(0.03),  # residue 3 with 1 atom
    ]

    assert len(mse_losses) == 3
    for actual, expecte in zip(mse_losses, expected, strict=False):
        assert np.isclose(actual, expecte)


@patch("builtins.open", new_callable=MagicMock)
def test_write_pdb_with_positions(mock_open):
    """Test write_pdb_with_positions function."""

    # Create mock file handles
    mock_file_in = MagicMock()
    mock_file_out = MagicMock()

    # Configure mock_open to return different file handles for input and output
    mock_open.side_effect = [mock_file_in, mock_file_out]

    # Set up mock file content
    mock_file_in.__enter__.return_value.readlines = lambda: [
        "HEADER    TEST FILE\n",
        "ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 10.00           N\n",  # noqa: E501
        "ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00 10.00           C\n",  # noqa: E501
        "TER\n",
    ]
    mock_file_in.__enter__.return_value.__iter__ = lambda self: iter(self.readlines())

    # Positions to write
    positions = [[12.345, 23.456, 34.567], [45.678, 56.789, 67.890]]

    # Call function
    write_pdb_with_positions("input.pdb", positions, "output.pdb")

    # Check that files were opened correctly
    mock_open.assert_any_call("input.pdb")
    mock_open.assert_any_call("output.pdb", "w")

    # Check that write was called with correct content
    calls = mock_file_out.__enter__.return_value.write.call_args_list

    # First line should be header (unchanged)
    assert calls[0][0][0] == "HEADER    TEST FILE\n"

    # Second line should be first ATOM with updated coordinates
    expected_line1 = "ATOM      1  N   ALA A   1      12.345  23.456  34.567  1.00 10.00           N\n"  # noqa: E501
    assert calls[1][0][0] == expected_line1

    # Third line should be second ATOM with updated coordinates
    expected_line2 = "ATOM      2  CA  ALA A   1      45.678  56.789  67.890  1.00 10.00           C\n"  # noqa: E501
    assert calls[2][0][0] == expected_line2

    # Fourth line should be TER (unchanged)
    assert calls[3][0][0] == "TER\n"


@pytest.fixture
def mock_unit_cell():
    """Create a mock unit cell with known fractionalization matrix."""
    mock_cell = MagicMock()
    # Create a simple fractionalization matrix
    # This matrix converts from orthogonal to fractional coordinates
    mock_cell.fractionalization_matrix = np.array([
        [0.1, 0.0, 0.0],  # a* = 0.1
        [0.0, 0.1, 0.0],  # b* = 0.1
        [0.0, 0.0, 0.1],  # c* = 0.1
    ])
    return mock_cell


def test_fractionalize_torch_simple_matrix(mock_unit_cell):
    """Test fractionalize_torch with a simple fractionalization matrix."""

    # Create a simple set of orthogonal coordinates
    atom_pos_orth = torch.tensor([
        [10.0, 0.0, 0.0],  # 10 Å along x-axis
        [0.0, 20.0, 0.0],  # 20 Å along y-axis
        [0.0, 0.0, 30.0],  # 30 Å along z-axis
        [5.0, 5.0, 5.0],  # 5 Å along each axis
    ])

    # Call the function
    atom_pos_frac = fractionalize_torch(atom_pos_orth, mock_unit_cell, device="cpu")

    # With our simple fractionalization matrix (0.1 on diagonal),
    # orthogonal coordinates are simply multiplied by 0.1
    expected_frac = torch.tensor([
        [1.0, 0.0, 0.0],  # 10 * 0.1 = 1.0 along a
        [0.0, 2.0, 0.0],  # 20 * 0.1 = 2.0 along b
        [0.0, 0.0, 3.0],  # 30 * 0.1 = 3.0 along c
        [0.5, 0.5, 0.5],  # 5 * 0.1 = 0.5 along each axis
    ])

    assert torch.allclose(atom_pos_frac, expected_frac)


def test_extract_allatoms_with_mocking():
    """Test extract_allatoms with mocked residue_constants."""

    # Create minimal test data
    cra_name_sfc = [
        "A-0-GLY-N",
        "A-0-GLY-CA",
        "A-0-GLY-C",
        "A-0-GLY-O",
        "A-1-ALA-N",
        "A-1-ALA-CA",
        "A-1-ALA-C",
        "A-1-ALA-O",
        "A-1-ALA-CB",
    ]

    # Let's see key parts of the function
    # 1. It expects residue_constants.atom_types to be a list of atom names
    # 2. It expects feats["aatype"] to contain residue type indices
    # 3. It expects outputs["final_atom_mask"] to be [n_res, N_atom_types]
    # 4. It expects outputs["final_atom_positions"] to be [n_res, N_atom_types, 3]

    # We'll patch more of the function's internal calls to isolate the test

    with (
        patch(
            "rocket.coordinates.residue_constants.atom_types",
            ["N", "CA", "C", "O", "CB"],
        ),
        patch(
            "rocket.coordinates.residue_constants.restype_1to3",
            {0: "GLY", 1: "ALA", -1: "UNK"},
        ),
    ):  # noqa: E501
        # Create expected outputs for different stages of the function
        atom_type_indices = {  # noqa: F841
            "N": 0,
            "CA": 1,
            "C": 2,
            "O": 3,
            "CB": 4,
        }

        # Mock atom_positions that will be returned at the end
        expected_positions = torch.tensor([
            [1.0, 2.0, 3.0],  # GLY-N
            [2.0, 3.0, 4.0],  # GLY-CA
            [3.0, 4.0, 5.0],  # GLY-C
            [4.0, 5.0, 6.0],  # GLY-O
            [5.0, 6.0, 7.0],  # ALA-N
            [6.0, 7.0, 8.0],  # ALA-CA
            [7.0, 8.0, 9.0],  # ALA-C
            [8.0, 9.0, 10.0],  # ALA-O
            [9.0, 10.0, 11.0],  # ALA-CB
        ])

        # Mock plddt that will be returned
        expected_plddt = torch.tensor([
            90.0,
            90.0,
            90.0,
            90.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
        ])  # noqa: E501

        # Now instead of mocking np.char.add, we'll mock the whole extraction process
        with (
            patch("torch.cat", return_value=expected_positions),
            patch("torch.repeat_interleave", return_value=expected_plddt),
        ):
            # Create minimal outputs and feats that will pass through the mocked parts
            outputs = {
                "final_atom_mask": torch.tensor([
                    [1.0, 1.0, 1.0, 1.0, 0.0],  # GLY
                    [1.0, 1.0, 1.0, 1.0, 1.0],  # ALA
                ]),  # Just needs correct shape
                "final_atom_positions": torch.concat(
                    [
                        expected_positions[:5].unsqueeze(0),
                        expected_positions[4:].unsqueeze(0),
                    ],
                    axis=0,
                ),  # Will be mocked
                "plddt": torch.tensor([90.0, 80.0]),  # Per-residue scores
            }
            feats = {"aatype": torch.tensor([[0], [1]])}  # GLY and ALA

            # Call the function
            positions, plddt = extract_allatoms(outputs, feats, cra_name_sfc)

    # Verify the results
    assert torch.allclose(positions, expected_positions)
    assert torch.allclose(plddt, expected_plddt)


def test_extract_atoms_and_backbone():
    """Test extract_atoms_and_backbone function with minimal mocking."""

    # Create mock inputs
    outputs = {
        "final_atom_mask": torch.tensor([
            # First residue (GLY) - only backbone atoms
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            # Second residue (ALA) - backbone + CB
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ]),
        "final_atom_positions": torch.tensor([
            # GLY residue
            [
                [1.0, 2.0, 3.0],  # N
                [2.0, 3.0, 4.0],  # CA
                [3.0, 4.0, 5.0],  # C
                [4.0, 5.0, 6.0],  # O
                [0.0, 0.0, 0.0],  # CB (not present in GLY)
                [0.0, 0.0, 0.0],  # CG (not present)
                [0.0, 0.0, 0.0],  # CD (not present)
            ],
            # ALA residue
            [
                [5.0, 6.0, 7.0],  # N
                [6.0, 7.0, 8.0],  # CA
                [7.0, 8.0, 9.0],  # C
                [8.0, 9.0, 10.0],  # O
                [9.0, 10.0, 11.0],  # CB
                [0.0, 0.0, 0.0],  # CG (not present in ALA)
                [0.0, 0.0, 0.0],  # CD (not present)
            ],
        ]),
    }

    feats = {
        "aatype": torch.tensor([[0], [1]])  # GLY and ALA
    }

    # Mock residue_constants.atom_types
    with patch(
        "rocket.coordinates.residue_constants.atom_types",
        ["N", "CA", "C", "O", "CB", "CG", "CD"],
    ):
        # Call the function
        positions, backbone_mask = extract_atoms_and_backbone(outputs, feats)

    # Expected positions - Only atoms with mask=1 should be included
    expected_positions = torch.tensor([
        [1.0, 2.0, 3.0],  # GLY-N
        [2.0, 3.0, 4.0],  # GLY-CA
        [3.0, 4.0, 5.0],  # GLY-C
        [4.0, 5.0, 6.0],  # GLY-O
        [5.0, 6.0, 7.0],  # ALA-N
        [6.0, 7.0, 8.0],  # ALA-CA
        [7.0, 8.0, 9.0],  # ALA-C
        [8.0, 9.0, 10.0],  # ALA-O
        [9.0, 10.0, 11.0],  # ALA-CB
    ])

    # Expected backbone mask - True for N, CA, C, O atoms
    expected_backbone_mask = torch.tensor(
        [
            True,  # GLY-N
            True,  # GLY-CA
            True,  # GLY-C
            True,  # GLY-O
            True,  # ALA-N
            True,  # ALA-CA
            True,  # ALA-C
            True,  # ALA-O
            False,  # ALA-CB
        ],
        dtype=torch.bool,
    )

    # Check results
    assert torch.equal(positions, expected_positions)
    assert torch.equal(backbone_mask, expected_backbone_mask)


def test_extract_atoms_and_backbone_all_atoms():
    """Test extract_atoms_and_backbone with all atom types present."""
    # Create mock inputs with a more complex amino acid (e.g., LYS)
    outputs = {
        "final_atom_mask": torch.tensor([
            # LYS residue with all atoms
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ]),
        "final_atom_positions": torch.tensor([
            # LYS residue
            [
                [1.0, 1.0, 1.0],  # N
                [2.0, 2.0, 2.0],  # CA
                [3.0, 3.0, 3.0],  # C
                [4.0, 4.0, 4.0],  # O
                [5.0, 5.0, 5.0],  # CB
                [6.0, 6.0, 6.0],  # CG
                [7.0, 7.0, 7.0],  # CD
            ]
        ]),
    }

    feats = {
        "aatype": torch.tensor([[3]])  # LYS (arbitrary residue type)
    }

    # Mock residue_constants.atom_types
    with patch(
        "rocket.coordinates.residue_constants.atom_types",
        ["N", "CA", "C", "O", "CB", "CG", "CD"],
    ):
        # Call the function
        positions, backbone_mask = extract_atoms_and_backbone(outputs, feats)

    # Expected positions - All atoms should be included
    expected_positions = torch.tensor([
        [1.0, 1.0, 1.0],  # N
        [2.0, 2.0, 2.0],  # CA
        [3.0, 3.0, 3.0],  # C
        [4.0, 4.0, 4.0],  # O
        [5.0, 5.0, 5.0],  # CB
        [6.0, 6.0, 6.0],  # CG
        [7.0, 7.0, 7.0],  # CD
    ])

    # Expected backbone mask - True for N, CA, C, O atoms
    expected_backbone_mask = torch.tensor(
        [
            True,  # N
            True,  # CA
            True,  # C
            True,  # O
            False,  # CB
            False,  # CG
            False,  # CD
        ],
        dtype=torch.bool,
    )

    # Check results
    assert torch.equal(positions, expected_positions)
    assert torch.equal(backbone_mask, expected_backbone_mask)


def test_extract_bfactors():
    """Test the extract_bfactors function."""

    # Create a mock protein structure
    class MockProtein:
        def __init__(self):
            # Create atom mask: 2 residues with different numbers of atoms
            self.atom_mask = np.array([
                [1.0, 1.0, 1.0, 1.0, 0.0],  # First residue has 4 atoms
                [1.0, 1.0, 1.0, 1.0, 1.0],  # Second residue has 5 atoms
            ])

            # Create aatype (residue types)
            self.aatype = np.array([
                0,  # GLY
                1,  # ALA
            ])

            # Create b-factors: assign different values to each atom
            self.b_factors = np.array([
                [10.0, 15.0, 20.0, 25.0, 0.0],  # First residue B-factors
                [30.0, 35.0, 40.0, 45.0, 50.0],  # Second residue B-factors
            ])

    # Create the mock protein
    mock_prot = MockProtein()

    # Call the function
    b_factor_array = extract_bfactors(mock_prot)

    # Expected B-factors (only for atoms with mask >= 0.5)
    expected_b_factors = np.array([
        10.0,  # First residue, first atom
        15.0,  # First residue, second atom
        20.0,  # First residue, third atom
        25.0,  # First residue, fourth atom
        30.0,  # Second residue, first atom
        35.0,  # Second residue, second atom
        40.0,  # Second residue, third atom
        45.0,  # Second residue, fourth atom
        50.0,  # Second residue, fifth atom
    ])

    # Check results
    assert np.array_equal(b_factor_array, expected_b_factors)
    assert b_factor_array.shape == (9,)


def test_extract_bfactors_with_masked_atoms():
    """Test extract_bfactors with partially masked atoms."""

    # Create a mock protein structure with some atoms masked out
    class MockProtein:
        def __init__(self):
            # Create atom mask with some atoms masked out
            self.atom_mask = np.array([
                [1.0, 0.4, 1.0, 1.0, 0.0],  # Second atom is masked out (below 0.5)
                [1.0, 1.0, 0.3, 1.0, 1.0],  # Third atom is masked out (below 0.5)
            ])

            # Create aatype (residue types)
            self.aatype = np.array([
                0,  # GLY
                1,  # ALA
            ])

            # Create b-factors
            self.b_factors = np.array([
                [10.0, 15.0, 20.0, 25.0, 0.0],  # First residue B-factors
                [30.0, 35.0, 40.0, 45.0, 50.0],  # Second residue B-factors
            ])

    # Create the mock protein
    mock_prot = MockProtein()

    # Call the function
    b_factor_array = extract_bfactors(mock_prot)

    # Expected B-factors (only for atoms with mask >= 0.5)
    # Note: second atom of first residue and third atom of second residue are masked out
    expected_b_factors = np.array([
        10.0,  # First residue, first atom
        # 15.0 is skipped (masked out)
        20.0,  # First residue, third atom
        25.0,  # First residue, fourth atom
        30.0,  # Second residue, first atom
        35.0,  # Second residue, second atom
        # 40.0 is skipped (masked out)
        45.0,  # Second residue, fourth atom
        50.0,  # Second residue, fifth atom
    ])

    # Check results
    assert np.array_equal(b_factor_array, expected_b_factors)
    assert b_factor_array.shape == (7,)  # 9 total atoms minus 2 masked atoms


def test_kabsch_align_matrices_identity():
    """Test kabsch_align_matrices with identical tensors."""

    # Create identical tensors (a simple cube)
    tensor1 = torch.tensor([
        [0.0, 0.0, 0.0],  # Origin
        [1.0, 0.0, 0.0],  # X-axis point
        [0.0, 1.0, 0.0],  # Y-axis point
        [0.0, 0.0, 1.0],  # Z-axis point
        [1.0, 1.0, 1.0],  # Diagonal point
    ])
    tensor2 = tensor1.clone()

    # Apply the function
    centroid1, centroid2, rotation_matrix = kabsch_align_matrices(tensor1, tensor2)

    # Expected results:
    # - Centroids should be the same
    # - Rotation matrix should be identity since tensors are identical
    expected_centroid = torch.tensor([[0.4, 0.4, 0.4]])
    expected_rotation = torch.eye(3)

    # Check results
    assert torch.allclose(centroid1, expected_centroid)
    assert torch.allclose(centroid2, expected_centroid)
    assert torch.allclose(rotation_matrix, expected_rotation, atol=1e-5)


def test_kabsch_align_matrices_translation():
    """Test kabsch_align_matrices with translated tensors."""

    # Create two tensors with the same shape but different positions
    # The second tensor is shifted by [1, 2, 3]
    tensor1 = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    translation = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = tensor1 + translation

    # Apply the function
    centroid1, centroid2, rotation_matrix = kabsch_align_matrices(tensor1, tensor2)

    # Expected results:
    # - Centroids should reflect the translation
    # - Rotation matrix should be identity since there's only translation
    expected_centroid1 = torch.tensor([[0.4, 0.4, 0.4]])
    expected_centroid2 = expected_centroid1 + translation
    expected_rotation = torch.eye(3)

    # Check results
    assert torch.allclose(centroid1, expected_centroid1)
    assert torch.allclose(centroid2, expected_centroid2)
    assert torch.allclose(rotation_matrix, expected_rotation, atol=1e-5)


def test_kabsch_align_matrices_rotation():
    """Test kabsch_align_matrices with rotated tensors."""

    # Create a tensor representing a simple structure
    tensor1 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Create a rotation matrix (90 degrees around Z axis)
    angle = np.pi / 2  # 90 degrees
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    # Apply rotation to create tensor2
    tensor2 = torch.matmul(tensor1, rotation.T)

    # Apply the function
    centroid1, centroid2, calculated_rotation = kabsch_align_matrices(tensor1, tensor2)

    # Expected results:
    # - Centroids should be the same in magnitude (the origin point is moved)
    # - Rotation matrix should match the original rotation

    # Check that the calculated rotation is close to the expected rotation
    # Note: The calculated rotation will align tensor1 to tensor2, so it should be
    # the transpose of the original rotation
    assert torch.allclose(calculated_rotation, rotation, atol=1e-5)

    # Check that the centroids are at the same distance from origin
    assert torch.allclose(torch.norm(centroid1), torch.norm(centroid2))


def test_kabsch_align_matrices_complex():
    """Test kabsch_align_matrices with a more complex transformation."""

    # Create a tensor representing a protein backbone fragment
    tensor1 = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # N
            [1.5, 0.0, 0.0],  # CA
            [2.5, 1.2, 0.0],  # C
            [3.5, 1.5, 1.0],  # N
            [4.0, 2.5, 1.5],  # CA
        ],
        dtype=torch.float32,
    )

    # Create a transformation: rotation + translation
    angle = np.pi / 4  # 45 degrees
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([5.0, -3.0, 2.0])

    # Apply transformation manually
    centroid1_manual = torch.mean(tensor1, dim=0, keepdim=True)
    tensor1_centered = tensor1 - centroid1_manual
    tensor2 = (
        torch.matmul(tensor1_centered, rotation.T) + centroid1_manual + translation
    )  # noqa: E501

    # Apply the function
    centroid1, centroid2, calculated_rotation = kabsch_align_matrices(tensor1, tensor2)

    # Apply the calculated transformation to tensor1
    aligned_tensor1 = align_tensors(tensor1, centroid1, centroid2, calculated_rotation)

    # Check that the aligned tensor matches tensor2
    assert torch.allclose(aligned_tensor1, tensor2, atol=1e-6)

    # Check that the rotation matrix is orthogonal (R^T R = I)
    identity = torch.matmul(calculated_rotation.T, calculated_rotation)
    assert torch.allclose(identity, torch.eye(3), atol=1e-6)

    # Check that the determinant is 1 (proper rotation, not a reflection)
    assert torch.allclose(torch.det(calculated_rotation), torch.tensor(1.0))


def test_weighted_kabsch_basic():
    """Test weighted_kabsch function with mocked dependencies."""

    # Create test inputs
    moving_tensor = torch.tensor([
        [1.0, 2.0, 3.0],  # Atom 1
        [2.0, 3.0, 4.0],  # Atom 2
        [3.0, 4.0, 5.0],  # Atom 3
        [4.0, 5.0, 6.0],  # Atom 4
    ])

    ref_tensor = torch.tensor([
        [2.0, 3.0, 4.0],  # Atom 1 (moved)
        [3.0, 4.0, 5.0],  # Atom 2 (moved)
        [4.0, 5.0, 6.0],  # Atom 3 (moved)
        [5.0, 6.0, 7.0],  # Atom 4 (moved)
    ])

    # Create cra_name list with backbone atoms
    cra_name = [
        "A-1-GLY-N",
        "A-2-ALA-CA",
        "A-3-VAL-C",
        "A-4-LEU-O",
    ]

    # Create a mock Rotation class
    class MockRotation:
        def as_matrix(self):
            # Identity rotation
            return np.eye(3)

        @classmethod
        def align_vectors(cls, target, source, weights=None):
            # Return the mock rotation and a dummy residual
            return (MockRotation(), 0.0)

    # Expected result to return from align_tensors
    expected_aligned = torch.tensor([
        [2.0, 3.0, 4.0],  # Aligned with reference
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
    ])

    # Patch both Rotation class and align_tensors function
    with (
        patch("rocket.coordinates.Rotation", MockRotation),
        patch("rocket.coordinates.align_tensors", return_value=expected_aligned),
    ):
        # Call the function
        aligned_tensor = weighted_kabsch(moving_tensor, ref_tensor, cra_name)

    # Check that we got the expected tensor back
    assert torch.equal(aligned_tensor, expected_aligned)
