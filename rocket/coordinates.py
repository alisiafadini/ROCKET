"""
Functions relating model coordinates/PDB file modificaitons
"""

from rocket import utils
import torch
import numpy as np
from openfold.np import residue_constants
from SFC_Torch import SFcalculator
import torch.nn as nn
import time


"""
def rigidbody_refine(xyz, llgloss):
    initial_model = utils.assert_numpy(xyz)
    propose_rmcom = torch.tensor(
        initial_model - np.mean(initial_model, axis=0),
        device=llgloss.device,
        dtype=torch.float32,
    )
    propose_com = torch.tensor(
        np.mean(initial_model, axis=0), device=llgloss.device, dtype=torch.float32
    )

    # llgloss.sfc.get_scales_lbfgs()
    trans_vec, rot_v1, rot_v2, loss_track_pose = find_rigidbody_matrix(
            llgloss, propose_com, propose_rmcom, llgloss.device
        )

    
    transform = construct_SO3(rot_v1, rot_v2)
    optimized_xyz = torch.matmul(propose_rmcom, transform) + propose_com + trans_vec

    return optimized_xyz, loss_track_pose
"""


def find_rigidbody_matrix_lbfgs_quat(
    llgloss, propose_com, propose_rmcom, device, added_chain=None
):
    q = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True
    )
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

    loss_track_pose = pose_train_lbfgs_quat(
        llgloss,
        q,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
        added_chain=added_chain,
    )
    print(loss_track_pose)
    return trans_vec, q, loss_track_pose


def rigidbody_refine_quat(xyz, llgloss, lbfgs=False, added_chain=None):
    propose_rmcom = xyz - torch.mean(xyz, dim=0)
    propose_com = torch.mean(xyz, dim=0)

    # llgloss.sfc.get_scales_lbfgs()
    if lbfgs:
        trans_vec, q, loss_track_pose = find_rigidbody_matrix_lbfgs_quat(
            llgloss,
            propose_com.clone().detach(),
            propose_rmcom.clone().detach(),
            llgloss.device,
            added_chain=added_chain,
        )
    else:
        trans_vec, rot_v1, rot_v2, loss_track_pose = find_rigidbody_matrix(
            llgloss,
            propose_com.clone().detach(),
            propose_rmcom.clone().detach(),
            llgloss.device,
            added_chain=added_chain,
        )

    transform = quaternions_to_SO3(q)
    optimized_xyz = torch.matmul(propose_rmcom, transform) + propose_com + trans_vec

    return optimized_xyz, loss_track_pose


def pose_train_lbfgs_quat(
    llgloss,
    q,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=150.0,
    n_steps=15,
    loss_track=[],
    added_chain=None,
):
    def closure():
        optimizer.zero_grad()
        temp_R = quaternions_to_SO3(q)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(
            temp_model,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=False,
            added_chain=added_chain,
        )

        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(
        [q, trans_vec],
        lr=lr,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-3,
        max_iter=1,
    )

    for k in range(n_steps):
        temp = optimizer.step(closure)
        loss_track.append(temp.item())

    print(f"Step {k+1}/{n_steps} - Loss: {temp.item()}")
    return loss_track


def rigidbody_refine(xyz, llgloss, lbfgs=False):
    propose_rmcom = xyz - torch.mean(xyz, dim=0)
    propose_com = torch.mean(xyz, dim=0)

    # llgloss.sfc.get_scales_lbfgs()
    if lbfgs:
        trans_vec, rot_v1, rot_v2, loss_track_pose = find_rigidbody_matrix_lbfgs(
            llgloss,
            propose_com.clone().detach(),
            propose_rmcom.clone().detach(),
            llgloss.device,
        )
    else:
        trans_vec, rot_v1, rot_v2, loss_track_pose = find_rigidbody_matrix(
            llgloss,
            propose_com.clone().detach(),
            propose_rmcom.clone().detach(),
            llgloss.device,
        )

    transform = construct_SO3(rot_v1, rot_v2)
    optimized_xyz = torch.matmul(propose_rmcom, transform) + propose_com + trans_vec

    return optimized_xyz, loss_track_pose


def pose_train(
    llgloss,
    rot_v1,
    rot_v2,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=5e-4,
    n_steps=500,
    loss_track=[],
):
    def pose_steptrain(optimizer):
        temp_R = construct_SO3(rot_v1, rot_v2)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(temp_model, bin_labels=None, num_batch=1, sub_ratio=1.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    start_time = time.time()
    optimizer = torch.optim.Adam([rot_v1, rot_v2, trans_vec], lr=lr)
    for k in range(n_steps):
        # start_time = time.time()
        temp = pose_steptrain(optimizer)
        # elapsed_time = time.time() - start_time
        loss_track.append(temp)

    elapsed_time = time.time() - start_time
    print(f"Step {k+1}/{n_steps} - Time taken: {elapsed_time:.4f} seconds")

    return loss_track


def pose_train_lbfgs(
    llgloss,
    rot_v1,
    rot_v2,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=0.005,
    n_steps=50,
    loss_track=[],
):
    def closure():
        start_time_loop = time.time()
        optimizer.zero_grad()
        temp_R = construct_SO3(rot_v1, rot_v2)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec
        loss = -llgloss(temp_model, bin_labels=None, num_batch=1, sub_ratio=1.0)
        loss.backward()
        return loss

    start_time = time.time()

    optimizer = torch.optim.LBFGS(
        [rot_v1, rot_v2, trans_vec],
        lr=lr,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-3,
        max_iter=1,
    )

    for k in range(n_steps):
        temp = optimizer.step(closure)
        loss_track.append(temp.item())
    elapsed_time = time.time() - start_time
    print(f"Step {k+1}/{n_steps} - Time taken optimizer: {elapsed_time:.4f} seconds")
    return loss_track


"""
def pose_train(
    llgloss,
    rot_v1,
    rot_v2,
    trans_vec,
    propose_com,
    propose_rmcom,
    lr=1e-4,
    max_epochs_total=500,
    max_epochs_without_improvement=10,
    loss_threshold=0.1,
    loss_track=[],
):
    def pose_steptrain(optimizer):
        temp_R = construct_SO3(rot_v1, rot_v2)
        temp_model = torch.matmul(propose_rmcom, temp_R) + propose_com + trans_vec

        loss = -llgloss(temp_model, bin_labels=None, num_batch=1, sub_ratio=1.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    optimizer = torch.optim.Adam([rot_v1, rot_v2, trans_vec], lr=lr)

    best_loss = float("inf")
    epochs_without_improvement = 0

    # start_time = time.time()
    for epoch in range(max_epochs_total):
        temp = pose_steptrain(optimizer)
        loss_track.append(temp)

        # Check if the loss has improved
        if temp < best_loss - loss_threshold:
            best_loss = temp
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == max_epochs_without_improvement:
            # print(
            #    f"Stopping optimization as the loss has not improved for {max_epochs_without_improvement} epochs."
            # )
            break

    # elapsed_time = time.time() - start_time
    # print(
    #        f"Epoch {epoch + 1} - Loss: {temp:.4f} - Time taken: {elapsed_time:.4f} seconds"
    #    )

    return loss_track

"""


def find_rigidbody_matrix(llgloss, propose_com, propose_rmcom, device):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    unit_R = quaternions_to_SO3(q)
    v1, v2 = decompose_SO3(unit_R)
    rot_v1 = torch.tensor(utils.assert_numpy(v1), device=device, requires_grad=True)
    rot_v2 = torch.tensor(utils.assert_numpy(v2), device=device, requires_grad=True)
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

    loss_track_pose = pose_train(
        llgloss,
        rot_v1,
        rot_v2,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
    )
    return trans_vec, rot_v1, rot_v2, loss_track_pose


def find_rigidbody_matrix_lbfgs(llgloss, propose_com, propose_rmcom, device):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    unit_R = quaternions_to_SO3(q)
    v1, v2 = decompose_SO3(unit_R)
    rot_v1 = torch.tensor(utils.assert_numpy(v1), device=device, requires_grad=True)
    rot_v2 = torch.tensor(utils.assert_numpy(v2), device=device, requires_grad=True)
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True)

    loss_track_pose = pose_train_lbfgs(
        llgloss,
        rot_v1,
        rot_v2,
        trans_vec,
        propose_com,
        propose_rmcom,
        loss_track=[],
    )
    return trans_vec, rot_v1, rot_v2, loss_track_pose


def construct_SO3(v1, v2):
    """
    Construct a continuous representation of SO(3) rotation with two 3D vectors
    https://arxiv.org/abs/1812.07035
    Parameters
    ----------
    v1, v2: 3D tensors
        Real-valued tensor in 3D space
    Returns
    -------
    R, A 3*3 SO(3) rotation matrix
    """
    e1 = v1 / torch.norm(v1)
    u2 = v2 - e1 * torch.tensordot(e1, v2, dims=1)
    e2 = u2 / torch.norm(u2)
    e3 = torch.cross(e1, e2)
    R = torch.stack((e1, e2, e3)).T
    return R


def decompose_SO3(R, a=1, b=1, c=1):
    """
    Decompose the rotation matrix into the two vector representation
    This decomposition is not unique, so a, b, c can be set as arbitray constants you like
    C != 0
    Parameters
    ----------
    R: 3*3 tensors
        Real-valued rotation matrix
    Returns
    -------
    v1, v2: Two real-valued 3D tensors, as the continuous representation of the rotation matrix
    """
    assert c != 0, "Give a nonzero c!"
    v1 = a * R[:, 0]
    v2 = b * R[:, 0] + c * R[:, 1]

    return v1, v2


def quaternions_to_SO3(q):
    """
    Normalizes q and maps to group matrix.
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    """
    # q = assert_tensor(q, torch.float32)
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            1 - 2 * j * j - 2 * k * k,
            2 * (i * j - r * k),
            2 * (i * k + r * j),
            2 * (i * j + r * k),
            1 - 2 * i * i - 2 * k * k,
            2 * (j * k - r * i),
            2 * (i * k - r * j),
            2 * (j * k + r * i),
            1 - 2 * i * i - 2 * j * j,
        ],
        -1,
    ).view(*q.shape[:-1], 3, 3)


def select_CA_elements(data):
    return [element.endswith("-CA") for element in data]


def select_CA_from_craname(cra_name_list):
    boolean_mask = select_CA_elements(cra_name_list)
    cra_CAs_list = [
        element for element, mask in zip(cra_name_list, boolean_mask) if mask
    ]
    return cra_CAs_list, boolean_mask


def update_bfactors(plddts):
    # Use Tom Terwilliger's formula to convert plddt to Bfactor and update sfcalculator instance
    deltas = 1.5 * torch.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * torch.pi**2 * deltas**2) / 3

    return b_factors


def calculate_mse_loss_per_residue(tensor1, tensor2, residue_numbers):
    mse_losses = []
    mse_criterion = nn.MSELoss(reduction="mean")

    for residue in set(residue_numbers):
        # Find indices of atoms with the current residue number in tensor1
        indices1 = [i for i, x in enumerate(residue_numbers) if x == residue]

        if len(indices1) > 0:
            # Extract coordinates for atoms with the current residue number in tensor1
            coords1 = tensor1[indices1, :]

            # Extract coordinates for atoms with the current residue number in tensor2
            coords2 = tensor2[indices1, :]

            # Calculate MSE loss for the coordinates of atoms with the same residue number
            # mse_loss = mse_criterion(coords1, coords2)
            mse_loss = torch.sqrt(torch.sum((coords1 - coords2) ** 2))
            mse_losses.append(mse_loss.item())

    return mse_losses


def write_pdb_with_positions(input_pdb_file, positions, output_pdb_file):
    # positions here expected to be rounded to 3 decimal points

    with open(input_pdb_file, "r") as f_in, open(output_pdb_file, "w") as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                atom_info = line[
                    :30
                ]  # Extract the first 30 characters containing atom information
                rounded_pos = positions.pop(
                    0
                )  # Pop the first rounded position from the list
                new_line = (
                    f"{atom_info}{rounded_pos[0]:8.3f}{rounded_pos[1]:8.3f}{rounded_pos[2]:8.3f}"
                    + line[54:]
                )
                f_out.write(new_line)
            else:
                f_out.write(line)


def fractionalize_torch(atom_pos_orth, unitcell, spacegroup, device=utils.try_gpu()):
    """
    Apply symmetry operations to real space asu model coordinates

    Parameters
    ----------
    atom_pos_orth: tensor, [N_atom, 3]
        ASU model ccordinates

    Will return fractional coordinates; Otherwise will return orthogonal coordinates

    Return
    ------
    atom_pos_sym_oped, [N_atoms, N_ops, 3] tensor in either fractional or orthogonal coordinates
    """
    atom_pos_orth.to(device=device)
    orth2frac_tensor = torch.tensor(
        unitcell.fractionalization_matrix.tolist(), device=device
    )
    atom_pos_frac = torch.tensordot(atom_pos_orth, orth2frac_tensor.T, 1)

    return atom_pos_frac


def extract_allatoms(outputs, feats, cra_name_sfc: list):
    atom_mask = outputs["final_atom_mask"]  # shape [n_res, 37]
    n_res = atom_mask.shape[0]

    # get atom positions in vectorized manner
    positions_atom = outputs["final_atom_positions"][atom_mask == 1.0]  # [n_atom, 3]

    # get plddt in vectorized manner
    plddt_atom = (
        outputs["plddt"].reshape([-1, 1]).repeat([1, 37])[atom_mask == 1.0]
    )  # shape [n_atom,]

    # get cra_name from AF2, [chain-resid-resname-atomname,...]
    res_names = utils.assert_numpy(
        [i + "-" for i in list(residue_constants.restype_1to3.values())] + ["UNK-"]
    )
    aatype = feats[
        "aatype"
    ]  # TODO: tackle the match between UNK and real non-standard aa name from SFC
    aatype_1d = res_names[utils.assert_numpy(aatype[:, 0], arr_type=int)]
    chain_resid = np.array(
        ["A-" + str(i) + "-" for i in range(n_res)]
    )  # TODO: here we assume all residues in same chain A
    crname_repeats = (
        np.char.add(chain_resid, aatype_1d).reshape(-1, 1).repeat(37, axis=-1)
    )  # [n_res, 37]
    crname_atom = crname_repeats[utils.assert_numpy(atom_mask) == 1]
    atom_types_repeats = (
        utils.assert_numpy(residue_constants.atom_types)
        .reshape(1, 37)
        .repeat(n_res, axis=0)
    )  # [n_res, 37]
    aname_atom = atom_types_repeats[utils.assert_numpy(atom_mask) == 1]
    cra_name_af = np.char.add(crname_atom, aname_atom).tolist()

    # reorder and assert the same topology
    reorder_index = utils.assert_numpy(
        [cra_name_af.index(i) for i in cra_name_sfc], arr_type=int
    )

    assert np.all(
        utils.assert_numpy(cra_name_af)[reorder_index]
        == utils.assert_numpy(cra_name_sfc)
    ), "Mismatch topolgy between AF and SFC!"

    return positions_atom[reorder_index], plddt_atom[reorder_index]


def extract_atoms_and_backbone(outputs, feats):
    atom_types = residue_constants.atom_types
    atom_mask = outputs["final_atom_mask"]
    pdb_lines = []
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    selected_atoms_mask = []

    n = aatype.shape[0]
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

            if atom_name in ["C", "CA", "O", "N"]:
                selected_atoms_mask.append(torch.tensor(1, dtype=torch.bool))
            else:
                selected_atoms_mask.append(torch.tensor(0, dtype=torch.bool))

    return torch.stack(pdb_lines), torch.stack(selected_atoms_mask)


def extract_bfactors(prot):
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    b_factors = prot.b_factors

    b_factor_lines = []

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for mask, b_factor in zip(atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            b_factor_lines.append(b_factor)

    return np.array(b_factor_lines)


def kabsch_align_matrices(tensor1, tensor2):
    # Center the atoms by subtracting their centroids
    centroid1 = torch.mean(tensor1, dim=0, keepdim=True)
    tensor1_centered = tensor1 - centroid1
    centroid2 = torch.mean(tensor2, dim=0, keepdim=True)
    tensor2_centered = tensor2 - centroid2

    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(tensor2_centered.t(), tensor1_centered)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = torch.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = torch.matmul(U, Vt)

    return centroid1, centroid2, rotation_matrix


def select_confident_atoms(current_pos, target_pos, bfacts=None):
    if bfacts is None:
        # If bfacts is None, set mask to all True
        reshaped_mask = torch.ones_like(current_pos, dtype=torch.bool)
    else:
        # Boolean mask for confident atoms
        mask = bfacts < 11.5
        reshaped_mask = mask.unsqueeze(1).expand_as(current_pos)

    # Select confident atoms using the mask
    current_pos_conf = torch.flatten(current_pos)[torch.flatten(reshaped_mask)]
    target_pos_conf = torch.flatten(target_pos)[torch.flatten(reshaped_mask)]

    N = current_pos_conf.numel() // 3

    return current_pos_conf.view(N, 3), target_pos_conf.view(N, 3)


def align_tensors(tensor1, centroid1, centroid2, rotation_matrix):
    tensor1_centered = tensor1 - centroid1

    # Apply the rotation and translation to align the first tensor to the second one
    aligned_tensor1 = torch.matmul(tensor1_centered, rotation_matrix.t()) + centroid2

    return aligned_tensor1


def align_positions(current_pos, target_pos, bfacts=None):
    # Perform alignment of positions
    # TO DO : add statement for when bfacts are not provided
    with torch.no_grad():
        current_pos_conf, target_pos_conf = select_confident_atoms(
            current_pos, target_pos, bfacts
        )

        centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
            current_pos_conf, target_pos_conf
        )

    aligned_pos = align_tensors(current_pos, centroid1, centroid2, rotation_matrix)
    return aligned_pos


def set_new_positions(orth_pos, frac_pos, sfmodel, device=utils.try_gpu()):
    sfmodel.atom_pos_orth = torch.squeeze(orth_pos, dim=1).to(device)
    sfmodel.atom_pos_frac = torch.squeeze(frac_pos, dim=1).to(device)
    return sfmodel


def transfer_positions(
    aligned_pos, sfcalculator_model, b_factors, device=utils.try_gpu()
):
    # Transfer positions to sfcalculator
    frac_pos = fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
        device=device,
    )
    sfcalculator_model = set_new_positions(
        aligned_pos, frac_pos, sfcalculator_model, device=device
    )

    # add bfactor calculation based on plddt
    # sfcalculator_model.atom_b_iso = b_factors

    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def update_sfcalculator(sfmodel):
    sfmodel.inspect_data(verbose=False)
    sfmodel.calc_fprotein()
    sfmodel.calc_fsolvent()
    sfmodel.init_scales(requires_grad=True)
    sfmodel.calc_ftotal()
    return sfmodel


def initialize_model_frac_pos(model_file, tng_file, device=utils.try_gpu()):
    sfcalculator_model = SFcalculator(
        model_file,
        tng_file,
        expcolumns=["FP", "SIGFP"],
        set_experiment=True,
        testset_value=0,
        device=device,
    )
    target_pos = sfcalculator_model.atom_pos_orth
    sfcalculator_model.atom_pos_frac = sfcalculator_model.atom_pos_frac * 0.00
    sfcalculator_model.atom_pos_orth = sfcalculator_model.atom_pos_orth * 0.00
    sfcalculator_model.atom_pos_frac.requires_grad = True

    return sfcalculator_model, target_pos
