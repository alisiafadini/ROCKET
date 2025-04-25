import numpy as np
from openfold import config
from openfold.data import data_pipeline, feature_pipeline
from openfold.np import protein, residue_constants

from rocket import utils as rk_utils


def make_decoy_seq(target_seq, template_resid, decoy_seq_ori):
    decoy_seq = ""
    j = 0
    for i in range(len(target_seq)):
        if i in template_resid:
            decoy_seq += decoy_seq_ori[j]
            j += 1
        else:
            decoy_seq += "-"
    return decoy_seq


def check_sequence(decoy_seqence, target_sequence):
    assert len(decoy_seqence) == len(target_sequence)
    for m, n in zip(decoy_seqence, target_sequence, strict=False):
        if m == n or m == "-":
            pass
        else:
            print(m, n, "do not match!", flush=True)
            return False
    return True


def make_processed_dict_from_template(
    template_pdb,
    target_seq=None,
    config_preset="model_1",
    msa_dict=None,
    device="cpu",
    seq_replacement="-",  # gap string
    mask_sidechains_add_cb=True,
    mask_sidechains=True,
    deterministic=True,
    max_recycling_iters=0,  # no recycling
):
    """
    template_pdb           :  path to pdb file used as template
    config_dict            :  the name of the model to get the parameters from.
                              Options: model_[1-5]
    msa_dict               :  dictionary containing precomputed msa properties in numpy
    device                 :  "cpu" or "cuda:i"
    mask_sidechains_add_cb :  mask out sidechain atoms except for C-Beta, add gly C-Beta
    mask_sidechains        :  mask out sidechain atoms except for C-Beta
    deterministic          :  make all data processing deterministic (no masking, etc.)
    """
    decoy_prot = protein.from_pdb_string(pdb_to_string(template_pdb))
    decoy_seq_ori = "".join([residue_constants.restypes[x] for x in decoy_prot.aatype])
    if target_seq is None:
        target_seq = decoy_seq_ori
        template_idxs = np.arange(len(decoy_prot.residue_index))
        decoy_seq = (
            seq_replacement * len(target_seq)
            if len(seq_replacement) == 1
            else target_seq
        )
    else:
        template_idxs = decoy_prot.residue_index - 1
        decoy_seq = make_decoy_seq(target_seq, template_idxs, decoy_seq_ori)
        assert check_sequence(decoy_seq, target_seq)
    template_idx_set = set(template_idxs)

    pos = np.zeros([1, len(target_seq), 37, 3])
    atom_mask = np.zeros([1, len(target_seq), 37])
    if mask_sidechains_add_cb:
        pos[0, template_idxs, :5] = decoy_prot.atom_positions[:, :5]
        backbone_modelled = np.asarray(
            np.all(decoy_prot.atom_mask[:, [0, 1, 2]] == 1, axis=1)
        )
        backbone_idx_set = set(template_idxs[backbone_modelled])
        projected_cb = [
            i
            for i, b, m in zip(
                template_idxs, backbone_modelled, decoy_prot.atom_mask, strict=False
            )
            if m[3] == 0 and b
        ]
        projected_cb_set = set(projected_cb)
        gly_idx = [i for i, a in enumerate(target_seq) if a == "G"]
        assert all(
            k in projected_cb_set
            for k in gly_idx
            if k in template_idx_set and k in backbone_idx_set
        )
        cbs = np.array(
            [
                extend(c, n, ca, 1.522, 1.927, -2.143)
                for c, n, ca in zip(
                    pos[0, :, 2], pos[0, :, 0], pos[0, :, 1], strict=False
                )
            ]
        )
        pos[0, projected_cb, 3] = cbs[projected_cb]
        atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:, :5]
        atom_mask[0, projected_cb, 3] = 1
    elif mask_sidechains:
        pos[0, template_idxs, :5] = decoy_prot.atom_positions[:, :5]
        atom_mask[0, template_idxs, :5] = decoy_prot.atom_mask[:, :5]
    else:
        pos[0, template_idxs] = decoy_prot.atom_positions
        atom_mask[0, template_idxs] = decoy_prot.atom_mask
    template = {
        "template_aatype": residue_constants.sequence_to_onehot(
            decoy_seq, residue_constants.HHBLITS_AA_TO_ID
        )[None],
        "template_all_atom_mask": atom_mask.astype(np.float32),
        "template_all_atom_positions": pos.astype(np.float32),
        "template_domain_names": np.asarray(["None"]),
    }

    # Make feature_dict
    feature_dict = {}
    feature_dict.update(
        data_pipeline.make_sequence_features(target_seq, "test", len(target_seq))
    )
    feature_dict.update(template)
    if msa_dict is None:
        # Make dummy msa features
        msa = [data_pipeline.make_dummy_msa_obj(target_seq)]
        msa_dict = data_pipeline.make_msa_features(msa)

    feature_dict.update(msa_dict)

    # create a config
    cfg = config.model_config(config_preset)
    cfg.data.common.max_recycling_iters = max_recycling_iters  # no
    if deterministic:
        cfg.data.eval.masked_msa_replace_fraction = 0.0
        cfg.data.predict.masked_msa_replace_fraction = 0.0

    # process_dict
    feature_processor = feature_pipeline.FeaturePipeline(cfg.data)
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode="predict"
    )
    if device != "cpu":
        processed_feature_dict = rk_utils.move_tensors_to_device(
            processed_feature_dict, device=device
        )

    return processed_feature_dict


"""
Read in a PDB file from a path
"""


def pdb_to_string(pdb_file):
    lines = []
    for line in open(pdb_file):  # noqa: SIM115
        if line[:6] == "HETATM" and line[17:20] == "MSE":
            line = "ATOM  " + line[6:17] + "MET" + line[20:]
        if line[:4] == "ATOM":
            lines.append(line)
    return "".join(lines)


"""
Function used to add C-Beta to glycine resides
input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
output: 4th coord
"""


def extend(a, b, c, L, A, D):
    def N(x):
        return x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-08)

    bc = N(b - c)
    n = N(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d, strict=False)])
