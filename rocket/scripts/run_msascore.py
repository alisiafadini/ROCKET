"""
Run LLG scoring for system with different msas
"""

import argparse
import pandas as pd
import numpy as np
import torch
import sys, os
import rocket
import os, glob, shutil
from tqdm import tqdm
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket import refinement_utils as rkrf_utils
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config
from openfold.data import feature_pipeline, data_pipeline

from loguru import logger


PRESET = "model_1_ptm"

def main():
    p = argparse.ArgumentParser(description=
    """
    Run LLG scoring for system with different msas
    """)
    p.add_argument("path", action="store", help="Path to parent folder")
    p.add_argument("system", action="store", help="file_id for the dataset")
    p.add_argument("-i", action='store', help='prefix for msas to use, path will prepend')
    p.add_argument("-o", action="store", help='name of output directory to write prediction and scoring to, path will prepend')
    p.add_argument(
        "--domain_segs",
        type=int,
        nargs="*",
        default=None,
        help=("A list of resid as domain boundaries"),
    )
    p.add_argument(
        "--additional_chain", action="store_true", help=("Additional Chain in ASU")
    )
    p.add_argument(
        "--init_recycling",
        default=4,
        type=int,
        help=("number of initial recycling"),
    )
    p.add_argument(
        "--free_flag",
        default="R-free-flags",
        type=str,
        help=("Column name of free flag"),
    )

    p.add_argument(
        "--testset_value",
        default=1,
        type=int,
        help=("Value for test set"),
    )
    p.add_argument(
        "--voxel_spacing",
        default=4.5,
        type=float,
        help=("Voxel spacing for solvent percentage estimation"),
    )
    p.add_argument(
        "--min_resolution",
        default=3.0,
        type=float,
        help=("min resolution cut"),
    )
    p.add_argument(
        "--chimera_profile", 
        action="store_true",
        help=("Use chimera profile")
    )

    p.add_argument(
        "--score_fullmsa", 
        action="store_true",
        help=("Also score the full msa")
    )

    config = p.parse_args()
    device = rk_utils.try_gpu()
    RBR_LBFGS = True

    output_directory_path = os.path.join(config.path, config.o)
    os.makedirs(output_directory_path, exist_ok=True)
    
    # Configure input paths
    logger.info(f"Working with system {config.system}", flush=True)
    tng_file = os.path.join(config.path, "ROCKET_inputs", f"{config.system}-Edata.mtz")
    input_pdb = os.path.join(config.path, "ROCKET_inputs", f"{config.system}-pred-aligned.pdb")

    if config.additional_chain:
        constant_fp_added_HKL = torch.load(
            "{p}/ROCKET_inputs/{r}_added_chain_atoms_HKL.pt".format(
                p=config.path, r=config.system
            )
        ).to(device=device)
        constant_fp_added_asu = torch.load(
            "{p}/ROCKET_inputs/{r}_added_chain_atoms_asu.pt".format(
                p=config.path, r=config.system
            )
        ).to(device=device)

    else:
        constant_fp_added_HKL = None
        constant_fp_added_asu = None

    # Initialize SFC
    logger.info(f"Initialize SFC and LLGloss...", flush=True)
    sfc = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=config.free_flag,
        device=device,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        spacing=config.voxel_spacing,
    )
    
    init_pos_bfactor = sfc.atom_b_iso.clone()
    reference_pos = sfc.atom_pos_orth.clone()

    sfc_rbr = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=config.free_flag,
        device=device,
        solvent=False,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        spacing=config.voxel_spacing,
    )

    # LLG initialization with resol cut
    llgloss = rkrf_utils.init_llgloss(
        sfc, tng_file, config.min_resolution, None
    )
    llgloss_rbr = rkrf_utils.init_llgloss(
        sfc_rbr, tng_file, config.min_resolution, None
    )

    af_bias = rocket.MSABiasAFv3(
        model_config(PRESET, train=True), PRESET
    ).to(device)
    af_bias.freeze()

    fasta_path = [f for ext in ('*.fa', '*.fasta') for f in glob.glob(os.path.join(config.path, ext))][0]

    # Edit by MH @ Nov 15, 2024, recombination of full msa and subset msa
    fullmsa_dir =  os.path.join(config.path, "alignments")
    data_processor = data_pipeline.DataPipeline(template_featurizer=None)
    fullmsa_feature_dict = rkrf_utils.generate_feature_dict(
        fasta_path,
        fullmsa_dir,
        data_processor,
        )
    # Do featurization
    afconfig = model_config(PRESET)
    afconfig.data.common.max_recycling_iters = config.init_recycling
    del afconfig.data.common.masked_msa
    afconfig.data.common.resample_msa_in_recycling = False  
    feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
    fullmsa_processed_feature_dict = feature_processor.process_features(
        fullmsa_feature_dict, mode='predict'
    )
    full_profile = fullmsa_processed_feature_dict["msa_feat"][:, :, 25:48].clone()

    
    
    # Save out the scoring statistics
    df = pd.DataFrame(
        columns = ["msa_name", "depth", "mean_plddt", "llg", "rfree", "rwork"]
    )
    df.to_csv(os.path.join(output_directory_path, "msa_scoring.csv"), index=False)
    
    
    if config.score_fullmsa:
        msa_name = "fullmsa"
        device_processed_features = rk_utils.move_tensors_to_device(
            fullmsa_processed_feature_dict, device=device
        )
        # Run the AF2 prediction
        af2_output, prevs = af_bias(
            device_processed_features,
            [None, None, None],
            num_iters=config.init_recycling,
            bias=False,
        )
        prevs = [tensor.detach() for tensor in prevs]
        deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
        af2_output, __ = af_bias(
            device_processed_features, deep_copied_prevs, num_iters=1, bias=False
        )
        plddt = torch.mean(af2_output["plddt"])
        # Position Kabsch Alignment
        aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
            af2_output=af2_output,
            device_processed_features=device_processed_features,
            cra_name=sfc.cra_name,
            best_pos=reference_pos,
            exclude_res=None,
            domain_segs=config.domain_segs,
            reference_bfactor=init_pos_bfactor,
        )
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
        llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach()

        # refine_sigmaA:
        llgloss, llgloss_rbr, Ecalc, Fc = rkrf_utils.update_sigmaA(
            llgloss=llgloss,
            llgloss_rbr=llgloss_rbr,
            aligned_xyz=aligned_xyz,
            constant_fp_added_HKL=constant_fp_added_HKL,
            constant_fp_added_asu=constant_fp_added_asu,
        )

        # Rigid body refinement (RBR) step
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss_rbr,
            sfc.cra_name,
            domain_segs=config.domain_segs,
            lbfgs=RBR_LBFGS,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            lbfgs_lr=150.0,
            verbose=False,
        )

        # LLG value
        llg = llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=True,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            )

        # Save postRBR PDB
        llgloss.sfc.atom_pos_orth = optimized_xyz
        llgloss.sfc.savePDB(
            f"{output_directory_path!s}/{msa_name}_postRBR.pdb"
        )
        plddt_i, llg_i, rfree_i, rwork_i = plddt.item(), llg.item(), llgloss.sfc.r_free.item(), llgloss.sfc.r_work.item()
        
        # Save out scoring
        df_tmp = pd.DataFrame(
            {
                "msa_name": [msa_name],
                "depth": [fullmsa_feature_dict["msa"].shape[0]],
                "mean_plddt": [plddt_i],
                "llg": [llg_i],
                "rfree": [rfree_i],
                "rwork": [rwork_i],
            }
        )
        df_tmp.to_csv(os.path.join(output_directory_path, "msa_scoring.csv"), mode="a", header=False, index=False)

    # Get available msas
    a3m_paths = glob.glob(os.path.join(config.path, config.i + "*.a3m"))
    print(f"{len(a3m_paths)} msa files available...", flush=True)
    a3m_paths.sort()
    
    for a3m_path in tqdm(a3m_paths):
        msa_name, ext = os.path.splitext(os.path.basename(a3m_path))
        data_processor = data_pipeline.DataPipeline(template_featurizer=None)
        temp_alignment_dir = os.path.join(os.path.dirname(a3m_path), "tmp_align")
        os.makedirs(temp_alignment_dir, exist_ok=True)
        shutil.copy(a3m_path, os.path.join(temp_alignment_dir, msa_name+".a3m"))
        feature_dict = rkrf_utils.generate_feature_dict(
            fasta_path,
            temp_alignment_dir,
            data_processor,
            )
        # Do featurization
        afconfig = model_config(PRESET)
        afconfig.data.common.max_recycling_iters = config.init_recycling
        del afconfig.data.common.masked_msa
        afconfig.data.common.resample_msa_in_recycling = False  
        feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict'
        )

        # # Edit by MH @ Nov 15, 2024, recombination of full msa and subset msa
        # recomb_feature_dict = fullmsa_processed_feature_dict.copy()
        # recomb_feature_dict["msa_feat"][:, :, 25:48] =  processed_feature_dict["msa_feat"][:, :, 25:48]
        # device_processed_features = rk_utils.move_tensors_to_device(
        #     recomb_feature_dict, device=device
        # )

        # Edit by MH @ Nov 20, 2024, try the chimera profile idea from Alisia
        if config.chimera_profile:
            sub_profile = processed_feature_dict["msa_feat"][:, :, 25:48].clone()
            processed_feature_dict["msa_feat"][:, :, 25:48] =  torch.where(sub_profile == 0.0, full_profile.clone(), sub_profile.clone())
        
        device_processed_features = rk_utils.move_tensors_to_device(
            processed_feature_dict, device=device
        )

        # Run the AF2 prediction
        af2_output, prevs = af_bias(
            device_processed_features,
            [None, None, None],
            num_iters=config.init_recycling,
            bias=False,
        )
        prevs = [tensor.detach() for tensor in prevs]
        deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
        af2_output, __ = af_bias(
            device_processed_features, deep_copied_prevs, num_iters=1, bias=False
        )
        plddt = torch.mean(af2_output["plddt"])
        # Position Kabsch Alignment
        aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
            af2_output=af2_output,
            device_processed_features=device_processed_features,
            cra_name=sfc.cra_name,
            best_pos=reference_pos,
            exclude_res=None,
            domain_segs=config.domain_segs,
            reference_bfactor=init_pos_bfactor,
        )
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
        llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach()

        # refine_sigmaA:
        llgloss, llgloss_rbr, Ecalc, Fc = rkrf_utils.update_sigmaA(
            llgloss=llgloss,
            llgloss_rbr=llgloss_rbr,
            aligned_xyz=aligned_xyz,
            constant_fp_added_HKL=constant_fp_added_HKL,
            constant_fp_added_asu=constant_fp_added_asu,
        )

        # Rigid body refinement (RBR) step
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss_rbr,
            sfc.cra_name,
            domain_segs=config.domain_segs,
            lbfgs=RBR_LBFGS,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            lbfgs_lr=150.0,
            verbose=False,
        )

        # LLG value
        llg = llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=True,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            )

        # Save postRBR PDB
        llgloss.sfc.atom_pos_orth = optimized_xyz
        llgloss.sfc.savePDB(
            f"{output_directory_path!s}/{msa_name}_postRBR.pdb"
        )
        plddt_i, llg_i, rfree_i, rwork_i = plddt.item(), llg.item(), llgloss.sfc.r_free.item(), llgloss.sfc.r_work.item()
        
        # Save out scoring
        df_tmp = pd.DataFrame(
            {
                "msa_name": [msa_name],
                "depth":[feature_dict["msa"].shape[0]],
                "mean_plddt": [plddt_i],
                "llg": [llg_i],
                "rfree": [rfree_i],
                "rwork": [rwork_i],
            }
        )
        df_tmp.to_csv(os.path.join(output_directory_path, "msa_scoring.csv"), mode="a", header=False, index=False)

        # delete the tmp alignment
        shutil.rmtree(temp_alignment_dir)
        


