from SFC_Torch import SFcalculator


def initial_cryoSFC(
    pdb_file,
    mtz_file,
    Elabel,
    PHIElabel,
    device,
    n_bins=15,
):
    """
    Initialize a SFC for docking target in cryoEM problem.
    Note, we override the attribute .Fo to store Emean, to easily compute scales
    Also no freeflag is needed for cryoEM cases
    Args:
        pdb_file  : path to initial dock model
        mtz_file  : path to map.mtz file
        Elabel    : column name for Emean
        PHIElabel : column name for PHIEmean
        device    : torch device
        n_bins    : number of bins in the resolution binning
    Return:
        SFcalculator with attributes
        .Fo is Emean from map.mtz
        .bins is resolution binning label
    """
    # it will store Emean in attribute Fo
    sfcalculator = SFcalculator(
        pdb_file,
        mtz_file,
        expcolumns=[Elabel, PHIElabel],
        freeflag="None",
        set_experiment=True,
        testset_value=0,
        device=device,
        n_bins=n_bins,
        mode="cryoem",
    )

    # SigF is actually PHIEmean, don't need it here
    sfcalculator.SigF = None

    # calculate fprotein from default pdb file, no solvent is needed
    sfcalculator.calc_fprotein()

    Ep = sfcalculator.calc_Ec(sfcalculator.Fprotein_HKL)
    sfcalculator.Fprotein_HKL = Ep

    # normalize the Fp, use it to replace the Fp, initialize scales
    # note @ Aug 27 by MH: treat Emean as Fmean
    # Ep = sfcalculator.calc_Ec(sfcalculator.Fprotein_HKL)
    # sfcalculator.Fprotein_HKL = Ep
    sfcalculator.get_scales_adam(sub_ratio=1.0)

    return sfcalculator
