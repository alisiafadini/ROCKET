import rocket as rk
from copy import deepcopy
import torch
from openfold import config
from openfold.np import protein
from rocket import coordinates as rk_coordinates
from openfold.utils.tensor_utils import tensor_tree_map

processed_dict = rk.make_processed_dict_from_template(
    "./3hak/3hak_noalts.pdb", device="cuda:0"
)
cfg = config.model_config("model_1", train=True)
af2 = rk.TemplateBiasAF(cfg, "model_1").to("cuda:0")
af2.freeze()

processed_dict["template_torsion_angles_sin_cos_bias"] = torch.zeros_like(
    processed_dict["template_torsion_angles_sin_cos"],
    requires_grad=True,
    device="cuda:0",
)

processed_dict["template_all_atom_positions_bias"] = torch.zeros_like(
    processed_dict["template_all_atom_positions"], requires_grad=True, device="cuda:0"
)

lr_1 = 0.01
lr_2 = 0.01
optimizer = torch.optim.Adam(
    [
        {"params": processed_dict["template_torsion_angles_sin_cos_bias"], "lr": lr_1},
        {"params": processed_dict["template_all_atom_positions_bias"], "lr": lr_2},
    ]
)

for i in range(1000):
    optimizer.zero_grad()
    working_batch = deepcopy(processed_dict)
    working_batch["template_torsion_angles_sin_cos_bias"] = processed_dict[
        "template_torsion_angles_sin_cos_bias"
    ].clone()
    working_batch["template_all_atom_positions_bias"] = processed_dict[
        "template_all_atom_positions_bias"
    ].clone()
    af2_output = af2(working_batch, num_iters=1, bias=True)

    features_np = tensor_tree_map(
        lambda t: t[..., -1].cpu().detach().numpy(), working_batch
    )
    results_np = tensor_tree_map(lambda t: t.cpu().detach().numpy(), af2_output)
    prot = protein.from_prediction(features_np, results_np)
    aligned = rk_coordinates.align_positions(
        af2_output["final_atom_positions"],
        processed_dict["template_all_atom_positions"][0, :, :, :, 0],
    )
    loss = torch.sum(
        (aligned - processed_dict["template_all_atom_positions"][0, :, :, :, 0]) ** 2
    )
    with open("./testing_af2rank/{}_out.pdb".format(i), "w") as file:
        file.write(protein.to_pdb(prot))

    print(i)
    print(loss.item())
    loss.backward()
    optimizer.step()
