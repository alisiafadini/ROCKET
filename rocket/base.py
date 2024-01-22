"""
Include modified subclasses of AlphaFold
"""
import re
from openfold.model.model import AlphaFold

# from openfold.utils.tensor_utils import tensor_tree_map
from rocket.utils import tensor_tree_map
from openfold.config import model_config
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.script_utils import get_model_basename


import torch


class MSABiasAFv1(AlphaFold):
    """
    AlphaFold with trainable bias in MSA space
    """

    def __init__(
        self,
        config,
        preset,
        params_root="/net/cci-gpu-00/raid1/scratch1/alisia/programs/openfold/openfold_xtal/openfold/resources/params/",
    ):
        super(MSABiasAFv1, self).__init__(config)

        # AlphaFold params
        params_path = params_root + f"params_{preset}.npz"
        model_basename = get_model_basename(params_path)
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(self, params_path, version=model_version)
        self.eval()  # without this, dropout enabled

    def freeze(self, skip_str=None):
        """
        freeze AF2 parameters, skip those parameters with str match
        """
        if skip_str is None:
            for params in self.parameters():
                params.requires_grad_(False)
        else:
            for name, params in self.named_parameters():
                if re.match(skip_str, name) is None:
                    params.requires_grad_(False)

    def unfreeze(self, skip_str=None):
        """
        unfreeze AF2 parameters, skip those parameters with str match
        """
        if skip_str is None:
            for params in self.parameters():
                params.requires_grad_(True)
        else:
            for name, params in self.named_parameters():
                if re.match(skip_str, name) is None:
                    params.requires_grad_(True)

    def _biasMSA(self, feats):
        feats["msa_feat"][:, :, 25:48] = (
            feats["msa_feat"][:, :, 25:48] + feats["msa_feat_bias"]
        )
        return feats

    def iteration(self, feats, prevs, _recycle=True, biasMSA=True):
        if biasMSA:
            feats = self._biasMSA(feats)
        return super(MSABiasAFv1, self).iteration(feats, prevs, _recycle)

    def forward(self, batch, num_iters=1, biasMSA=True):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

            num_iters:
                Number of recycling loops. Default 1, no recycling
        """
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats, prevs, _recycle=(num_iters > 1), biasMSA=biasMSA
                )

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs


class MSABiasAFv2(MSABiasAFv1):
    """
    AlphaFold with trainable bias + trainable linear combination in MSA space
    """

    """
    def _biasMSA(self, feats):
        print(feats["msa_feat"][:, :, 25:48].shape)
        print(feats["msa_feat_weights"].shape)
        feats["msa_feat"][:, :, 25:48] = (
            torch.einsum(
                "ijkl,in->njkl",
                feats["msa_feat"][:, :, 25:48],
                feats["msa_feat_weights"],
            )
            + feats["msa_feat_bias"]
        )
        return feats
    """

    def _biasMSA(self, feats):
        feats["msa_feat"][:, :, 25:48] = (
            torch.einsum(
                "ijk,in->njk",
                feats["msa_feat"][:, :, 25:48],
                feats["msa_feat_weights"],
            )
            + feats["msa_feat_bias"]
        )
        return feats


class MSABiasAFv3(MSABiasAFv1):
    """
    AlphaFold with trainable bias + trainable linear combination in MSA space
    """

    def _biasMSA(self, feats):
        feats["msa_feat"][:, :, 25:48] = (
            feats["msa_feat"][:, :, 25:48].clone() * feats["msa_feat_weights"]
            + feats["msa_feat_bias"]
        )
        return feats
