import warnings
import os
import sys
import numpy as np
from collections.abc import Iterable
from typing import Literal, Union

import anndata as AnnData
import torch

import scvi
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    BaseMinifiedModeModelClass,
)
from scvi.utils import setup_anndata_dsp
sys.path.append(os.path.abspath("."))
from vae import InformedVAE
from train import InformedUnsupervisedTrainingMixin


class InformedSCVI(EmbeddingMixin,RNASeqMixin,VAEMixin,ArchesMixin,InformedUnsupervisedTrainingMixin,BaseMinifiedModeModelClass,):
    _module_cls = InformedVAE
    _LATENT_QZM_KEY = "scvi_latent_qzm"
    _LATENT_QZV_KEY = "scvi_latent_qzv"
    
    def __init__(
        self,
        adata: Union[AnnData, None] = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        activation: Literal["relu", "leaky_relu", "tanh", "sigmoid"] = "relu",
        # Include masks
        masks: Iterable[np.ndarray] = None,
        use_masking: Literal["none", "encoder", "decoder", "both"] = "encoder",
        **kwargs,
    ):
        super().__init__(adata)

        # Get network shapes from masks
        n_layers = len(masks) if masks is not None else 1
        n_hidden = [mask.shape[0] for mask in masks] if masks is not None else 128
        n_latent = masks[-1].shape[0] // 2 if masks is not None else 10
        if masks is None:
            use_masking = "none"

        
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            "activation": activation,
            **kwargs,
        }
        self._model_summary_string = (
            "SCVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}, "
            f"activation: {activation}, use_masking: {use_masking}."
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            n_cats_per_cov = (
                self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                else None
            )
           
            n_batch = self.summary_stats.n_batch
            use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
            library_log_means, library_log_vars = None, None
            if (
                not use_size_factor_key
                and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                and not use_observed_lib_size
            ):
                library_log_means, library_log_vars = _init_library_size(
                    self.adata_manager, n_batch
                )
            
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                use_observed_lib_size=use_observed_lib_size,
                latent_distribution=latent_distribution,
                activation=activation,
                use_size_factor_key=use_size_factor_key,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                # Pass masks to VAE
                masks=masks,
                use_masking=use_masking,
                **kwargs,
            )
            self.module.minified_data_type = self.minified_data_type

        self.init_params_ = self._get_init_params(locals())
        
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Union[str, None] = None,
        batch_key: Union[str, None] = None,
        labels_key: Union[str, None] = None,
        size_factor_key: Union[str, None] = None,
        categorical_covariate_keys: Union[list[str], None] = None,
        continuous_covariate_keys: Union[list[str], None] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        print(f"setup_anndata: {cls.__name__}")
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_hidden_activations(
        self,
        adata=None,
        indices=None,
        batch_size=None,
        layers_to_capture=None,  # list of layer indices or names
    ):
        """
        Returns a dict of activations for each specified hidden layer.
        """
        # Prepare dataloader as in get_latent_representation
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # Prepare hooks
        activations = {}
        handles = []
        fc_layers = self.module.z_encoder.encoder.fc_layers

        # Default: capture all layers
        if layers_to_capture is None:
            layers_to_capture = list(range(len(fc_layers)))

        def get_hook(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())
            return hook

        # Register hooks
        for idx in layers_to_capture:
            # Each layer is a Sequential, so get the Linear sublayer
            seq = fc_layers[idx]
            for sub in seq:
                if isinstance(sub, torch.nn.Linear):
                    handles.append(sub.register_forward_hook(get_hook(f"layer_{idx}")))
                    break

        # Run batches through encoder
        for tensors in dataloader:
            x = self.module._get_inference_input(tensors)["x"]
            self.module.z_encoder.encoder(x)

        # Remove hooks
        for h in handles:
            h.remove()

        # Concatenate activations for each layer
        for k in activations:
            activations[k] = torch.cat(activations[k]).numpy()

        return activations