import numpy as np
import collections
from collections.abc import Callable, Iterable
from typing import Literal, Union
import pandas as pd
import torch
from torch import nn
from scvi import REGISTRY_KEYS
from scvi.nn import DecoderSCVI, Encoder, FCLayers
from scvi.module import VAE

torch.set_float32_matmul_precision("high")


def _identity(x):
    return x

# Class to build fully connected layers with different sizes for hidden layers
class InformedFCLayers(FCLayers):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_cont: int = 0,
        n_layers: int = 1,
        # n_hidden can be a list
        n_hidden: Union[int, Iterable[int]] = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super(FCLayers, self).__init__()
        self.inject_covariates = inject_covariates

        # Deal with n_hidden being an int or a list of ints
        if isinstance(n_hidden, int):
            hidden_sizes = [n_hidden] * (n_layers - 1)
        elif isinstance(n_hidden, list):
            if len(n_hidden) != (n_layers - 1):
                raise ValueError(f"Length of n_hidden list (len={len(n_hidden)}) must be n_layers - 1 (n_layers={n_layers})")
            hidden_sizes = n_hidden
        else:
            raise TypeError("n_hidden must be an int or a list of ints, instead got: " + str(type(n_hidden)))

        layers_dim = [n_in] + hidden_sizes + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.n_cov = n_cont + sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + self.n_cov * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

class InformedDecoderSCVI(DecoderSCVI):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: Union[int, Iterable[int]] = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super(DecoderSCVI, self).__init__()

        # Get last hidden layer size
        if isinstance(n_hidden, list):
            n_hidden_last = n_hidden[-1]
            n_hidden=n_hidden[:-1]
        else:
            n_hidden_last = n_hidden

        print(f"Activation function decoder: {activation_fn.__name__}")

        # Custom fully connected layers
        self.px_decoder = InformedFCLayers(
            n_in=n_input,
            n_out=n_hidden_last,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
            **kwargs,
        )
        
            
        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden_last, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden_last, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden_last, n_output)


class InformedEncoder(Encoder):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        # n_hidden can be a list
        n_hidden: Union[int, Iterable[int]] = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Union[Callable, None] = None,
        activation_fn: nn.Module = nn.ReLU, 
        return_dist: bool = False,
        **kwargs,
    ):
        super(Encoder, self).__init__()

        print(f"Activation function encoder: {activation_fn.__name__}")

        self.distribution = distribution
        self.var_eps = var_eps
        # Get last hidden layer size
        if isinstance(n_hidden, list):
            n_hidden_last = n_hidden[-1]
            n_hidden=n_hidden[:-1]
        else:
            n_hidden_last = n_hidden

        # Custom fully connected layers
        self.encoder = InformedFCLayers(
            n_in=n_input,
            n_out=n_hidden_last,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            **kwargs,
        )
        

        self.mean_encoder = nn.Linear(n_hidden_last, n_output)
        self.var_encoder = nn.Linear(n_hidden_last, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation


class InformedVAE(VAE):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        # n_hidden can be a list
        n_hidden: Union[int, Iterable[int]] = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Union[list[int], None] = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        extra_payload_autotune: bool = False,
        library_log_means: Union[np.ndarray, None] = None,
        library_log_vars: Union[np.ndarray, None] = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        activation: Literal["relu", "leaky_relu", "tanh", "sigmoid"] = "relu",
        #activation_encoder: Literal["relu", "leaky_relu", "tanh", "sigmoid"] = "relu",
        #activation_decoder: Literal["relu", "leaky_relu", "tanh", "sigmoid"] = "relu",
        # Include masks
        masks: Union[Iterable[np.ndarray], None] = None,
        use_masking: Literal["none", "encoder", "decoder", "both"] = "encoder",
        extra_encoder_kwargs: Union[dict, None] = None,
        extra_decoder_kwargs: Union[dict, None] = None,
        batch_embedding_kwargs: Union[dict, None] = None,
    ):

        super(VAE, self).__init__()

        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.extra_payload_autotune = extra_payload_autotune
        self.activation = activation
        #self.activation_encoder = activation_encoder
        #self.activation_decoder = activation_decoder
        self.use_masking = use_masking
        

        # Register masks as buffers
        if masks is not None:
            for i, mask in enumerate(masks):
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask).float()
                elif isinstance(mask, pd.DataFrame):
                    mask = torch.tensor(mask.values, dtype=torch.float32)
                elif not isinstance(mask, torch.Tensor):
                    raise ValueError("Masks must be numpy arrays, pandas dataframes or torch tensors.")
                self.register_buffer(f"mask_layer{i}", mask)

            self.masks = [getattr(self, f"mask_layer{i}") for i in range(n_layers)]
        else:
            self.masks = None

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "`dispersion` must be one of 'gene', 'gene-batch', 'gene-label', 'gene-cell'."
            )
        # activation_fn = [nn.ReLU, nn.ReLU]
        # for i, act in enumerate([self.activation_encoder, self.activation_decoder]):
        #     if act == "relu":
        #         activation_fn[i] = nn.ReLU
        #     elif act == "leaky_relu":
        #         activation_fn[i] = nn.LeakyReLU
        #     elif act == "tanh":
        #         activation_fn[i] = nn.Tanh
        #     elif act == "sigmoid":
        #         activation_fn[i] = nn.Sigmoid
        #     else:
        #         raise ValueError(
        #             f"'activation'{i} must be one of 'relu', 'leaky_relu', 'tanh', 'sigmoid'."
        #         )

        if self.activation == "relu":
            activation_fn = nn.ReLU
        elif self.activation == "leaky_relu":
            activation_fn = nn.LeakyReLU
        elif self.activation == "tanh":
            activation_fn = nn.Tanh
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid
        else:
            raise ValueError(
                f"'activation' must be one of 'relu', 'leaky_relu', 'tanh', 'sigmoid'."
            )

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        # Call custom encoder
        self.z_encoder = InformedEncoder(
            n_input=n_input_encoder,
            n_output=n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=activation_fn,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
                # l encoder goes from n_input-dimensional data to 1-d library size
        # Call custom encoder
        self.l_encoder = InformedEncoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=1,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        # Call custom decoder
        if isinstance(n_hidden, list):
            n_hidden_decoder = n_hidden[::-1]
        else:
            n_hidden_decoder = n_hidden

        self.decoder = InformedDecoderSCVI(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden_decoder,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            activation_fn=nn.ReLU,
            **_extra_decoder_kwargs,
        )