from typing import Union
import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from torch import nn # type: ignore
from lightning import LightningDataModule # type: ignore

from scvi.dataloaders import DataSplitter # type: ignore
from scvi.model._utils import get_max_epochs_heuristic, use_distributed_sampler # type: ignore
from scvi.train import TrainingPlan, TrainRunner # type: ignore
from scvi.utils._docstrings import devices_dsp # type: ignore

torch.set_float32_matmul_precision("high")

def plot_loss(model, save_path="../results/figures/loss.png"):
    plt.figure(figsize=(10, 6))
    
    train_loss = np.array(model.history["train_loss_epoch"])
    val_loss = np.array(model.history["validation_loss"])
    elbo_val = np.array(model.history["elbo_validation"])
    recon_val = np.array(model.history["reconstruction_loss_validation"])
    kl_local_val = np.array(model.history["kl_local_validation"])
    elbo = np.array(model.history["elbo_train"])
    recon = np.array(model.history["reconstruction_loss_train"])
    kl_local =  np.array(model.history["kl_local_train"])
    kl_global = np.array(model.history["kl_global_train"])
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="blue", linestyle="--")
    plt.plot(elbo, label="ELBO Training", color="orange")
    plt.plot(elbo_val, label="ELBO Validation", color="orange", linestyle="--")
    plt.plot(recon, label="Reconstruction Loss Training", color="green")
    plt.plot(recon_val, label="Reconstruction Loss Validation", color="green", linestyle="--")
    plt.title("Loss over epochs")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def split_train_val(adata, cell_type_key):
    train_indices = []
    val_indices = []
    cell_types = adata.obs[cell_type_key].unique()
    for cell_type in cell_types:
        #get indices of cell type
        cell_type_indices = np.where(adata.obs[cell_type_key] == cell_type)[0]
        #shuffle indices
        np.random.shuffle(cell_type_indices)
        #split indices into train (80%) and val (20%)
        split_index = int(len(cell_type_indices) * 0.8)
        cell_type_train_indices = cell_type_indices[:split_index]
        cell_type_val_indices = cell_type_indices[split_index:]
        #add indices to train and val lists
        train_indices.extend(cell_type_train_indices)
        val_indices.extend(cell_type_val_indices)
    #get numpy arrays of indices
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    return train_indices, val_indices


class MaskedWeights:
    
    def __init__(self, masks: torch.Tensor, module_type: str):
        self._masks = masks
        self.module_type = module_type
        if self.module_type == "decoder":
            self._masks = self._masks[::-1]
        self.mask_idx = 0
        dimensions = [mask.shape for mask in masks]
        print(f"{len(self._masks)} masks found for module {self.module_type}.")
        print(f"Mask dimensions for module {self.module_type}: {dimensions}")

    def reset(self):
        self.mask_idx = 0
        
    def __call__(self, module):
        for name, layer in module.named_children():
                if isinstance(layer, nn.Linear):
                    if self.mask_idx < len(self._masks):
                        #print(f"Masking layer {self.mask_idx} of {self.module_type}")
                        mask = self._masks[self.mask_idx]
                        if layer.in_features != mask.shape[1] or layer.out_features != mask.shape[0]:
                            raise ValueError(
                                f"Mask shape {mask.shape} does not match layer shape ({layer.in_features}, {layer.out_features})"
                            )
                        
                        layer.weight.data.mul_(mask)
                        self.mask_idx += 1

# Override the optimizer step to apply the mask after 
class MaskedTrainingPlan(TrainingPlan):
    def __init__(self, *args, masks: torch.Tensor, use_masking: str = "encoder", **kwargs):
        super().__init__(*args, **kwargs)
        self.use_masking = use_masking
        self.log_every_n_steps = 1
        self.training_step_idx = 0
        if self.use_masking == "encoder" or self.use_masking == "both":
            self.masker_encoder = MaskedWeights(masks, "encoder")
        if self.use_masking == "decoder" or self.use_masking == "both":
            self.masker_decoder = MaskedWeights(masks, "decoder")

        self.train_loss_history = []
        self.train_kl_history = []
        self.train_recon_history = []

    def optimizer_step(self, *args, **kwargs):
        out = super().optimizer_step(*args, **kwargs)
        if self.use_masking == "encoder" or self.use_masking == "both":
            self.masker_encoder.reset()
            self.module.z_encoder.encoder.fc_layers.apply(self.masker_encoder)
            if self.masker_encoder.mask_idx != len(self.masker_encoder._masks):
                raise ValueError(
                    f"Masker has not been applied to all layers of encoder. {self.masker_encoder.mask_idx} out of {len(self.masker_encoder._masks)} layers masked."
                )
        if self.use_masking == "decoder" or self.use_masking == "both":
            self.masker_decoder.reset()
            self.module.decoder.decoder.fc_layers.apply(self.masker_decoder)
        return out
    
    def training_step(self, batch, batch_idx):
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log(
            "train_loss",
            scvi_loss.loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        # next part is for the usage of scib-metrics autotune with scvi
        if scvi_loss.extra_metrics is not None and len(scvi_loss.extra_metrics.keys()) > 0:
            self.prepare_scib_autotune(scvi_loss.extra_metrics, "training")

        self.train_loss_history.append(scvi_loss.loss.detach().cpu())
        self.train_recon_history.append(scvi_loss.reconstruction_loss_sum.detach().cpu())
        self.train_kl_history.append(scvi_loss.kl_local_sum.detach().cpu())

        self.training_step_idx += 1

        return scvi_loss.loss

        


# Lightning train() method modified from UnsupervisedTrainingMixin
class InformedUnsupervisedTrainingMixin:
    _data_splitter_cls = DataSplitter
    _training_plan_cls = MaskedTrainingPlan
    _train_runner_cls = TrainRunner

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Union[int, None] = None,
        accelerator: str = "auto",
        devices: Union[int, str, list[int]] = "auto",
        train_size: Union[float, None] = None,
        validation_size: Union[float, None] = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: Union[dict, None] = None,
        plan_kwargs: Union[dict, None] = None,
        datamodule: Union[LightningDataModule, None] = None,
        **trainer_kwargs,
    ):
        """
        Train the model using the specified parameters.

        Parameters
        ----------
        max_epochs : int or None
            Maximum number of epochs to train.
        accelerator : str
            Accelerator to use for training (e.g., "gpu", "cpu").
        devices : int, str, or list[int]
            Devices to use for training.
        train_size : float or None
            Proportion of data to use for training.
        validation_size : float or None
            Proportion of data to use for validation.
        shuffle_set_split : bool
            Whether to shuffle the dataset when splitting.
        load_sparse_tensor : bool
            Whether to load sparse tensors.
        batch_size : int
            Batch size for training.
        early_stopping : bool
            Whether to use early stopping.
        datasplitter_kwargs : dict or None
            Additional arguments for the data splitter.
        plan_kwargs : dict or None
            Additional arguments for the training plan.
        datamodule : LightningDataModule or None
            Data module to use for training.
        **trainer_kwargs
            Additional arguments for the trainer.
        """
        if datamodule is not None and not self._module_init_on_train:
            raise ValueError(
                "Cannot pass in `datamodule` if the model was initialized with `adata`."
            )
        elif datamodule is None and self._module_init_on_train:
            raise ValueError(
                "If the model was not initialized with `adata`, a `datamodule` must be passed in."
            )

        if max_epochs is None:
            if datamodule is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            elif hasattr(datamodule, "n_obs"):
                max_epochs = get_max_epochs_heuristic(datamodule.n_obs)
            else:
                raise ValueError(
                    "If `datamodule` does not have `n_obs` attribute, `max_epochs` must be "
                    "passed in."
                )

        if datamodule is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            datamodule = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                shuffle_set_split=shuffle_set_split,
                distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
                load_sparse_tensor=load_sparse_tensor,
                **datasplitter_kwargs,
            )
        elif self.module is None:
            self.module = self._module_cls(
                datamodule.n_vars,
                n_batch=datamodule.n_batch,
                n_labels=getattr(datamodule, "n_labels", 1),
                n_continuous_cov=getattr(datamodule, "n_continuous_cov", 0),
                n_cats_per_cov=getattr(datamodule, "n_cats_per_cov", None),
                **self._module_kwargs,
            )

        plan_kwargs = plan_kwargs or {}
        masks = getattr(self.module, "masks", None)
        use_masking = getattr(self.module, "use_masking", "encoder")
        training_plan = self._training_plan_cls(
            self.module, 
            masks=masks, 
            use_masking=use_masking,
            **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        print("Trainer kwargs: ", trainer_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=datamodule,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )

        runner()
        return training_plan