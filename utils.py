import os
from copy import deepcopy
from typing import Optional, Union, Dict
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
from getcomm import calu, heap_com, heap_com_threshold, locate_community_BFS, heap_com_topk, heap_com_threshold_nonc
import numpy  as np
import json
def eval(args, model, dataloader, initmap, method_names, Discriminative_model, pre_model):
    model.eval()
    pred_comms = {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            X, node_mask, _ = model.sample_batch(batch, Discriminative_model, pre_model)
            sg = batch['graph'].cpu().to_networkx()
            seeds = torch.tensor(batch['seed']) + batch['sg_size'][:-1]
            if args.prob_normal == 'softmax':
                probx = X[node_mask].detach().cpu()
                probx = torch.sigmoid(probx)
                print(probx)
            elif args.prob_normal == 'minmax':
                for i in range(len(X)):
                    X[i] = (X[i] - min(X[i])) / (max(X[i]) - min(X[i]))
                probx = X[node_mask].detach().cpu()
            else:
                probx = X[node_mask].detach().cpu()
            for method_name in method_names:
                if method_name not in pred_comms.keys():
                    pred_comms[method_name] = []
                pred_comms[method_name].extend(get_com(args, seeds, sg, probx, batch, method_name))
    return pred_comms


def get_com(args, seeds, sg, probx, batch, method_name, predsize=[0]):
    comms = []
    com_method, threshold = method_name.split('+')
    threshold = float(threshold)
    # print(com_method,threshold)

    if com_method == 'heap_com_threshold':
        sg_coms = [heap_com_threshold(int(seed), sg, probx, threshold) for seed in seeds]
    elif com_method == 'heap_topk':
        sg_coms = [heap_com_topk(int(seed), sg, probx, size) for seed, size in zip(seeds, predsize)]
    elif com_method == 'heap':
        sg_coms = [heap_com(int(seed), sg, probx) for seed in seeds]
    elif com_method == 'heap_com_threshold_nonc':
        sg_coms = [heap_com_threshold_nonc(int(seed), sg, probx, threshold, list(batch['rmapper'][index].keys()),
                                           int(batch['sg_size'][index])) for index, seed in enumerate(seeds)]
    assert len(sg_coms) == len(batch['rmapper'])
    coms = []
    for index in range(len(sg_coms)):
        com = [batch['rmapper'][index][int(node - batch['sg_size'][index])] for node in sg_coms[index]]
        coms.append(com)
    for com in coms:
        if len(com):
            comms.append(com)
    return comms


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrDiscriminative can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrDiscriminative, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrDiscriminative, etc.
            patterns_to_ignore = ("metrDiscriminative1", "metrDiscriminative2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @overrides
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, batch, batch_idx, *args,
                             **kwargs) -> None:
        if self.original_state_dict != {}:
            # Replace EMA weights with training weights
            pl_module.load_state_dict(self.original_state_dict, strict=False)

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

        # Setup EMA for sampling in on_train_batch_end
        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        ema_state_dict = pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    @overrides
    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict
    ) -> dict:
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict):
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X):
        self.X = X

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.X[node_mask == 0] = - 1
        else:
            self.X = self.X * x_mask
        return self


def save_model(model, args, save_path, epoch=None, method=''):
    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    latest_save_path = os.path.join(save_path, f'{method}_checkpoint')
    final_save_path = os.path.join(save_path, f'{method}_checkpoint_{epoch}', )
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args},
        final_save_path
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args},
        latest_save_path
    )


def initialize_from_checkpoint(init_checkpoint, model):
    checkpoint = torch.load(init_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model, checkpoint['args']


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('set seed for random numpy and torch')
