import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        map_input: bool = False,
    ) -> None:
        """
        NoMaD ViNT Encoder class

        map_input (MapMaD): False = the original NoMaD. True adds a map encoder whose token is appended last
        (slot context_size + 2) with its own mask (see vint_train/mapmad/README.md).
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.map_input = map_input

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        # Initialize the goal encoder
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + (3 if map_input else 2))
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True # Mask out the goal 
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)

        if map_input:
            from vint_train.mapmad.map_encoder import MapEncoder
            self.map_encoder = MapEncoder(out_dim=self.obs_encoding_size)
            # Key padding masks for [obs x (context_size + 1), goal, map], row = goal_mask + 2 * map_mask
            self.map_masks = torch.zeros((4, self.context_size + 3), dtype=torch.bool)
            self.map_masks[[1, 3], -2] = True  # photo goal hidden
            self.map_masks[[2, 3], -1] = True  # map hidden


    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None,
                map_img: torch.tensor = None, input_map_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device

        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        obs_encoding = self._encode_images(obs_img, goal_img)
        if self.map_input:
            return self._attend_with_map(obs_encoding, goal_mask, map_img, input_map_mask)

        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens

    def _encode_images(self, obs_img: torch.tensor, goal_img: torch.tensor) -> torch.Tensor:
        """(B, context_size + 2, D) tokens: one per context frame (oldest first), then the obs+goal token."""
        device = obs_img.device

        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)

        # Get the goal encoding
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1) # concatenate the obs image/context and goal image --> non image goal?
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # get encoding of this img 
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding) # avg pooling 
        
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # Get the observation encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        return obs_encoding

    def map_tokens(self, image_tokens: torch.Tensor, goal_mask: torch.Tensor, map_img: torch.Tensor,
                   map_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MapMaD: image tokens + map token (last), positional encoding added -> (tokens, key padding mask)."""
        device = image_tokens.device
        map_token = self.map_encoder(map_img.to(device)).unsqueeze(1)
        tokens = self.positional_encoding(torch.cat((image_tokens, map_token), dim=1))
        mask_row = goal_mask.long().to(device) + 2 * map_mask.long().to(device)
        return tokens, torch.index_select(self.map_masks.to(device), 0, mask_row)

    def _attend_with_map(self, image_tokens: torch.Tensor, goal_mask: torch.Tensor, map_img: torch.Tensor,
                         map_mask: torch.Tensor) -> torch.Tensor:
        """MapMaD attention + pooling. Hidden tokens are excluded from attention and from the pooling:
        map hidden -> NoMaD's own avg_pool_mask formula over the first context_size + 2 tokens (incl. its fixed
        scale when the goal is hidden); map shown -> plain mean over the visible tokens."""
        tokens, padding = self.map_tokens(image_tokens, goal_mask, map_img, map_mask)
        out = self.sa_encoder(tokens, src_key_padding_mask=padding)
        n_old = self.context_size + 2
        old_mask = torch.index_select(self.avg_pool_mask.to(out.device), 0, goal_mask.long().to(out.device)).unsqueeze(-1)
        pooled_map_hidden = torch.mean(out[:, :n_old] * old_mask, dim=1)
        visible = (~padding).float().unsqueeze(-1)
        pooled_map_shown = (out * visible).sum(dim=1) / visible.sum(dim=1)
        return torch.where(map_mask.to(out.device).bool().unsqueeze(-1), pooled_map_hidden, pooled_map_shown)



# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    