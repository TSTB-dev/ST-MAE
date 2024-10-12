import torch
from torch import nn, einsum
from torchvision import models
from torchvision.models import VGG19_Weights
from einops import rearrange

# check path
import sys
from models import vision_transformer as vit

SUPPORTED_BACKBONES = [
    "vgg19"
]

def get_vision_transformer(model_name):
    assert model_name in vit.SUPPORTED_MODELS, f"Model {model_name} not supported"
    return getattr(vit, model_name)

def get_backbone_model(model_name):
    assert model_name in SUPPORTED_BACKBONES, f"Model {model_name} not supported"
    if model_name == "vgg19":
        return models.vgg19(weights=VGG19_Weights)

def get_intermediate_output_hook(layer, input, output):
    SiameseTransitionMAE.intermediate_cache.append(output)

class SiameseTransitionMAE(nn.Module):
    intermediate_cache = []
    
    def __init__(
        self,
        input_res: int = 256,
        feature_res = 64,
        patch_size: int = 4,
        extract_indices=[3, 8, 17, 26],
        feature_dim: int = 960,
        backbone_model: str = "vgg19",
        stmae_model: str = "stmae_base",
    ):
        super(SiameseTransitionMAE, self).__init__()
        self.input_res = input_res
        self.feature_res = feature_res
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.extract_indices = extract_indices
        self.backbone = get_backbone_model(backbone_model)
        self.feature_proj = None
        
        self._register_hook()
        if self.org_feature_dim != feature_dim:
            self.feature_proj = nn.Conv2d(self.org_feature_dim, feature_dim, kernel_size=1)
        self.h, self.w = feature_res // patch_size, feature_res // patch_size
        self.num_patches = self.h * self.w
        
        self.stmae_enc = get_vision_transformer(stmae_model + "_enc")(feature_res, feature_dim, patch_size)
        self.stmae_dec = get_vision_transformer(stmae_model + "_dec")(self.num_patches, self.stmae_enc.emb_size)
        
        self.patch_embed = vit.PatchEmbed(feature_dim, patch_size, self.stmae_enc.emb_size)
        self.pos_embed = vit.PosEmbedding(self.stmae_enc.emb_size, self.num_patches)
        self.layer_norm = nn.LayerNorm(self.stmae_enc.emb_size)
        
        self.out_proj = nn.ConvTranspose2d(self.stmae_dec.emb_size, feature_dim, kernel_size=4, stride=4)
        
        self.intermediate_outputs = []
    
    def _reset_cache(self):
        SiameseTransitionMAE.intermediate_cache = []
    
    def _register_hook(self):
        self.layer_hooks = []
        feature_dim = 0
        for layer_idx in self.extract_indices:
            feature_dim += self.backbone.features[layer_idx-1].out_channels
            layer_to_hook = self.backbone.features[layer_idx]
            hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
            self.layer_hooks.append(hook)
        self.org_feature_dim = feature_dim
        
    def indices_to_mask(self, mask_indices, L):
        """Convert indices to binary mask.
        Args:
            masks_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
            L (int): The total number of patches.
        Returns:
            mask (tensor): The binary mask. shape of (B, L), where L is the number of patches.
        """
        B, M = mask_indices.shape
        masks = torch.zeros(B, L, device=mask_indices.device)
        masks.scatter_(dim=1, index=mask_indices, value=True)
        inverse_masks = torch.logical_not(masks).float()
        return masks, inverse_masks
    
    def mask_to_indices(self, masks):
        """Convert binary mask to indices.
        Args:
            masks (tensor): The binary mask. shape of (B, L), where L is the number of patches.
        Returns:
            mask_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
        """
        mask_indices_ = torch.nonzero(masks, as_tuple=False)  # (L, 2)
        mask_indices = []
        for i in range(masks.shape[0]):
            mask_idx = mask_indices_[mask_indices_[:, 0] == i, 1]
            mask_indices.append(mask_idx)
        mask_indices = torch.stack(mask_indices, dim=0)
        return mask_indices
        
    def extract_features(self, x: torch.Tensor):
        """Extract features from the backbone model, which is refered as "Local perceptual semantic representation" in the paper.
        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            extract_indices (list): List of indices to extract features from the backbone model.
        Returns:
            torch.Tensor: Extracted features, shape (B, C, H', W')
        Examples:
            >>> backbone = get_backbone_model("vgg19")
            >>> features = backbone.extract_features(x)  # x shape (B, 960, 64, 64)
        """
        
        with torch.no_grad():
            _ = self.backbone(x)
        self.intermediate_outputs = SiameseTransitionMAE.intermediate_cache
        self._reset_cache()
        
        for i, intermediate_output in enumerate(self.intermediate_outputs):
            self.intermediate_outputs[i] = nn.functional.interpolate(intermediate_output, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
        features = torch.cat(self.intermediate_outputs, dim=1)
        
        if self.feature_proj is not None:
            features = self.feature_proj(features)

        return features
    
    def calculate_loss(self, x: torch.Tensor, x_hat: torch.Tensor, lambda_cos=5):
        """Calculate the loss of the model.
        Args:
            x (torch.Tensor): Input feature tensor, shape (B, C, H, W)
            x_hat (torch.Tensor): Reconstructed feature tensor, shape (B, C, H, W')
            lambda_cos (float): The weight of the cosine similarity loss.
        Returns:
            torch.Tensor: The loss value.
        """
        assert x.shape == x_hat.shape, f"Input feature tensor and reconstructed feature tensor should have the same shape. Got {x.shape} and {x_hat.shape}"
        mse_loss = nn.functional.mse_loss(x, x_hat)
        cos_loss = 1 - nn.functional.cosine_similarity(x.flatten(1), x_hat.flatten(1)).mean()
        total_loss = mse_loss + lambda_cos * cos_loss
        return total_loss, mse_loss, cos_loss
    
    def forward(self, x: torch.Tensor, mask_indices: torch.Tensor):
        """Forward pass of the Siamese Transition MAE model.

        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            mask_indices (torch.Tensor): Mask indices for the input image, shape (B, N)
        Returns:
            torch.Tensor: Reconstructed image tensor, shape (B, 3, H', W')
            torch.Tensor: The loss value.
            list: List of attention weights
        """
        B, M = mask_indices.shape
        masks, masks_inv = self.indices_to_mask(mask_indices, self.num_patches)
        visible_indices = self.mask_to_indices(masks_inv)
        
        ### 1. Extract features from the backbone model
        # features shape (B, C, H', W')
        with torch.no_grad():
            features = self.extract_features(x)
            
        ### 2. Patch Embed & Pos Embed
        x = self.patch_embed(features)  # (b, n, e)
        x = self.pos_embed(x)  # (b, n, e)
        x = self.layer_norm(x)
        
        ### 2. Apply ST-MAE encoder (The process is mentioned as "Feature Transition" in the paper)
        
        # First, we need to split the patches into two groups: masked patches and visible patches
        x_masked = x[masks.bool()].view(B, M, -1)  # shape (B, N/2, D)
        x_visible = x[masks_inv.bool()].view(B, x.shape[1] - M, -1)  # shape (B, N/2, D)
        
        assert x_masked.shape[1] == x_visible.shape[1], f"Number of masked patches and visible patches should be the same. Got {x_masked.shape[1]} and {x_visible.shape[1]}"
        x_cat = torch.cat([x_masked, x_visible], dim=0) # shape (B*2, N/2, D)

        # Then we apply the ST-MAE encoder to the concatenated patches
        x, attns_enc = self.stmae_enc(x_cat)  # shape (B*2, N/2, D)
        
        ### 3. Apply ST-MAE decoder
        x_masked_enc, x_visible_enc = torch.split(x, B, dim=0)
        transition_loss = torch.nn.functional.mse_loss(x_masked, x_masked_enc) + \
                            torch.nn.functional.mse_loss(x_visible, x_visible_enc)
        
        # First, we need to apply positional encoding 
        x_masked_enc = self.pos_embed(x_masked_enc, apply_indices=visible_indices)  # encode position information of visible patches
        x_visible_enc = self.pos_embed(x_visible_enc, apply_indices=mask_indices)  # encode position information of masked patches
        
        # Then we apply the ST-MAE decoder to the masked and visible patches
        x_cat = torch.cat([x_masked_enc, x_visible_enc], dim=1)  # shape (B, N, D)
        x, attns_dec = self.stmae_dec(x_cat)  
        
        ### 4. Apply output projection
        x = rearrange(x, "b (h w) c -> b c h w", h=self.h, w=self.w)
        x = self.out_proj(x)  # shape (B, 3, H', W')
        
        ### 5. Calculate loss
        total_loss, mse_loss, cos_loss = self.calculate_loss(features, x)
        outputs = {
            "recon_features": x,
            "org_features": features,
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "cos_loss": cos_loss,
            "trans_loss": transition_loss, 
            "attns_enc": attns_enc,
            "attns_dec": attns_dec
        }
        return outputs
    
if __name__ == "__main__":
    x = torch.randn(4, 3, 256, 256)
    mask_indices = torch.arange(256)[:128].unsqueeze(0).repeat(4, 1)
    smae = SiameseTransitionMAE()
    x_hat, loss, trans_loss = smae(x, mask_indices, return_trans_loss=True)
    
    print(x_hat.shape, loss, trans_loss)
