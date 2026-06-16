WAN_WRAPPED_MODULE = "openwam.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedModule"
WAN_WRAPPED_LINEAR = "openwam.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedLinear"
WAN_WRAPPED_NON_RECURSE = "openwam.model.video_backbone.wan.shared.core.vram.layers.AutoWrappedNonRecurseModule"

VRAM_MANAGEMENT_MODULE_MAPS = {
    "openwam.model.video_backbone.wan.dit.WanModel": {
        "openwam.model.video_backbone.wan.dit.MLP": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.dit.DiTBlock": WAN_WRAPPED_NON_RECURSE,
        "openwam.model.video_backbone.wan.dit.Head": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.dit.RMSNorm": WAN_WRAPPED_MODULE,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
    },
    "openwam.model.video_backbone.wan.image_encoder.WanImageEncoder": {
        "openwam.model.video_backbone.wan.image_encoder.VisionTransformer": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
    },
    "openwam.model.video_backbone.wan.text_encoder.WanTextEncoder": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Embedding": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.text_encoder.T5RelativeEmbedding": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.text_encoder.T5LayerNorm": WAN_WRAPPED_MODULE,
    },
    "openwam.model.video_backbone.wan.vace.VaceWanModel": {
        "openwam.model.video_backbone.wan.dit.DiTBlock": WAN_WRAPPED_MODULE,
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv3d": WAN_WRAPPED_MODULE,
        "torch.nn.LayerNorm": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.dit.RMSNorm": WAN_WRAPPED_MODULE,
    },
    "openwam.model.video_backbone.wan.vae.WanVideoVAE": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
    "openwam.model.video_backbone.wan.vae.WanVideoVAE38": {
        "torch.nn.Linear": WAN_WRAPPED_LINEAR,
        "torch.nn.Conv2d": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.RMS_norm": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.CausalConv3d": WAN_WRAPPED_MODULE,
        "openwam.model.video_backbone.wan.vae.Upsample": WAN_WRAPPED_MODULE,
        "torch.nn.SiLU": WAN_WRAPPED_MODULE,
        "torch.nn.Dropout": WAN_WRAPPED_MODULE,
    },
}
