import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.clip import clip

class CLIPLoss(torch.nn.Module):
    def __init__(self, opt, clip_model='ViT-B/32'):
        super(CLIPLoss, self).__init__()

        self.args = opt
        self.model_name = clip_model
        self.model, clip_preprocess = clip.load(clip_model)
        self.visual_model = self.model.visual

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.lambda_vgg = self.args.lambda_vgg

        if self.lambda_vgg > 0:
            self.hook_handlers = []
            self.feat_keys = {}
            self.feat_tokens = {}
            self.gen_attn_weights = {}
            self._register_hooks(layer_ids=[8], facet='key')

    def _get_hook(self, facet):
        if facet in ['token']:
            def _hook(model, input, output):
                input = model.ln_1(input[0])
                attnmap = model.attn(input, input, input, need_weights=True, attn_mask=model.attn_mask)[1]
                self.feat_tokens[input.device].append(output[1:].permute(1, 0, 2))
                self.gen_attn_weights[input.device].append(attnmap)
            return _hook
        elif facet == 'feat':
            def _outer_hook(model, input, output):
                output = output[1:].permute(1, 0, 2)  # LxBxD -> BxLxD
                # TODO: Remember to add VisualTransformer ln_post, i.e. LayerNorm
                output = F.layer_norm(output, self.visual_model.ln_post.normalized_shape, \
                                      self.visual_model.ln_post.weight.type(output.dtype), \
                                      self.visual_model.ln_post.bias.type(output.dtype), \
                                      self.visual_model.ln_post.eps)
                output = output @ self.visual_model.proj
                self.feat_tokens[output.device].append(output)

            return _outer_hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            N, B, C = input.shape
            weight = module.in_proj_weight.detach()
            bias = module.in_proj_bias.detach()
            qkv = F.linear(input, weight, bias)[1:]  # remove cls key
            qkv = qkv.reshape(-1, B, 3, C).permute(2, 1, 0, 3)  # BxNxC
            self.feat_keys[input.device].append(qkv[facet_idx])

        return _inner_hook

    def _register_hooks(self, layer_ids=[11], facet='key'):
        for block_idx, block in enumerate(self.visual_model.transformer.resblocks):
            if block_idx in layer_ids:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook('token')))
                assert facet in ['key', 'query', 'value']
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)
        return self.model.encode_image(images)

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        if self.lambda_vgg > 0:
            self.feat_keys[img.device] = []
            self.feat_tokens[img.device] = []
            self.gen_attn_weights[img.device] = []
        self.encode_images(img)

    def remd_loss(self, tgt_tokens, style_tokens):
        '''
        REMD Loss referring to style transfer
        '''
        tgt_tokens /= tgt_tokens.clone().norm(dim=-1, keepdim=True)
        style_tokens /= style_tokens.clone().norm(dim=-1, keepdim=True)

        attn_weights = torch.bmm(tgt_tokens, style_tokens.permute(0, 2, 1))

        cost_matrix = 1 - attn_weights
        B, N, M = cost_matrix.shape
        row_values, row_indices = cost_matrix.min(dim=2)
        col_values, col_indices = cost_matrix.min(dim=1)

        row_sum = row_values.mean(dim=1)
        col_sum = col_values.mean(dim=1)

        overall = torch.stack([row_sum, col_sum], dim=1)
        return overall.max(dim=1)[0].mean()

    def forward(self, x, y):
        self.get_image_features(x)
        x_tokens = self.feat_tokens[x.device][0]
        x_tokens /= x_tokens.clone().norm(dim=-1, keepdim=True)

        self.get_image_features(y)
        y_tokens = self.feat_tokens[y.device][0]
        y_tokens /= y_tokens.clone().norm(dim=-1, keepdim=True)
        return self.remd_loss(x_tokens, y_tokens)