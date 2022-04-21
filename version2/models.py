from basic_block import *
from diff_aug import *


class Generator(nn.Module):
    def __init__(self, depth=[4, 3, 2], latent_dim=128, initial_size=8, dim=96, heads=4, mlp_ratio=4, window_size=2):
        super(Generator, self).__init__()
        self.initial_size = initial_size
        self.latent_dim = latent_dim
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # N, 128 -> N, 8 * 8 * 96
        self.mlp = nn.Linear(self.latent_dim, (self.initial_size ** 2) * self.dim)

        # N, 8 * 8, 96 -> N, 8 * 8, 96
        self.decode1 = TransformerBlock(dim=dim, input_resolution=[self.initial_size, self.initial_size], depth=self.depth[0], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio)
        # N, 8 * 8, 96 -> N, 16 * 16, 48
        self.up1 = BilinearUpsample(input_resolution=[self.initial_size, self.initial_size], dim=dim, out_dim=(dim // 2))

        # N, 16 * 16, 48 -> N, 16 * 16, 48
        self.decode2 = TransformerBlock(dim=(dim // 2), input_resolution=[self.initial_size * 2, self.initial_size * 2], depth=self.depth[1], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio)
        # N, 16 * 16, 48 -> N, 32 * 32, 24
        self.up2 = BilinearUpsample([self.initial_size * 2, self.initial_size * 2], dim=(dim // 2), out_dim=(dim // 4))

        # N, 32 * 32, 24 -> N, 32 * 32, 24
        self.decode3 = TransformerBlock(dim=(dim // 4), input_resolution=[self.initial_size * 4, self.initial_size * 4], depth=self.depth[2], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=self.mlp_ratio)

        # N, 24, 32, 32 -> N, 3, 32, 32
        self.linear = nn.Sequential(nn.Conv2d(self.dim // 4, 3, 1, 1, 0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, noise):
        # N, 8 * 8 * 384 -> N, 8 * 8, 96
        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        # N, 8 * 8, 96 -> N, 8 * 8, 96
        x = self.decode1(x)
        # N, 8 * 8, 96 -> N, 16 * 16, 48
        x = self.up1(x)
        # N, 16 * 16, 48 -> N, 16 * 16, 48
        x = self.decode2(x)
        # N, 16 * 16, 48 -> N, 32 * 32, 24
        x = self.up2(x)
        # N, 32 * 32, 24 -> N, 32 * 32, 24
        x = self.decode3(x)
        # N, 32 * 32, 24 -> N, 24, 32 * 32 -> N, 24, 32, 32 -> N, 3, 32, 32
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 4, self.initial_size * 4, self.initial_size * 4))

        return x


class Discriminator(nn.Module):
    def __init__(self, diff_aug, image_size=32, patch_size=4, input_channels=3, dim=96, depth=[2, 2, 2], heads=4, \
        mlp_ratio=4, drop_rate=0., window_size=2, num_classes=1):
        super(Discriminator, self).__init__()

        if image_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        # num_patches: 8 * 8
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.window_size = window_size

        # split image into non-overlapping patches
        # N, 3, 32, 32 -> N, 8 * 8, 96
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=input_channels, embed_dim=dim)

        # absolute position embedding
        # N, 64, 96
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        self.droprate = nn.Dropout(p=drop_rate)

        # N, 8 * 8, 96 -> N, 8 * 8, 96
        self.encode1 = TransformerBlock(dim=self.dim, input_resolution=[image_size // 4, image_size // 4], depth=self.depth[0], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=mlp_ratio)
        # N, 8 * 8, 96 -> N, 4 * 4, 192
        self.down1 = PatchMerging(input_resolution=[image_size // 4, image_size // 4], dim=self.dim)

        # N, 4 * 4, 192 -> N, 4 * 4, 192
        self.encode2 = TransformerBlock(dim=(self.dim * 2), input_resolution=[image_size // 8, image_size // 8], depth=self.depth[1], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=mlp_ratio)
        # N, 4 * 4, 192 -> N, 2 * 2, 384
        self.down2 = PatchMerging(input_resolution=[image_size // 8, image_size // 8], dim=(self.dim * 2))

        # N, 2 * 2, 384 -> N, 2 * 2, 384
        self.encode3 = TransformerBlock(dim=(self.dim * 4), input_resolution=[image_size // 16, image_size // 16], depth=self.depth[2], \
            num_heads=self.heads, window_size=self.window_size, mlp_ratio=mlp_ratio)

        # N, 2 * 2, 384 -> N, 1
        self.norm = nn.LayerNorm(dim * 4)
        self.out = nn.Linear(dim * 4, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # data augmentation 
        x = DiffAugment(x, self.diff_aug)

        # N, 64, 96
        x = self.patch_embed(x)
        x += self.positional_embedding
        x = self.droprate(x)

        # N, 8 * 8, 96 -> N, 8 * 8, 96
        x = self.encode1(x)
        # N, 8 * 8, 96 -> N, 4 * 4, 192
        x = self.down1(x)
        # N, 4 * 4, 192 -> N, 4 * 4, 192
        x = self.encode2(x)
        # N, 4 * 4, 192 -> N, 2 * 2, 384
        x = self.down2(x)
        # N, 2 * 2, 384 -> N, 2 * 2, 384
        x = self.encode3(x)
        # N, 2 * 2, 384 -> N, 2 * 2, 384
        x = self.norm(x)
        # N, 2 * 2, 384 -> N, 1
        x = self.out(x[:, 0])
        
        return x