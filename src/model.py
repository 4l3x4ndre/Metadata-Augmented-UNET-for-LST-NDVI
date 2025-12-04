import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# Unet++ code from https://github.com/4uiiurz1/pytorch-nested-unet/
# VGGBlock for UNet++
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, seq_len, hidden_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x: (B, seq_len)
        x = x.unsqueeze(-1)  # (B, seq_len, 1) — feature dimension = 1
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # (B, hidden_dim)
        return self.fc(h)  # (B, out_dim)



class MetadataEncoder(nn.Module):
    def __init__(self, in_features, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        
    def forward(self, x):
        return self.fc(x)  # Output: (B, out_dim)


class UrbanPredictor_unetpp(nn.Module):
    def __init__(self, spatial_channels, seq_len, temporal_dim, meta_features, meta_dim, lstm_dim, out_channels, 
                 base_filters=32, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2,2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.embed_dim = temporal_dim + meta_dim

        # Encoders
        self.conv0_0 = VGGBlock(spatial_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # Decoders
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1]+temporal_dim+meta_dim, nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2]+temporal_dim+meta_dim, nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3]+temporal_dim+meta_dim, nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4]+temporal_dim+meta_dim, nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1]+temporal_dim+meta_dim, nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2]+temporal_dim+meta_dim, nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3]+temporal_dim+meta_dim, nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1]+temporal_dim+meta_dim, nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2]+temporal_dim+meta_dim, nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1]+temporal_dim+meta_dim, nb_filter[0], nb_filter[0])

        # embeddings
        self.temporal_encoder = TemporalEncoder(seq_len, hidden_dim=lstm_dim, out_dim=temporal_dim)
        self.meta_encoder = MetadataEncoder(meta_features, meta_dim)

        # Output layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def _broadcast_embeddings(self, temporal_emb, meta_emb, H, W):
        """Broadcast temporal and metadata embeddings to spatial dimensions"""
        B = temporal_emb.shape[0]
        
        # Concatenate temporal and metadata embeddings
        combined_emb = torch.cat([temporal_emb, meta_emb], dim=1)  # (B, temporal_dim + meta_dim)
        
        # Broadcast to spatial dimensions
        emb_map = combined_emb[:, :, None, None].expand(B, self.embed_dim, H, W)
        
        return emb_map

    
    def _upsample_match(self, x, target_size):
        """
        Upsample x to match target spatial size. 
        Reason of use:
        When using simple upsampling with scale factor 2, due to rounding issue with odd-sized inputs,
        the upsampled size may not exactly match the target size.
            500 → pool → 250 → pool → 125 → pool → 62 → pool → 31
            When we upsample 62 back up: 62 × 2 = 124 ≠ 125.
        This function specifically upsamples to the target size to avoid such issues.
        """
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

    def forward(self, maps, temp_series, metadata):
        # Encode temporal and metadata
        temporal_emb = self.temporal_encoder(temp_series)
        meta_emb = self.meta_encoder(metadata)
        
        # Encoder path (no embeddings, pure spatial)
        x0_0 = self.conv0_0(maps)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        # Start decoder with embedding fusion
        # Level 1
        H, W = x0_0.shape[2], x0_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_match(x1_0, (H, W)), emb_map], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        H, W = x1_0.shape[2], x1_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_match(x2_0, (H, W)), emb_map], 1))
        
        H, W = x0_0.shape[2], x0_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample_match(x1_1, (H, W)), emb_map], 1))

        # Level 2
        x3_0 = self.conv3_0(self.pool(x2_0))
        H, W = x2_0.shape[2], x2_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_match(x3_0, (H, W)), emb_map], 1))
        
        H, W = x1_0.shape[2], x1_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample_match(x2_1, (H, W)), emb_map], 1))
        
        H, W = x0_0.shape[2], x0_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample_match(x1_2, (H, W)), emb_map], 1))

        # Level 3 (deepest)
        x4_0 = self.conv4_0(self.pool(x3_0))
        H, W = x3_0.shape[2], x3_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_match(x4_0, (H, W)), emb_map], 1))
        
        H, W = x2_0.shape[2], x2_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample_match(x3_1, (H, W)), emb_map], 1))
        
        H, W = x1_0.shape[2], x1_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample_match(x2_2, (H, W)), emb_map], 1))
        
        H, W = x0_0.shape[2], x0_0.shape[3]
        emb_map = self._broadcast_embeddings(temporal_emb, meta_emb, H, W)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_match(x1_3, (H, W)), emb_map], 1))

        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            # Apply activation functions per channel
            if output.shape[1] == 2:
                output_ndvi = torch.tanh(output[:, 0:1, :, :])
                output_temp = output[:, 1:2, :, :]
                return torch.cat([output_ndvi, output_temp], dim=1)
            return output

class UrbanPredictor_unet(nn.Module):
    def __init__(self, spatial_channels, seq_len, temporal_dim, meta_features, 
                 meta_dim, lstm_dim, out_channels, nb_filter=None, 
                 temporal_embeddings=True,
                 metadata_embeddings=True
                 ):
        super().__init__()
        logger.info(f'UrbanPredictor_unet initialized with temporal_embeddings={temporal_embeddings}, metadata_embeddings={metadata_embeddings}')
        print(f'UrbanPredictor_unet initialized with temporal_embeddings={temporal_embeddings}, metadata_embeddings={metadata_embeddings}')

        
        if nb_filter is None:
            nb_filter = [32, 64, 128, 256, 512]
        
        self.temporal_dim = temporal_dim
        self.meta_dim = meta_dim
        self.temporal_embeddings = temporal_embeddings
        self.metadata_embeddings = metadata_embeddings
        
        # Temporal and metadata encoders
        self.temporal_encoder = TemporalEncoder(seq_len, hidden_dim=lstm_dim, out_dim=temporal_dim)
        self.meta_encoder = MetadataEncoder(meta_features, meta_dim)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder path
        self.conv0_0 = VGGBlock(spatial_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        
        # Bottleneck - optionally fuse embeddings
        bottleneck_in = nb_filter[3]
        if self.temporal_embeddings:
            bottleneck_in += temporal_dim
        if self.metadata_embeddings:
            bottleneck_in += meta_dim
        self.conv4_0 = VGGBlock(bottleneck_in, nb_filter[4], nb_filter[4])
        
        # Decoder path
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def _upsample_match(self, source, target):
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=True)
        return source
    
    def fuse_embeddings(self, spatial_feat, temporal_emb, meta_emb):
        B, C, H, W = spatial_feat.shape
        
        to_cat = [spatial_feat]
        if temporal_emb is not None:
            temporal_map = temporal_emb[:, :, None, None].expand(B, self.temporal_dim, H, W)
            to_cat.append(temporal_map)
        if meta_emb is not None:
            meta_map = meta_emb[:, :, None, None].expand(B, self.meta_dim, H, W)
            to_cat.append(meta_map)
        
        return torch.cat(to_cat, dim=1)

    def forward(self, maps, temp_series, metadata):
        # Encode temporal and metadata information
        temporal_emb = self.temporal_encoder(temp_series) if self.temporal_embeddings else None
        meta_emb = self.meta_encoder(metadata) if self.metadata_embeddings else None
        
        # Encoder path
        x0_0 = self.conv0_0(maps)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        
        # Bottleneck
        x4_0 = self.pool(x3_0)
        if self.temporal_embeddings or self.metadata_embeddings:
            x4_0 = self.fuse_embeddings(x4_0, temporal_emb, meta_emb)
        x4_0 = self.conv4_0(x4_0)
        
        # Decoder path
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_match(self.up(x4_0), x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_match(self.up(x3_1), x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_match(self.up(x2_1), x1_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_match(self.up(x1_1), x0_0)], 1))
        
        output = self.final(x0_1)

        # Apply activation functions per channel
        if output.shape[1] == 2:
            output_ndvi = torch.tanh(output[:, 0:1, :, :])
            output_temp = output[:, 1:2, :, :]
            return torch.cat([output_ndvi, output_temp], dim=1)

        return output


class UrbanPredictor(nn.Module):
    def __init__(self, model_type, spatial_channels, seq_len, temporal_dim, meta_features, meta_dim, lstm_dim, out_channels, 
                 base_filters=64, deep_supervision=False, **kwargs):
        super().__init__()

        if model_type == 'unet++':
            self.model = UrbanPredictor_unetpp(
                spatial_channels=spatial_channels,
                seq_len=seq_len,
                temporal_dim=temporal_dim,
                meta_features=meta_features,
                meta_dim=meta_dim,
                lstm_dim=lstm_dim,
                out_channels=out_channels,
                base_filters=base_filters,
                deep_supervision=deep_supervision,
                **kwargs
            )
        elif model_type == 'unet':
            self.model = UrbanPredictor_unet(
                spatial_channels=spatial_channels,
                seq_len=seq_len,
                temporal_dim=temporal_dim,
                meta_features=meta_features,
                meta_dim=meta_dim,
                lstm_dim=lstm_dim,
                out_channels=out_channels,
                nb_filter=[base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16],
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, maps, temp_series, metadata):
        return self.model(maps, temp_series, metadata)
