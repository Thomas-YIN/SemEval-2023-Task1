from torch import nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import einops
import torch

class TransformerFuser(nn.Module):

    def __init__(self, num_layers=2):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer_encoders = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, prompt, retrieved_images, candidate_images, clip, tokenizer, device):

        b = len(prompt)
        clip.to(device)
        clip.eval()

        retrieved_images = retrieved_images.reshape(b*3, 3, 224, 224)
        candidate_images = candidate_images.reshape(b*10, 3, 224, 224)
        with torch.no_grad():
            text_feat = clip.encode_text(tokenizer(prompt).to(device)).reshape(b, 1, 512) # b x 512
            retrieved_images_feat = clip.encode_image(retrieved_images) # b*k x 512
            candidate_images_feat = clip.encode_image(candidate_images) # b*10 x 512

        retrieved_images_feat = retrieved_images_feat.reshape(b, 3, 512)
        fused_feat = self.transformer_encoders(torch.cat((text_feat, retrieved_images_feat), dim=1)).sum(dim=1) # b x 512

        fused_feat = einops.repeat(fused_feat, 'm n -> m k n', k=10) # b x 10 x 512
        candidate_images_feat = candidate_images_feat.reshape(b, 10, 512) # b x 10 x 512

        logits = torch.einsum('ijk,ijk->ij', F.normalize(fused_feat, dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)

        return logits # b x 10
    
class AverageFuser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prompt, retrieved_images, candidate_images, clip, tokenizer, device):
        b = len(prompt)
        clip.to(device)
        clip.eval()

        retrieved_images = retrieved_images.reshape(b*3, 3, 224, 224)
        candidate_images = candidate_images.reshape(b*10, 3, 224, 224)
        with torch.no_grad():
            text_feat = clip.encode_text(tokenizer(prompt).to(device)) # b x 512
            retrieved_images_feat = clip.encode_image(retrieved_images).reshape(b, 3, 512) # b*k x 512
            candidate_images_feat = clip.encode_image(candidate_images).reshape(b, 10, 512) # b*10 x 512

        r_1 = retrieved_images_feat[:, 0, :]
        r_2 = retrieved_images_feat[:, 1, :]
        r_3 = retrieved_images_feat[:, 2, :]

        logits_0 = torch.einsum('ijk,ijk->ij', F.normalize(einops.repeat(text_feat, 'm n -> m k n', k=10), dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)
        logits_1 = torch.einsum('ijk,ijk->ij', F.normalize(einops.repeat(r_1, 'm n -> m k n', k=10), dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)
        logits_2 = torch.einsum('ijk,ijk->ij', F.normalize(einops.repeat(r_2, 'm n -> m k n', k=10), dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)
        logits_3 = torch.einsum('ijk,ijk->ij', F.normalize(einops.repeat(r_3, 'm n -> m k n', k=10), dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)

        return (logits_0+logits_1+logits_2+logits_3)/4

class MLPFuser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPFuser, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, prompt, retrieved_images, candidate_images, clip, tokenizer, device):
        b = len(prompt)
        clip.to(device)
        clip.eval()

        retrieved_images = retrieved_images.reshape(b*3, 3, 224, 224)
        candidate_images = candidate_images.reshape(b*10, 3, 224, 224)
        with torch.no_grad():
            text_feat = clip.encode_text(tokenizer(prompt).to(device)) # b x 512
            retrieved_images_feat = clip.encode_image(retrieved_images).reshape(b, 3, 512) # b*k x 512
            candidate_images_feat = clip.encode_image(candidate_images).reshape(b, 10, 512) # b*10 x 512

        x = torch.cat(text_feat, retrieved_images_feat)
        
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        fused_feat = self.fc3(y)
        
        fused_feat = einops.repeat(fused_feat, 'm n -> m k n', k=10) # b x 10 x 512
        candidate_images_feat = candidate_images_feat.reshape(b, 10, 512) # b x 10 x 512
        logits = torch.einsum('ijk,ijk->ij', F.normalize(fused_feat, dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)

        return logits # b x 10