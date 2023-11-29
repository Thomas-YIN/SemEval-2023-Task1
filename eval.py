import numpy as np
import torch
import einops
import torch.nn.functional as F

def MRR(logits, gold_indices):
    # expect logits to be numpy array of shape [batch, 10], gold_indices [batch,]
    n = logits.shape[0]
    pred_indices = torch.argsort(logits, dim=1, descending=True)
    gold_indices = gold_indices.reshape(n, -1).repeat(1, 10)
    ranks = (pred_indices == gold_indices).nonzero(as_tuple=False)[:,1] + 1
    mrr = (1/ranks).sum() / n

    return mrr

def hit_rate(logits, gold_indices):
    pred = logits.argmax(axis=1).numpy()
    gold_indices = gold_indices.cpu().numpy()
    return np.mean(pred == gold_indices)

def evaluate(dataloader, clip, tokenizer, device, model=None):
    hit = []
    mrr = []
    for i, (prompt, retrieved_images, candidate_images, gold_index) in enumerate(dataloader):

        b = len(prompt)

        if model is not None:
            logits = model(prompt, retrieved_images.to(device), candidate_images.to(device))
        else:
            # CLIP baseline
            with torch.no_grad():
                text_feat = clip.encode_text(tokenizer(prompt).to(device)) # b x 512
                candidate_images_feat = clip.encode_image(candidate_images.reshape(-1, 3, 224, 224).to(device)).reshape(-1, 10, 512) # b*10 x 512
            text_feat = einops.repeat(text_feat, 'm n -> m k n', k=10) # b x 10 x 512
            logits = torch.einsum('ijk,ijk->ij', F.normalize(text_feat, dim=(1, 2)), F.normalize(candidate_images_feat, dim=(1, 2))).softmax(dim=-1)
        hit.append(hit_rate(logits.detach().cpu(), gold_index))
        mrr.append(MRR(logits.detach().cpu(), gold_index))
    
    print("hit@1:", np.mean(hit))
    print("mrr:", np.mean(mrr))
