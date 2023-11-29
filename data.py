import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from biencoder.models import *
from biencoder.biencoder import *
import torch

import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import open_clip
from torch.utils.data import Dataset, DataLoader

root = ''

def produce_prompt(biencoder_preds, contexts):
    prompts = []
    for i, pred in enumerate(biencoder_preds):

        word, gloss = pred

        prompt = "This is a picture of " + contexts[i]

        if gloss != "None":
            synset = wordnet.synset_from_sense_key(gloss)
            synonyms = [l.name() for l in synset.lemmas()]

            if not word in synonyms:
                prompt += ", also known as "
                for s in synonyms:
                    prompt += s+", "
            else:
                prompt += ", "

            prompt += "where " + word + " refers to " + synset.definition()
        prompts.append(prompt)
    return prompts

def prepare_prompts(mode='train'):
    if mode == 'train' or mode == 'valid':
        filename = root + 'prompts.txt'
    if mode == 'test':
        filename = root + 'test_prompts.txt'

    if not os.path.exists(filename):
        tokenizer = load_tokenizer('bert-base')
        biencoder = BiEncoderModel('bert-base')
        checkpoint = torch.load('biencoder.ckpt')
        biencoder.load_state_dict(checkpoint)

        data = load_data(mode)
        eval_data = preprocess_context(tokenizer, data)
        gloss_dict = load_and_preprocess_glosses(data, tokenizer, max_len=32)
        eval_preds = do_eval(eval_data, biencoder.cuda(), gloss_dict)

        contexts = []

        for d in data:
            context = ""
            for i in range(len(d)):
                if i == len(d)-1:
                    context += d[i][0]
                else:
                    context += d[i][0]+" "

            contexts.append(context)
        prompts = produce_prompt(eval_preds, contexts)
        with open(filename, 'w') as fp:
            for item in prompts:
                fp.write("%s\n" % item)
            print('Done')
    else:
        prompts = []
        with open(filename, 'r') as fp:
            for line in fp:
                x = line[:-1]
                prompts.append(x)

    if mode == 'train':
        return prompts[:12000]
    if mode == 'valid':
        return prompts[12000:]
    return prompts

def prepare_candidates(mode='train'):
    candidates = []
    # valid data is part of the training data
    if mode == 'train' or mode == 'valid':
        with open(root + 'data/train_v1/train.data.v1.txt', 'r') as fp:
            for line in fp:
                x = line[:-1].split("\t")[-10:]
                candidates.append(x)
        if mode == 'train':
            return candidates[:12000]
        else:
            return candidates[12000:]
    else:
        with open(root + 'data/test_v1/test.data.v1.txt', 'r') as fp:
            for line in fp:
                x = line[:-1].split("\t")[-10:]
                candidates.append(x)

        return candidates

def prepare_gold(mode='train'):
    gold = []

    if mode == 'train' or mode == 'valid':
        with open(root + 'data/train_v1/train.gold.v1.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]
                gold.append(x)
        if mode == 'train':
            return gold[:12000]
        else:
            return gold[12000:]
    else:
        with open(root + 'data/test_v1/test.gold.v1.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]
                gold.append(x)

        return gold

class ImageTextDataset(Dataset):
    def __init__(self, mode='train', preprocess=None):

        self.mode = mode
        self.preprocess = preprocess
        self.prompts = prepare_prompts(mode)
        self.phrases = []
        for p in self.prompts:
            phrase = (" ".join(p.split(" ")[5:7]).replace(",", "")).replace("/", "")
            self.phrases.append(phrase)

        self.candidates = prepare_candidates(mode)
        self.gold_labels = prepare_gold(mode)

        if mode == 'train' or mode == 'valid':
            self.candidate_dir = root + 'data/train_v1/train_images_v1/'
        else:
            self.candidate_dir = root + 'data/test_v1/test_images_v1/'

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):

        prompt = self.prompts[idx]

        retrieved = [str(self.phrases[idx])+str(i)+".jpg" for i in range(3)]
        retrieved_images = []
        for r in retrieved:
            retrieved_images.append(self.preprocess(Image.open(root + "retrieved_images/" + r)))
        retrieved_images = torch.stack(retrieved_images) # tensors

        candidates = self.candidates[idx]
        candidate_images = []
        for c in candidates:
            candidate_images.append(self.preprocess(Image.open(self.candidate_dir + c)))
        candidate_images = torch.stack(candidate_images) # tensors

        gold_label = self.gold_labels[idx]
        gold_index = candidates.index(gold_label)

        return prompt, retrieved_images, candidate_images, gold_index