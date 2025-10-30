import streamlit as st
import argparse
from PIL import Image
import numpy as np
import random
import time
from tqdm import tqdm
import torch
import torchvision.utils as vutils
import re
import os
import pickle
import sys
import os.path as osp
from code.lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_netG
from code.lib.utils import truncated_noise, prepare_sample_data
from code.lib.perpare import prepare_models

def get_tokenizer():
    def tokenizer(text):
        return re.findall(r'\w+', text)
    return tokenizer

def sample_example1(text):
    tokenizer = get_tokenizer()
    tokens = tokenizer(text)
    return tokens

def tokenize(wordtoix, sentences):
    tokenizer = get_tokenizer()
    captions = []
    cap_lens = []
    new_sent = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokens = sample_example1(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue
        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
        new_sent.append(sent)    
    return captions, cap_lens, new_sent

def sample_example(wordtoix, _netG, _text_encoder, args):
    batch_size, device = args.imgs_per_sent, args.device
    text_filepath, img_save_path = args.example_captions, args.samples_save_dir
    truncation, trunc_rate = args.truncation, args.trunc_rate
    z_dim = args.z_dim
    captions, cap_lens, _ = tokenize(wordtoix, text_filepath)
    sent_embs, _  = prepare_sample_data(captions, cap_lens, _text_encoder, device)
    caption_num = sent_embs.size(0)
    if truncation:
        noise = truncated_noise(batch_size, z_dim, trunc_rate)
        noise = torch.tensor(noise, dtype=torch.float).to(device)
    else:
        noise = torch.randn(batch_size, z_dim).to(device)
    with torch.no_grad():
        fakes = []        
        for i in tqdm(range(caption_num)):
            sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
            fakes = _netG(noise, sent_emb)
            img_name = osp.join(img_save_path,'Sent%03d.png' % (i+1))
            torch.cuda.empty_cache()
    return fakes.data

def parse_args():
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./code/cfg/bird.yml', help='optional config file')
    parser.add_argument('--imgs_per_sent', type=int, default=1, help='the number of images per sentence')
    parser.add_argument('--imsize', type=int, default=256, help='image size')
    parser.add_argument('--cuda', type=bool, default=False, help='if use GPU')
    parser.add_argument('--train', type=bool, default=False, help='if training')
    parser.add_argument('--multi_gpus', type=bool, default=False, help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true', default=True, help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args

def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix

def main(args):
    st.title("Text to Image DFGAN Demo")
    st.write('\n\n')

    caption = st.text_input("Enter The Caption")
    n_copies = st.slider('Number of Generated Images', min_value=1, max_value=12, value=6, step=1)

    if st.button('Generate Image'):
        my_bar = st.progress(0)
        placeholder = st.empty()
        placeholder.info('Loading Model', icon="ℹ️")

        time_stamp = get_time_stamp()
        args.example_captions = list(caption.split('\n'))
        args.imgs_per_sent = n_copies
        args.samples_save_dir = osp.join(args.samples_save_dir, time_stamp)

        if (args.multi_gpus == True) and (get_rank() != 0):
            None
        else:
            mkdir_p(args.samples_save_dir)

        for percent_complete in range(15, 30):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        pickle_path = "./code/data/birds/captions_DAMSM.pickle"
        args.vocab_size, wordtoix = build_word_dict(pickle_path)

        _, _text_encoder, _netG, _, _ = prepare_models(args)
        model_path = "./code/saved_models/bird/pretrained/state_epoch_080_bird.pth"
        _netG = load_netG(_netG, model_path, args.multi_gpus, train=False)
        _netG.eval()

        placeholder.info('Providing Input', icon="ℹ️")
        for percent_complete in range(40, 60):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        if (args.multi_gpus == True) and (get_rank() != 0):
            None
        else:
            print('Load %s for NetG' % (args.checkpoint))
            print("************ Start sampling ************")

        start_t = time.time()
        image = sample_example(wordtoix, _netG, _text_encoder, args)
        end_t = time.time()

        if (args.multi_gpus == True) and (get_rank() != 0):
            None
        else:
            print('*' * 40)
            print('Sampling done, %.2fs cost, saved to %s' % (end_t-start_t, args.samples_save_dir))
            print('*' * 40)

        placeholder.info('Generating Images', icon="ℹ️")
        for percent_complete in range(70, 100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        placeholder.empty()
        my_bar.empty()

        st.write("#### Output Image")
        grid = vutils.make_grid(image, nrow=4, normalize=True, scale_each=True)
        image = grid.permute(1, 2, 0).detach().cpu().numpy()
        st.image(image, width=700)

if __name__ == "__main__":
    st.markdown("Celeba")
    st.sidebar.markdown("Celeba")
    args = merge_args_yaml(parse_args())
    
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)
