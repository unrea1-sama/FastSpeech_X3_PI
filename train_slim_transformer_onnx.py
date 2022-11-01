# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import load_config

from slim_transformer_onnx import SlimTransformerONNX, IndexedDataset, collateFn
from utils import plot_tensor, save_plot

import argparse
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="config file path")
    args = parser.parse_args()
    configs = load_config(args.c)
    return configs


def build_model(n_vocab_size, configs):
    return SlimTransformerONNX(
        n_vocab_size,
        configs["enc_channels"],
        configs["enc_filter_channels"],
        configs["dp_filter_channels"],
        configs["enc_heads"],
        configs['enc_layers'],
        configs["enc_kernel"],
        configs["enc_dropout"],
        configs["dec_channels"],
        configs["dec_filter_channels"],
        configs["dec_heads"],
        configs["dec_layers"],
        configs["dec_kernel"],
        configs["dec_dropout"],
        configs["mel_dim"],
        configs["enc_max_length"],
        configs["dec_max_length"],
    )


if __name__ == "__main__":
    configs = get_args()
    torch.manual_seed(configs["random_seed"])
    np.random.seed(configs["random_seed"])
    log_dir = pathlib.Path(configs["log_dir"])
    print("Initializing data loaders...")
    cfn = collateFn(configs['enc_max_length'],configs['dec_max_length'])
    train_dataset = IndexedDataset(configs["train_dataset_path"],
                                   configs['phnset'], configs['sample_rate'],
                                   configs['window_size'], configs['hop_size'],
                                   configs['mel_dim'])
    val_dataset = IndexedDataset(configs["train_dataset_path"],
                                   configs['phnset'], configs['sample_rate'],
                                   configs['window_size'], configs['hop_size'],
                                   configs['mel_dim'])
    train_loader = DataLoader(train_dataset,
                              batch_size=configs["batch_size"],
                              collate_fn=cfn)
    val_loader = DataLoader(val_dataset,
                            batch_size=configs["batch_size"],
                            shuffle=True,
                            collate_fn=cfn)
    print("Initializing model...")
    model = build_model(train_dataset.get_vocab_size(),configs)
    print(
        f"Number of encoder + duration predictor parameters: {model.encoder.nparams}"
    )
    print(f"Number of decoder parameters: {model.decoder.nparams}")
    print(f"Total parameters: {model.nparams}")
    if configs["checkpoint"]:
        print(f'loading {configs["checkpoint"]}')
        model.load_state_dict(
            torch.load(configs["checkpoint"], map_location="cpu"))
    model = model.cuda()

    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=configs["learning_rate"])

    print("Initializing logger...")
    logger = SummaryWriter(log_dir=log_dir)

    ckpt_dir = log_dir / "ckpt"
    pic_dir = log_dir / "pic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)
    print("Start training...")
    iteration = (configs["start_from"] - 1) * (len(train_dataset) //
                                               configs["batch_size"])
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    for epoch in range(configs["start_from"],
                       configs["start_from"] + configs["epoch"] + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(train_loader,
                  total=len(train_dataset) //
                  configs["batch_size"]) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = (
                    batch["x"].unsqueeze(1).unsqueeze(3).cuda(),
                    batch["x_lengths"].reshape(-1, 1, 1, 1).cuda(),
                )
                y, y_lengths = batch["y"].unsqueeze(
                    2).cuda(), batch["y_lengths"]
                duration = batch["duration"].unsqueeze(1).unsqueeze(1).cuda()
                logw, x_mask, mel, mel_mask, _ = model(x, x_lengths, duration)
                mel_loss = l1_loss(mel * mel_mask, y * mel_mask)
                dur_loss = l2_loss(logw * x_mask,
                                   torch.log(duration + 1) * x_mask)
                loss = sum([dur_loss, mel_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), max_norm=1)
                optimizer.step()

                logger.add_scalar("training/duration_loss",
                                  dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/mel_loss",
                                  mel_loss.item(),
                                  global_step=iteration)
                logger.add_scalar("training/encoder_grad_norm",
                                  enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar("training/decoder_grad_norm",
                                  dec_grad_norm,
                                  global_step=iteration)

                if batch_idx % 5 == 0:
                    msg = f"slimtransformer Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, mel_loss: {mel_loss.item()}"
                    progress_bar.set_description(msg)

                iteration += 1
                break
        model.eval()
        with torch.no_grad():
            all_dur_loss = []
            all_mel_loss = []
            for _, item in enumerate(val_loader):
                x, x_lengths = (
                    batch["x"].unsqueeze(1).unsqueeze(3).cuda(),
                    batch["x_lengths"].reshape(-1, 1, 1, 1).cuda(),
                )
                y, y_lengths = batch["y"].unsqueeze(
                    2).cuda(), batch["y_lengths"]
                duration = batch["duration"].unsqueeze(1).unsqueeze(1).cuda()
                logw, x_mask, mel, mel_mask, _ = model(x, x_lengths, duration)
                mel_loss = l1_loss(mel * mel_mask, y * mel_mask)
                dur_loss = l2_loss(logw * x_mask,
                                   torch.log(duration + 1) * x_mask)
                all_dur_loss.append(dur_loss)
                all_mel_loss.append(mel_loss)
            average_dur_loss = sum(all_dur_loss) / len(all_dur_loss)
            average_mel_loss = sum(all_mel_loss) / len(all_mel_loss)
            logger.add_scalar("val/duration_loss",
                              average_dur_loss,
                              global_step=epoch)
            logger.add_scalar("val/mel_loss",
                              average_mel_loss,
                              global_step=epoch)

            idx = random.randrange(0, mel.shape[0])
            length = y_lengths[idx]
            mel_l = mel[idx].cpu().squeeze(1)
            y = y[idx].cpu().squeeze(1)
            logger.add_image(
                "image/generated_mel",
                plot_tensor(mel_l[:,:y_lengths]),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/ground_truth",
                plot_tensor(y[:,:y_lengths]),
                global_step=epoch,
                dataformats="HWC",
            )
            save_plot(mel_l, pic_dir / f"generated_mel_{epoch}.png")
            save_plot(y, pic_dir / f"ground_truth_{epoch}.png")

        ckpt = model.state_dict()
        torch.save(ckpt, f=ckpt_dir / f"slimtransformer_{epoch}.pt")
        if iteration >= configs["max_step"]:
            exit()