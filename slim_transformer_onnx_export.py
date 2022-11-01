# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib
import onnxruntime
import matplotlib.pyplot as plt
from config import load_config
from train_slim_transformer_onnx import build_model, IndexedDataset, collateFn
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--export_dir", type=str)
    parser.add_argument("--sample_count", type=int, default=200)
    parser.add_argument("-c", type=str, help="config file path")
    return parser.parse_args()


def save_plot(tensors, titles, text, savepath):
    plt.style.use("default")
    xlim = max([t.shape[1] for t in tensors])
    fig, axs = plt.subplots(nrows=len(tensors),
                            ncols=1,
                            figsize=(12, 9),
                            constrained_layout=True)
    for i in range(len(tensors)):
        im = axs[i].imshow(tensors[i],
                           aspect="auto",
                           origin="lower",
                           interpolation="none")
        plt.colorbar(im, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlim([0, xlim])
    fig.canvas.draw()
    plt.suptitle(text)
    plt.savefig(savepath, dpi=300)
    plt.close()


if __name__ == "__main__":
    args = get_args()
    configs = load_config(args.c)
    root_export_dir = pathlib.Path(args.export_dir)

    torch.manual_seed(configs["random_seed"])
    np.random.seed(configs["random_seed"])
    cfn = collateFn(configs["enc_max_length"], configs["dec_max_length"])
    print("Initializing data loaders...")
    test_dataset = IndexedDataset(configs["test_dataset_path"],
                                  configs['phnset'], configs['sample_rate'],
                                  configs['window_size'], configs['hop_size'],
                                  configs['mel_dim'])
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfn)
    print("Initializing model...")
    model = build_model(test_dataset.get_vocab_size(), configs)
    print("Number of encoder + duration predictor parameters: {}".format(
        model.encoder.nparams))
    print("Number of decoder parameters: {}".format(model.decoder.nparams))
    print("Total parameters: {}".format(model.nparams))

    print("loading ", configs["checkpoint"])
    model.load_state_dict(torch.load(configs["checkpoint"],
                                     map_location="cpu"))

    model.eval()

    export_file = root_export_dir / "slim_transformer.onnx"
    print("Start onnx export slim_transformer, dataset: {}".format(
        configs["test_dataset_path"]))
    inference_result_export_dir = root_export_dir / "export_mels"
    x_export_dir = root_export_dir / "calibration" / "x"
    x_length_dir = root_export_dir / "calibration" / "x_length"

    inference_result_export_dir.mkdir(parents=True, exist_ok=True)
    x_export_dir.mkdir(parents=True, exist_ok=True)
    x_length_dir.mkdir(parents=True, exist_ok=True)

    for i, item in tqdm(enumerate(test_loader)):
        x, x_length, txt = (
            item["x"].unsqueeze(1).unsqueeze(3),
            item["x_lengths"].reshape(1, 1, 1, 1),
            item["phns"][0],
        )
        x = x.float()
        x_length = x_length.float()
        logw, x_mask, mel, mel_mask, mel_length_pytorch = model(x, x_length)
        if i == 0:
            torch.onnx.export(
                model,
                (x, x_length),
                str(export_file),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["x", "x_length"],
                output_names=["logw", "x_mask", "mel", "mel_mask", "mel_length"],
                verbose=True,
            )
            print("Export onnx for model: slimtransformer to: {}".format(
                args.export_dir))
        if i >= args.sample_count:
            break
        ort_session = onnxruntime.InferenceSession(
            str(export_file), providers=["CPUExecutionProvider"])

        ort_inputs = {"x": x.numpy(), "x_length": x_length.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        logw_onnx, x_mask_onnx, mel_onnx, mel_mask_onnx, mel_length_onnx = ort_outs
        mel_length_onnx = int(mel_length_onnx.flatten())
        mel_length_pytorch = int(mel_length_pytorch.flatten())
        save_plot(
            [
                mel_onnx[0].squeeze(1)[:, :mel_length_onnx],
                mel[0].squeeze(1).detach().numpy()[:, :mel_length_pytorch],
            ],
            [
                "onnx",
                "pytorch",
            ],
            txt,
            inference_result_export_dir / f"{i}.png",
        )
        x = x.numpy().astype(np.float32)
        x_length = x_length.numpy().astype(np.float32)
        x.tofile(x_export_dir / f"x_{i}.bin")
        x_length.tofile(x_length_dir / f"x_length_{i}.bin")
