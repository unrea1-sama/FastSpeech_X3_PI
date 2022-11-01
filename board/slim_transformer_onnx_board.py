#!/usr/bin/env python3
# Copyright (c) 2022 Tsinghua University(Jie Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hobot_dnn import pyeasy_dnn as dnn
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import re
import argparse
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--str", type=str)
    parser.add_argument("--dict", type=str)
    return parser.parse_args()


def save_plot(tensors, titles, text, savepath):
    plt.style.use("default")
    xlim = max([t.shape[1] for t in tensors])
    fig, axs = plt.subplots(
        nrows=len(tensors), ncols=1, figsize=(12, 9), constrained_layout=True
    )
    if len(tensors) > 1:
        for i in range(len(tensors)):
            im = axs[i].imshow(
                tensors[i], aspect="auto", origin="lower", interpolation="none"
            )
            plt.colorbar(im, ax=axs[i])
            axs[i].set_title(titles[i])
            axs[i].set_xlim([0, xlim])
    else:
        im = axs.imshow(tensors[0], aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=axs)
        axs.set_title(titles[0])
        axs.set_xlim([0, xlim])
    fig.canvas.draw()
    plt.suptitle(text)
    plt.savefig(savepath, dpi=300)
    plt.close()


INITIALS = [
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "zh",
    "ch",
    "sh",
    "r",
    "z",
    "c",
    "s",
    "j",
    "q",
    "x",
]

FINALS = [
    "a",
    "ai",
    "ao",
    "an",
    "ang",
    "e",
    "er",
    "ei",
    "en",
    "eng",
    "o",
    "ou",
    "ong",
    "ii",
    "iii",
    "i",
    "ia",
    "iao",
    "ian",
    "iang",
    "ie",
    "io",
    "iou",
    "iong",
    "in",
    "ing",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uo",
    "uen",
    "ueng",
    "v",
    "ve",
    "van",
    "vn",
]

SPECIALS = ["sil", "sp"]


def rule(C, V, R, T):
    # Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # adopted from PaddleSpeech(https://github.com/PaddlePaddle/PaddleSpeech)
    """Generate a syllable given the initial, the final, erhua indicator,
    and tone. Orthographical rules for pinyin are
    applied. (special case for y, w, ui, un, iu)

    Note that in this system, 'ü' is alway written as 'v' when appeared in
    phoneme, but converted to 'u' in syllables when certain conditions
    are satisfied.

    'i' is distinguished when appeared in phonemes, and separated into 3
    categories, 'i', 'ii' and 'iii'.

    Erhua is is possibly applied to every finals, except for finals that
    already ends with 'r'.

    When a syllable is impossible or does not have any characters with this
    pronunciation, return None to filter it out.
    """

    # 不可拼的音节, ii 只能和 z, c, s 拼
    if V in ["ii"] and (C not in ["z", "c", "s"]):
        return None
    # iii 只能和 zh, ch, sh, r 拼
    if V in ["iii"] and (C not in ["zh", "ch", "sh", "r"]):
        return None

    # 齐齿呼或者撮口呼不能和 f, g, k, h, zh, ch, sh, r, z, c, s
    if (
        (V not in ["ii", "iii"])
        and V[0] in ["i", "v"]
        and (C in ["f", "g", "k", "h", "zh", "ch", "sh", "r", "z", "c", "s"])
    ):
        return None

    # 撮口呼只能和 j, q, x l, n 拼
    if V.startswith("v"):
        # v, ve 只能和 j ,q , x, n, l 拼
        if V in ["v", "ve"]:
            if C not in ["j", "q", "x", "n", "l", ""]:
                return None
        # 其他只能和 j, q, x 拼
        else:
            if C not in ["j", "q", "x", ""]:
                return None

    # j, q, x 只能和齐齿呼或者撮口呼拼
    if (C in ["j", "q", "x"]) and not ((V not in ["ii", "iii"]) and V[0] in ["i", "v"]):
        return None

    # b, p ,m, f 不能和合口呼拼，除了 u 之外
    # bm p, m, f 不能和撮口呼拼
    if (C in ["b", "p", "m", "f"]) and (
        (V[0] in ["u", "v"] and V != "u") or V == "ong"
    ):
        return None

    # ua, uai, uang 不能和 d, t, n, l, r, z, c, s 拼
    if V in ["ua", "uai", "uang"] and C in ["d", "t", "n", "l", "r", "z", "c", "s"]:
        return None

    # sh 和 ong 不能拼
    if V == "ong" and C in ["sh"]:
        return None

    # o 和 gkh, zh ch sh r z c s 不能拼
    if V == "o" and C in [
        "d",
        "t",
        "n",
        "g",
        "k",
        "h",
        "zh",
        "ch",
        "sh",
        "r",
        "z",
        "c",
        "s",
    ]:
        return None

    # ueng 只是 weng 这个 ad-hoc 其他情况下都是 ong
    if V == "ueng" and C != "":
        return

    # 非儿化的 er 只能单独存在
    if V == "er" and C != "":
        return None

    if C == "":
        if V in ["i", "in", "ing"]:
            C = "y"
        elif V == "u":
            C = "w"
        elif V.startswith("i") and V not in ["ii", "iii"]:
            C = "y"
            V = V[1:]
        elif V.startswith("u"):
            C = "w"
            V = V[1:]
        elif V.startswith("v"):
            C = "yu"
            V = V[1:]
    else:
        if C in ["j", "q", "x"]:
            if V.startswith("v"):
                V = re.sub("v", "u", V)
        if V == "iou":
            V = "iu"
        elif V == "uei":
            V = "ui"
        elif V == "uen":
            V = "un"
    result = C + V

    # Filter  er 不能再儿化
    if result.endswith("r") and R == "r":
        return None

    # ii and iii, change back to i
    result = re.sub(r"i+", "i", result)

    result = result + R + T
    return result


def get_current_time():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Tokenizer:
    def __init__(self, phn2id) -> None:
        self.load_dict(phn2id)
        self.generate_lexicon(True, True)

    def load_dict(self, dict_path):
        with open(dict_path) as f:
            self.dict = {token: i+1 for i, token in enumerate(json.load(f))}

    def tokenize(self, seq, max_length=128):
        # transform pingyin to phonemes
        tokens = []
        for x in seq.strip().split(" "):
            if x in self.syllables:
                tokens.append(self.syllables[x])
            else:
                tokens.append(x)
        seq = " ".join(tokens)
        # convert phonemes to phoneme ids
        ids = [self.dict[x] for x in seq.split(" ")]
        length = [len(ids)]
        ids += [0] * (max_length - len(ids))
        return (
            np.array(ids, dtype=np.float32).reshape(1, 1, max_length, 1),
            np.array(length, dtype=np.float32).reshape(1, 1, 1, 1),
        )

    def generate_lexicon(self, with_tone=False, with_erhua=False):
        """Generate lexicon for Mandarin Chinese."""
        self.syllables = OrderedDict()

        for C in [""] + INITIALS:
            for V in FINALS:
                for R in [""] if not with_erhua else ["", "r"]:
                    for T in [""] if not with_tone else ["1", "2", "3", "4", "5"]:
                        result = rule(C, V, R, T)
                        if result:
                            self.syllables[result] = f"{C} {V}{R}{T}".strip()


def print_properties(tensor):
    print(
        f"tensor type: {tensor.properties.tensor_type}, data type: {tensor.properties.dtype}, "
        f"layout: {tensor.properties.layout}, shape: {tensor.properties.shape}"
    )


def print_model_info(model):
    for i in range(len(model.inputs)):
        print(f"input {i}, {model.inputs[i].name}: {print_properties(model.inputs[i])}")
    for i in range(len(model.outputs)):
        print(
            f"output {i}, {model.outputs[i].name}: {print_properties(model.outputs[i])}"
        )


if __name__ == "__main__":
    args = get_args()
    model = dnn.load(args.model_path)[0]
    tokenizer = Tokenizer(args.dict)
    print_model_info(model)

    print(f"synthesizing {args.str}")

    x, x_length = tokenizer.tokenize(args.str)

    # output: logw, x_mask, mel, mel_mask, mel_length
    output = model.forward([x, x_length])

    duration = np.ceil((np.exp(output[0].buffer) - 1)) * np.array(output[1].buffer)
    mel_length = np.sum(duration, dtype=int)
    print(f"predicted duration:{duration}, mel length:{mel_length}")

    mel = np.array(output[2].buffer)[0, :, 0,:mel_length]

    output_mel_name = f"{get_current_time()}.png"
    save_plot([mel], ["output"], args.str, output_mel_name)
    print(f"save melspectrogram to {output_mel_name}")
