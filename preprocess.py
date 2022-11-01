import tgt
import argparse
import pathlib
import os
import json
import tqdm
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="path to dataset")
    parser.add_argument("-o", "--output", type=str, help="path to output json file")
    return parser.parse_args()


def parse_textgrid(filename):
    textgrid = tgt.read_textgrid(filename)
    intervals = textgrid.tiers[0]
    phonemes = []
    for interval in intervals:
        phonemes.append(
            (interval.text, interval.start_time, interval.end_time, interval.duration())
        )
    return phonemes


def generate_json_file(filename, obj):
    with open(filename, "w") as fout:
        json.dump(obj, fout)


def main():
    args = get_args()
    dataset_path = pathlib.Path(args.dir)
    phone_label_path = dataset_path / "PhoneLabeling"
    wav_path = dataset_path / "Wave.48k"
    samples = []
    all_phns = []
    for root, dirs, names in os.walk(phone_label_path):
        for name in tqdm.tqdm(names):
            textgrid_filename = phone_label_path / name
            wav_filename = wav_path / f"{textgrid_filename.stem}.wav"
            if wav_filename.exists():
                phonemes = parse_textgrid(textgrid_filename)
                samples.append({"path": str(wav_filename), "phonemes": phonemes, 'item_name':textgrid_filename.stem})
                all_phns.extend([x[0] for x in phonemes])
    phn_set = set(all_phns)
    random.shuffle(samples)
    num_training_samples = int(len(samples) * 0.9)
    num_test_samples = int(len(samples) * 0.05)
    training_samples = samples[:num_training_samples]
    valid_samples = samples[
        num_training_samples : num_training_samples + num_test_samples
    ]
    test_samples = samples[num_training_samples + num_test_samples :]

    training_sample_file = f"{args.output}.train"
    valid_sample_file = f"{args.output}.valid"
    test_sample_file = f"{args.output}.test"
    generate_json_file(training_sample_file, training_samples)
    generate_json_file(valid_sample_file, valid_samples)
    generate_json_file(test_sample_file, test_samples)
    generate_json_file(f'{args.output}.phnset', list(phn_set))


if __name__ == "__main__":
    main()
