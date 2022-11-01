# FastSpeech for X3PI
Requirements
```
python3 -m pip install torch torchaudio tgt tensorboard matplotlib tqdm librosa
```

## Training
### Dataset Preparation

Download and extract [BZNSYP](https://www.data-baker.com/data/index/TNtts).
Run preprocess.py and generate json file for training, validation and test dataset.
```
python3 preprocess.py -d PATH_TO_DATASET -o dataset.json
```

### Train Model

Run train_slim_transformer_onnx.py
```
python3 train_slim_transformer_onnx.py -c config.yaml
```
Training log and checkpoint will be saved in `log/bznsyp` and `log/bznsyp/ckpt`, respectively.
To modify training and model configuration, edit `config.yaml`.
Note that `enc_max_length` and `dec_max_length` in `config.yaml` determine the maximum input phoneme sequence length and maximum mel-spectrogram length, respectively.

## Export Model

Set `checkpoint` in `config.yaml` to the path of checkpoint you want to export, e.g. `log/bznsyp/ckpt/slimtransformer_1000.pt`.
Run slim_transformer_onnx_export.py
```
python3 slim_transformer_onnx_export.py -e log/export -c config.yaml
```
Then, ONNX model will be saved in `log/export/slim_transformer.onnx` by default.
Also, dataset for calibration will be saved in `log/export/calibration`.

## Convert Model

First, follow [Horizon_ai_toolchain_user_guide](https://developer.horizon.ai/api/v1/fileData/doc/ddk_doc/navigation/ai_toolchain/docs_cn/horizon_ai_toolchain_user_guide/introduction.html) to setup environment for deploying model on X3PI.

Then, run
```
hb_mapper checker --model-type onnx --march bernoulli2 --model log/export/slim_transformer.onnx --input-shape x 1x1x128x1 --input-shape x_length 1x1x1x1
```
to check exported model. Note that input shape for `x` should be the same as `enc_max_length` in `config.yaml`.

After the model has passed check, run
```
cd board
hb_mapper makertbin --config slim_transformer_onnx.yaml --model-type onnx
```
to convert model into an executable model for X3PI.
Converted model will be saved in `log/export/converted`.

## Inference

Copy `board/slim_transformer_onnx_board.py`, `log/export/converted/slim_transformer.bin` and `dataset.json.phnset` into X3PI.
Run
```
python3 slim_transformer_onnx_board.py --model_path PATH_TO_slim_transformer.bin --str 'PINGYIN SEQUENCE FOR SYNTHESIZING' --dict dataset.json.phnset
```
to perform inference on X3PI.
An example command is
```
python3 slim_transformer_onnx_board.py --model_path slim_transformer.bin --str 'ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1' --dict dataset.json.phnset
```
and it will synthesizing corresponding mel-spectrogram of 'ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1'.
Generated mel-spectrogram will be saved as a PNG with timestamp as its file name.


