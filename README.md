# CIFAR-10 Swin Transformer Training

PyTorch Lightning の `LightningCLI` を使って、`torchvision.datasets.CIFAR10` を PyTorch 実装の Swin Transformer で学習するサンプルです。学習パラメータは `config.yaml` から読み込みます。

## セットアップ

```bash
pip install -r requirements.txt
```

## 学習

```bash
python train.py fit --config config.yaml
```

## テスト

```bash
python train.py test --config config.yaml --ckpt_path best
```

`config.yaml` の `model` で Swin Transformer の構造、`data` で CIFAR-10 のデータ設定、`trainer` で Lightning Trainer の設定を変更できます。

## TensorRT 推論

学習済みチェックポイントは次のパスにあります。

```text
lightning_logs/version_1/checkpoints/swin-cifar10-epoch=19-val_acc=0.7693.ckpt
```

CustomMoE plugin は実験的な実装です。まずは MoE ブロックを標準 ONNX 演算に展開する `dense` モードでエクスポートし、TensorRT engine を作成して C++ runner から実行します。

```bash
python scripts/export_moe_onnx.py \
	--model-py model.py \
	--checkpoint lightning_logs/version_1/checkpoints/swin-cifar10-epoch=19-val_acc=0.7693.ckpt \
	--lightning-checkpoint \
	--mode dense \
	--output artifacts/swin_cifar10_dense.onnx

python scripts/build_engine.py \
	--onnx artifacts/swin_cifar10_dense.onnx \
	--engine artifacts/swin_cifar10_dense.engine \
	--fp16 \
	--min-batch 1 --opt-batch 8 --max-batch 32

cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j

./build/swin_cifar10_trt_infer \
	--engine artifacts/swin_cifar10_dense.engine \
	--batch 1 \
	--warmup 20 \
	--iters 100 \
	--topk 5 \
	--output artifacts/logits.bin
```

`CMAKE_CUDA_ARCHITECTURES` は GPU に合わせて変更してください。例: A100 は `80`、RTX 30xx は `86`、Ada/L4 は `89`、H100 は `90` です。

`swin_cifar10_trt_infer` の `--input input.bin` は任意です。指定する場合、入力は `[batch, 3, 32, 32]` の raw FP32 NCHW で、`data.py` の eval transform と同じ正規化済みテンソルにしてください。

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

plugin 経路を試す場合は `--mode plugin` で ONNX を出力し、CMake に `-DBUILD_CUSTOM_MOE_PLUGIN=ON` を渡して `libcustom_moe_plugin.so` をビルドしてください。
