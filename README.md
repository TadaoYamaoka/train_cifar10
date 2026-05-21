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
