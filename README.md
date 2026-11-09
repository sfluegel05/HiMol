# HiMol
# Hierarchical Molecular Graph Self-supervised Learning for Property Prediction
Official Pytorch implementation of HiMol model in the paper "Zang, Xuan., Zhao, Xianbing. & Tang, Buzhou. Hierarchical Molecular Graph Self-Supervised Learning for property prediction. Commun Chem 6, 34 (2023)." https://doi.org/10.1038/s42004-023-00825-5. 

## Environment Setup 


python
rdkit
scipy 
torch
torch-geometric
torch-sparse
tqdm
networkx
numpy
pandas


## Training
You can pretrain the model by
```
mkdir saved_model
python pretrain.py
```

## Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks

Download the downstream data from https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function, and save the .csv files in the ./finetune/dataset/[dataset_name]/raw/, where [dataset_name] is replaced by the downstream dataset name. 
For example, bace.csv is saved in './finetune/dataset/bace/raw/bace.csv'.

```
cd finetune
mkdir model_checkpoints
python finetune.py
```

### MoleculeNet benchmark runner

The repository includes [`finetune/molnet.sh`](finetune/molnet.sh), which
automates the six classification benchmarks used in the HiMol comparison:
BBBP, BACE, SIDER, ClinTox, Tox21, and ToxCast. It runs each dataset with
seeds `0`, `42`, and `12345`, using the dataset-specific learning rates and
dropout values defined in the script. Each run:

1. fine-tunes HiMol from the ZINC15 checkpoint;
2. saves the dataset checkpoint under `finetune/model_checkpoints/`;
3. records the selected test AUC and F1 metrics; and
4. writes per-run results to `finetune/logs/himol_molnet_results.tsv`, followed
   by mean and sample standard deviation summaries.

Before running it, update `HIMOL_DIR` and the
`--input_model_file` path in the script for the local checkout, prepare the
MoleculeNet raw data as described above, and enable or disable Weights & Biases
logging with `--wandb_mode`. Run:

```bash
bash finetune/molnet.sh
```

For comparison with the published MoleculeNet protocol, use scaffold splits
for BBBP and BACE and random splits for Tox21, ToxCast, SIDER, and ClinTox.
The loader uses the DeepChem-provided folds when those processed datasets are
available.

### Fine-tuning HiMol on ChEBI

HiMol expects ChEBI data in its own raw format. The repository provides
[`finetune/prepare_chebi.py`](finetune/prepare_chebi.py) to convert the
`data.pkl` produced by `python-chebai` into the files expected by the HiMol
loader:

- `smiles.txt` — one canonical SMILES per source row;
- `ids.txt` — the ChEBI identifier for each row;
- `labels.npy` — the multi-label ChEBI targets; and
- `classes.txt` — class names in label-column order.

Run the converter with a Python environment that can read the source pickle
(for example the `python-chebai` environment), rather than the HiMol
environment if the pickle was written with NumPy 2:

```bash
cd /path/to/HiMol
python finetune/prepare_chebi.py \
  --data_pkl /path/to/python-chebai/data/chebi_v252/ChEBI50/processed/data.pkl \
  --out_dir finetune/dataset/chebi/raw
```

The converter preserves source-row alignment, including rows with missing or
unparseable structures. Such rows receive an empty SMILES entry and are
skipped when HiMol builds molecular graphs. The generated `ids.txt` must match
the identifiers used by the split CSV.

Prepare a split CSV with columns `id,split`, where `split` is `train`,
`validation` (or `valid`/`val`), or `test`. The `python-chebai` ChEBI split
can be used directly, for example:

```bash
mkdir -p finetune/dataset/chebi/processed
cp /path/to/python-chebai/data/chebi_v252/ChEBI50/processed/splits.csv \
  finetune/dataset/chebi/processed/chebi_splits.csv
```

HiMol generates `finetune/dataset/chebi/processed/chebi_ids.csv`
automatically while processing `ids.txt`; it contains the identifiers for
graphs that were successfully built. The split CSV must therefore use the
same ChEBI IDs as `ids.txt`.

From `HiMol/finetune`, fine-tune the pretrained model on the ChEBI split:

```bash
cd /path/to/HiMol/finetune
python finetune.py \
  --dataset=chebi \
  --split_file=dataset/chebi/processed/chebi_splits.csv \
  --input_model_file=../.saved_model/pretrain.pth \
  --lr_feat=0.001 \
  --lr_pred=0.001 \
  --dropout_ratio=0 \
  --runseed=0 \
  --wandb_run_name=himol_chebi_s0 \
  --wandb_mode=online
```

Use `--wandb_mode=disabled` to run without Weights & Biases. Repeat with
different `--runseed` values for independent runs. The ChEBI loader joins the
split file to `ids.txt`, creates train/validation/test subsets, infers the
number of prediction tasks from `classes.txt`, and writes the fine-tuned
best-validation checkpoint to `model_checkpoints/chebi.pth` (relative to
`HiMol/finetune`).

Citation
--------
Please cite our paper as follows. Thank you.
```
@article{zang2023hierarchical,
  title={Hierarchical Molecular Graph Self-Supervised Learning for property prediction},
  author={Zang, Xuan and Zhao, Xianbing and Tang, Buzhou},
  journal={Communications Chemistry},
  volume={6},
  number={1},
  pages={34},
  year={2023}
}
```
