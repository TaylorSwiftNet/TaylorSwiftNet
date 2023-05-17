# TaylorSwiftNet
This repository contains the code for the paper [Taylor Swift: Taylor Driven Temporal Modeling for Swift Future Frame
Prediction](https://arxiv.org/pdf/2110.14392.pdf).

![TaylorSwiftNet](TaylorSwiftNet_Teaser-1.jpg)

## Installation

Setup a conda environment and install all project dependencies.

```bash
conda env create --name taylor --file environment.yml
activate taylor
pip install -e .
```

## How to run the code

To train the MovingMNIST model, use

```bash
python core/main.py --cfg configs/moving_mnist/latest_config.yaml \
--set dataset.root <path_to_dataset>
```

All config parameters are described in `configs/default_config.py`. You can specify parameters by
setting them in a yaml config file or by passing them after `--set` (Format: `--set <key1>
<value1> <key2> <value2> ...`). 

To evaluate a previously trained model checkpoint, use

```bash
python core/main.py --cfg configs/moving_mnist/latest_config.yaml \
--set dataset.root <path_to_dataset> eval_only True model.resume True model.model_state_path <path_to_checkpoint.pt>
```


## Citation
If you use this code or our models, please cite our paper:
```latex
@inproceedings{taylor2022,
    Author    = {Saber Pourheydari, Emad Bahrami, Mohsen Fayyaz, Gianpiero Francesca, Mehdi Noroozi, Juergen Gall},
    Title     = {TaylorSwiftNet: Taylor Driven Temporal Modeling for Swift Future Frame Prediction},
    Booktitle = {British Machine Vision Conference (BMVC)},
    Year      = {2022}
}
```

### Contributors

<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/msaberp">
            <img src="https://avatars.githubusercontent.com/u/30444896?v=4" width="100;" alt="saber"/>
            <br />
            <sub><b>Saber Pourheydari</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://3madbr.github.io">
            <img src="https://raw.githubusercontent.com/3madbr/3madbr.github.io/main/assets/img/emadbr.jpg" width="100;" alt="emad"/>
            <br />
            <sub><b>Emad Bahrami</b></sub>
        </a>
    </td>
    <td align="center">
    <a href="https://mohsenfayyaz89.github.io/">
        <img src="https://avatars.githubusercontent.com/u/14911583?v=4" width="100;" alt="mohsen"/>
        <br />
        <sub><b>Mohsen Fayyaz</b></sub>
    </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

#### Acknowledgment
Felix helped us for refactoring and cleaning the original code.
<!-- readme: samslow,alandefreitas,atharwa-24,EmilStenstrom -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/felixbmuller">
            <img src="https://avatars.githubusercontent.com/u/57685553?v=4" width="100;" alt="felix"/>
            <br />
            <sub><b>Felix B. Müller</b></sub>
        </a>
    </td>
</table>
