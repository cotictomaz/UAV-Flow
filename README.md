<div align="center">

<h1>UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning</h1>

<div>
    Xiangyu Wang&emsp;
    Donglin Yang&emsp;
    Yue Liao&emsp;
    Wenhao Zheng&emsp;
    Bin Dai&emsp;
    Hongsheng Li&emsp;
    Si Liu&emsp;
    et al.
</div>

<div>
    <strong>NeurIPS 2025</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow/tree/main" target="_blank">
        <img src="https://img.shields.io/badge/Dataset-UAV--Flow-orange">
        </a>
        <a href="https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim/tree/main" target="_blank">
        <img src="https://img.shields.io/badge/Dataset-UAV--Flow--Sim-orange">
        </a>
        <a href="https://huggingface.co/wangxiangyu0814/OpenVLA-UAV/tree/main" target="_blank">
        <img src="https://img.shields.io/badge/Checkpoint-OpenVLA--UAV-blue">
        </a>
        <a href="#-citation" target="_blank">
        <img src="https://img.shields.io/badge/Cite-BibTeX-green">
        </a>
    </h4>
</div>

<strong>UAV-Flow provides real-world and simulated UAV trajectories, an UnrealZoo-based evaluation environment, and an OpenVLA-UAV baseline for instruction-conditioned UAV control.</strong>


---

</div>

## 📢 News

* **[Latest]** UAV-Flow real-world trajectories and UAV-Flow-Sim simulation trajectories are available on Hugging Face.
* **[Latest]** UAV-Flow-Eval and the OpenVLA-UAV training/evaluation code are included in this repository.
* **[Coming Soon]** Pi-0-UAV support will be released later.

## 💡 Highlights

* **Real and simulated UAV data.** UAV-Flow contains real-world UAV trajectories, while UAV-Flow-Sim provides simulation trajectories for scalable training and evaluation.
* **Instruction-conditioned evaluation.** UAV-Flow-Eval runs UAV control tasks in UnrealZoo Gym and evaluates generated trajectories with nDTW.
* **OpenVLA-UAV baseline.** The repository includes OpenVLA-based finetuning code, a Flask inference server, and a released OpenVLA-UAV checkpoint.

## 🛠️ Usage

### Installation

This repository contains two main Python environments: one for OpenVLA-UAV training/inference and one for UAV-Flow-Eval.

Set up OpenVLA-UAV:

```bash
cd OpenVLA-UAV
conda create -n openvla python=3.10 -y
conda activate openvla

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -e .

pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

Set up UAV-Flow-Eval:

```bash
cd UAV-Flow-Eval
conda create -n unrealcv python=3.11 -y
conda activate unrealcv
pip install -e .
```

### Data Preparation

Download the datasets:

* [UAV-Flow](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow/tree/main): real-world trajectories.
* [UAV-Flow-Sim](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim/tree/main): simulation trajectories.

If you need to convert parquet files into the folder format used by the training code, update the local paths in `dataset_tools/prepare_data.py`, then run:

```bash
cd dataset_tools
python prepare_data.py
```

### Model Preparation

For finetuning, download the pretrained [OpenVLA checkpoint](https://huggingface.co/openvla/openvla-7b).

For evaluation, download the released [OpenVLA-UAV checkpoint](https://huggingface.co/wangxiangyu0814/OpenVLA-UAV/tree/main), then set `model_path` in `OpenVLA-UAV/vla-scripts/openvla_act.py`:

```python
cfg = {
    "gpu_id": 0,
    "model_path": "/path/to/your/finetuned_model",
    "http_port": 5007,
    "unnorm_key": "sim",
    "do_sample": False,
}
```

### Training

Open `OpenVLA-UAV/vla-scripts/finetune_uav.sh` and update all local paths, including the pretrained model path, dataset path, run directory, adapter directory, and Weights & Biases settings.

Then launch finetuning:

```bash
cd OpenVLA-UAV
bash vla-scripts/finetune_uav.sh
```

### Evaluation

Download and extract the packaged UnrealZoo environment:

[Collection_WinNoEditor_0424_25.zip](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/file/view/master/Collection_WinNoEditor_0424_25.zip?id=77779&status=2)

Configure the simulator path in:

```text
UAV-Flow-Eval/gym_unrealcv/envs/setting/Track/DowntownWest.json
```

Update `env_bin_win` to the extracted Unreal environment executable path.

Start the OpenVLA-UAV inference server:

```bash
cd OpenVLA-UAV
python vla-scripts/openvla_act.py
```

In another terminal, run UAV-Flow-Eval:

```bash
cd UAV-Flow-Eval
python batch_run_act_all.py
```

To use a different inference server port:

```bash
python batch_run_act_all.py --server_port 5006
```

After evaluation finishes, compute nDTW metrics:

```bash
python metric.py
```

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{wang2026uav,
  title={Uav-flow colosseo: A real-world benchmark for flying-on-a-word uav imitation learning},
  author={Wang, Xiangyu and Yang, Donglin and Liao, Yue and Zheng, Wenhao and Dai, Bin and Li, Hongsheng and Liu, Si and others},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License. See [LICENSE](./LICENSE) for more information.

## 🙏 Acknowledgement

This project builds on [OpenVLA](https://github.com/openvla/openvla), [UnrealZoo Gym](https://github.com/UnrealZoo/unrealzoo-gym). We thank the maintainers of these projects for releasing their code, models, and simulation resources.
