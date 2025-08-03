# Install

## 前提

※！！**絶対にログインノードで環境をインストールしないでください。ログインノードに過度な負荷がかかり、停止して全体がログインできなくなる恐れがあります。**

* 計算環境:  1 node, 8 GPU (Nvidia H100)
  * YOU_TEAM を案内された partition 番号に置き換えてください。
  * 例: `$ srun --partition=YOU_TEAM --nodes=1 --gpus-per-node=8 --cpus-per-task=240 --time=30:30:00 --nodelist=osk-gpu[YOU_TEAM] --job-name="env_install_test" --pty bash -i`

## Step 0. 環境構築

### Step 0-1. Python仮想環境作成前における下準備

```sh
cd ~/

mkdir -p ~/conda_env

# 念のためSSH等が故障したときなどに備えて~/.bashrcをバックアップしておく。
cp ~/.bashrc ~/.bashrc.backup

# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

```

### Step 0-2. conda環境生成

```sh
export CONDA_PATH="$HOME/conda_env"
# 確認
echo $CONDA_PATH
```

```sh
# Python仮想環境を作成。
conda create --prefix $CONDA_PATH python=3.11 -y
```

Python仮想環境を有効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を編集するように設定。
```sh
LD_LIB_APPEND="/usr/lib64:/usr/lib:"$CONDA_PATH"/lib:"$CONDA_PATH"/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "LD_LIB_APPEND:"$LD_LIB_APPEND

mkdir -p $CONDA_PATH/etc/conda/activate.d && \
    echo 'export ORIGINAL_LD_LIBRARY_PATH='$LD_LIBRARY_PATH > $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDNN_PATH='$CUDNN_PATH          >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDA_HOME='$CUDA_HOME            >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    # echo "export LD_LIBRARY_PATH=\"/usr/lib64:/usr/lib:"$CONDA_PATH"/lib:$CONDA_PATH/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH\"" >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CONDA_PATH='$CONDA_PATH            >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export LD_LIBRARY_PATH='$LD_LIB_APPEND             >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDNN_PATH='$CONDA_PATH'/lib'               >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDA_HOME='$CONDA_PATH'/'                   >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CONDA_PATH='$CONDA_PATH'/'                  >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    chmod +x $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh
```

Python仮想環境を無効化した時に自動で環境変数 `$LD_LIBRARY_PATH` を元に戻すように設定。
```sh
mkdir -p $CONDA_PATH/etc/conda/deactivate.d && \
    echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export LD_CUDNN_PATH='$ORIGINAL_CUDNN_PATH       >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export LD_CUDA_HOME='$ORIGINAL_CUDA_HOME         >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'export CONDA_PATH='$ORIGINAL_CONDA_PATH          >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_LD_LIBRARY_PATH'                  >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CUDNN_PATH'                       >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CUDA_HOME'                        >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_CONDA_PATH'                        >> $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    chmod +x $CONDA_PATH/etc/conda/deactivate.d/rollback_environment_variable.sh
```

.bashrcの内容を再読み込みし、最新の設定を現在のシェルに反映させる
```sh
source ~/.bashrc

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh
```

condaの初期化スクリプトを自動で追記
```sh
conda init
```

以下が~/.bashrcに追記される。
```.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/appli/miniconda3/24.7.1-py311/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh" ]; then
        . "/home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh"
    else
        export PATH="/home/appli/miniconda3/24.7.1-py311/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```


シェル起動時に自動でbase環境が有効化されるのを防ぐ設定
```sh
conda config --set auto_activate_base false
```

念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
```sh
conda deactivate
```

作成したPython仮想環境を有効化。
※無効化するときのコマンドは `$ conda deactivate` 。
```sh
conda activate $CONDA_PATH
```


### Step 0-3. パッケージ等のインストール

GPU計算やディープラーニングに必要なCUDA Toolkit・cuDNN・GCC/G++などの開発ツール、Pythonパッケージのビルド・管理ツール、Gitおよび大容量ファイル管理用のGit LFSをインストール。
``` sh
# CUDA Toolkit 12.4.1（NVIDIAのGPU計算用ライブラリ
conda install cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 -y
# cuDNN（ディープラーニング向けNVIDIA GPU用ライブラリ）
conda install -c conda-forge cudnn -y
# Linux 64bit用のGCC（Cコンパイラ）とG++（C++コンパイラ）
conda install gcc_linux-64 gxx_linux-64 -y
# pipを最新版にアップグレード
pip install --upgrade pip
# wheel（Pythonパッケージビルド用）、cmake（ビルドシステム）、ninja（ビルドツール）を最新版にアップグレード
pip install --upgrade wheel cmake ninja
# Gitをインストール
conda install git -y
# Git Large File Storage（大容量ファイル管理用Git拡張）をインストール
conda install anaconda::git-lfs -y
# Git LFSの初期化を行い、Gitで大容量ファイルを扱えるようにする。
git lfs install
```

### Step 0-4. このgitレポジトリのクローン

``` sh
cd ~/

git clone git@github.com:matsuolab/llm_bridge_prod.git

cd ~/llm_bridge_prod/train

ls -lh

cd ../
```

### Step 0-5. conda環境プリント確認

``` sh
conda deactivate
conda activate $CONDA_PATH
conda env list
echo "--- CONDA_PREFIX: ---"
echo "CONDA_PREFIX:"$CONDA_PREFIX
echo "--- pip, python パスはCONDA_PREFIXで始まる ---"
echo "pip:"$(which pip)
echo "python:"$(which python)
echo "--- 環境変数 ---"
printenv |grep CUDA
printenv |grep CUDNN
printenv |grep LD_LIB
```

### Step 0-6. Verlのインストール

``` sh
#home ディレクトリを例にしていますが、～は任意のディレクトリに置き換えられます。
cd ~/

mkdir -p deps

cd ~/deps

# verlのレポジトリをクローン。
git clone https://github.com/volcengine/verl.git

cd verl
# 必ず USE_MEGATRON=1 にしてください。
# ※不要なエラーを防ぐため、PyTorch と vllm のバージョンをむやみに変更せず、公式のバージョンとできるだけ一致させてください。
USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh
```

インストール時のメッセージでエラーがでているが、気にせず進む。
後続でインストール予定のライブラリと合わせて、依存関係が両立しない（作業を継続して左記を確認した）。
```markdown
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.6.0 requires nvidia-cudnn-cu12==9.1.0.70; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cudnn-cu12 9.8.0.87 which is incompatible.
Successfully installed nvidia-cudnn-cu12-9.8.0.87
Successfully installed all packages
```


```sh
pip install --no-deps -e .

pip install --no-cache-dir six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate safetensors einops peft datasets trl matplotlib sortedcontainers brotli zstandard cryptography colorama audioread soupsieve defusedxml babel codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython

pip install -U "ray[data,train,tune,serve]"

pip install --upgrade protobuf 

cd ../
```
ここでもインストール時のメッセージでエラーがでているが、気にせず進む。
依存関係が両立が現実的に難しいため。
```markdown
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.6.0 requires nvidia-cudnn-cu12==9.1.0.70; platform_system == "Linux" and platform_machine == "x86_64", but you have nvidia-cudnn-cu12 9.8.0.87 which is incompatible.
Successfully installed nvidia-cudnn-cu12-9.8.0.87
Successfully installed all packages
```

### Step 0-7. apexのインストール

``` sh
cd  ~/deps
# apexのレポジトリをクローン。
git clone https://github.com/NVIDIA/apex
cd apex
pip cache purge
# apexのインストール
# ※80coresで20分ほどかかるので注意。
python setup.py install \
       --cpp_ext --cuda_ext \
       --distributed_adam \
       --deprecated_fused_adam \
       --xentropy \
       --fast_multihead_attn
cd ../
```

インストール時のメッセージ末尾。
```markdown
Installed /home/Competition2025/P12/P12U020/conda_env/lib/python3.11/site-packages/apex-0.1-py3.11-linux-x86_64.egg
Processing dependencies for apex==0.1
Searching for packaging==25.0
Best match: packaging 25.0
Adding packaging 25.0 to easy-install.pth file
detected new path './apex-0.1-py3.11-linux-x86_64.egg'

Using /home/Competition2025/P12/P12U020/conda_env/lib/python3.11/site-packages
Finished processing dependencies for apex==0.1
```

### Step 0-8. Flash Attention 2のインストール

``` sh
# ※80coresで25分ほどかかるので注意。
ulimit -v unlimited
MAX_JOBS=64 pip install flash-attn==2.6.3 --no-build-isolation
```

インストール時のメッセージ末尾。
```
  Building wheel for flash-attn (setup.py) ... done
  Created wheel for flash-attn: filename=flash_attn-2.6.3-cp311-cp311-linux_x86_64.whl size=179967482 sha256=be8e9e04e150c38cf34fd920444dcf1705b0d27c91ee49456563e298987bcdb7
  Stored in directory: /home/Competition2025/P12/P12U020/.cache/pip/wheels/e3/ef/b1/7889928ffa2dea61032e61480db4e4c20d00a9d9e28cd4f55a
Successfully built flash-attn
Installing collected packages: flash-attn
Successfully installed flash-attn-2.6.3
```


### Step 0-9. TransformerEngineのインストール

``` sh
cd  ~/deps
git clone https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
git submodule update --init --recursive
git checkout release_v2.4
# ※80coresでXX分ほどかかるので注意。
NMAX_JOBS=64 VTE_FRAMEWORK=pytorch pip install --no-cache-dir .
cd ../
```

インストール時のメッセージ末尾。
```
Successfully built transformer_engine nvdlfw-inspect
Installing collected packages: nvdlfw-inspect, transformer_engine
  Attempting uninstall: transformer_engine
    Found existing installation: transformer_engine 2.2.0+d0c452c
    Uninstalling transformer_engine-2.2.0+d0c452c:
      Successfully uninstalled transformer_engine-2.2.0+d0c452c
Successfully installed nvdlfw-inspect-0.1.0 transformer_engine-2.4.0+3cd6870c
```

### Step 0-10. インストール状況のチェック
※以下のコマンドでPythonライブラリにエラーがないか確認。
Apexのバージョンは「unknown」でも問題ありませんが、エラーが発生した場合は再インストールする。
``` sh
cd ~/

python - <<'PY'
import importlib, apex, torch, sys

# 各モジュールがインポートできるかを順に確認
for mod in (
    "apex.transformer",
    "apex.normalization.fused_layer_norm",
    "apex.contrib.optimizers.distributed_fused_adam",
    "flash_attn",
    "verl.trainer",
    "ray",
    "transformer_engine",
):
    print("✅" if importlib.util.find_spec(mod) else "❌", mod)

# flash-attention のバージョン
try:
    import flash_attn
    flash_ver = getattr(flash_attn, "__version__", "unknown")
except ImportError:
    flash_ver = "not installed"

# verl.trainer.main_ppo が存在するか
try:
    from verl.trainer import main_ppo as _main_ppo   # noqa: F401
    main_ppo_flag = "✅ main_ppo in verl.trainer"
except ImportError:
    main_ppo_flag = "❌ main_ppo in verl.trainer"
print(main_ppo_flag)

# Ray のバージョン
try:
    import ray
    ray_ver = getattr(ray, "__version__", "unknown")
except ImportError:
    ray_ver = "not installed"

# TransformerEngine のバージョン
try:
    import transformer_engine
    te_ver = getattr(transformer_engine, "__version__", "unknown")
except ImportError:
    te_ver = "not installed"

# バージョン情報を出力（元のスクリプトと同じ2段階出力）
print("Flash-Attention ver.:", flash_ver, end=" | ")
print("Ray ver.:", ray_ver, end=" | ")
print("TransformerEngine ver.:", te_ver, end=" | ")
print("Apex ver.:", getattr(apex, "__version__", "unknown"),
      "| Torch CUDA:", torch.version.cuda,
      "| Python:", sys.version.split()[0])
PY
```

実行メッセージに以下のようなエラーがでる可能性あり。
同じエラーメッセージの場合は、verlで引っかかっている。
```markdown
Traceback (most recent call last):
  File "<stdin>", line 13, in <module>
  File "<frozen importlib.util>", line 95, in find_spec
  File "/home/Competition2025/P12/P12U020/deps/verl/verl/__init__.py", line 23, in <module>
    from .protocol import DataProto
  File "/home/Competition2025/P12/P12U020/deps/verl/verl/protocol.py", line 37, in <module>
    from verl.utils.device import get_device_id, get_torch_device
  File "/home/Competition2025/P12/P12U020/deps/verl/verl/utils/__init__.py", line 15, in <module>
    from . import config, tokenizer
  File "/home/Competition2025/P12/P12U020/deps/verl/verl/utils/config.py", line 18, in <module>
    from omegaconf import DictConfig, ListConfig, OmegaConf
ModuleNotFoundError: No module named 'omegaconf'
```

(エラー時の追加対応)
omegaconfを追加
```sh
pip install -U hydra-core
```

以下を確認できれば無事完了。
```
✅ apex.transformer
✅ apex.normalization.fused_layer_norm
✅ apex.contrib.optimizers.distributed_fused_adam
✅ flash_attn
✅ verl.trainer
✅ ray
✅ transformer_engine
✅ main_ppo in verl.trainer
Flash-Attention ver.: 2.6.3 | Ray ver.: 2.48.0 | TransformerEngine ver.: 2.4.0+3cd6870c | Apex ver.: unknown | Torch CUDA: 12.4 | Python: 3.11.13
```
