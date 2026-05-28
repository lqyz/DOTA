#!/bin/bash
# DOTA 数据集下载脚本，所有数据存放于 /root/data
# 运行方式: bash download_datasets.sh
# 注意: ImageNet、ImageNet-Sketch、ImageNet-A、ImageNet-R 需手动下载

set -e

DATA=/root/data
mkdir -p "$DATA"

echo "=== 预检查: 安装必要工具 ==="
which wget  >/dev/null 2>&1 || apt-get install -y wget
which unzip >/dev/null 2>&1 || apt-get install -y unzip
which gdown >/dev/null 2>&1 || pip install gdown

# ============================================================
# 1. ImageNet (需手动下载验证集)
# ============================================================
echo "=== [1/15] ImageNet ==="
mkdir -p "$DATA/imagenet/images"
# 手动: 将 ImageNet 验证集放到 $DATA/imagenet/images/val/
# 即: $DATA/imagenet/images/val/n01440764/, n01443537/, ...
cd "$DATA/imagenet"
if [ ! -f classnames.txt ]; then
    gdown "https://drive.google.com/uc?id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF" -O classnames.txt
fi

# ============================================================
# 2. ImageNet-A (需手动下载)
# ============================================================
echo "=== [2/15] ImageNet-A ==="
mkdir -p "$DATA/imagenet-adversarial"
# 手动: 从 https://github.com/hendrycks/natural-adv-examples 下载并解压
# 最终结构: $DATA/imagenet-adversarial/imagenet-a/n01440764/, ...
if [ ! -f "$DATA/imagenet-adversarial/classnames.txt" ]; then
    cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-adversarial/classnames.txt" 2>/dev/null || true
fi

# ============================================================
# 3. ImageNetV2
# ============================================================
echo "=== [3/15] ImageNetV2 ==="
mkdir -p "$DATA/imagenetv2"
cd "$DATA/imagenetv2"
if [ ! -d imagenetv2-matched-frequency-format-val ]; then
    wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
    tar -xzf imagenetv2-matched-frequency.tar.gz
    rm -f imagenetv2-matched-frequency.tar.gz
fi
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" .

# ============================================================
# 4. ImageNet-R (需手动下载)
# ============================================================
echo "=== [4/15] ImageNet-R ==="
mkdir -p "$DATA/imagenet-rendition"
# 手动: 从 https://github.com/hendrycks/imagenet-r 下载并解压
# 最终结构: $DATA/imagenet-rendition/imagenet-r/n01440764/, ...
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-rendition/classnames.txt" 2>/dev/null || true

# ============================================================
# 5. ImageNet-Sketch (需手动下载)
# ============================================================
echo "=== [5/15] ImageNet-Sketch ==="
mkdir -p "$DATA/imagenet-sketch"
# 手动: 从 https://github.com/HaohanWang/ImageNet-Sketch 下载并解压
# 最终结构: $DATA/imagenet-sketch/images/n01440764/, ...
[ -f classnames.txt ] || cp "$DATA/imagenet/classnames.txt" "$DATA/imagenet-sketch/classnames.txt" 2>/dev/null || true

# ============================================================
# 6. Caltech101
# ============================================================
echo "=== [6/15] Caltech101 ==="
mkdir -p "$DATA/caltech-101"
cd "$DATA/caltech-101"
if [ ! -d 101_ObjectCategories ]; then
    wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
    tar -xzf 101_ObjectCategories.tar.gz
    rm -f 101_ObjectCategories.tar.gz
fi
if [ ! -f split_zhou_Caltech101.json ]; then
    gdown "https://drive.google.com/uc?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" -O split_zhou_Caltech101.json
fi

# ============================================================
# 7. OxfordPets
# ============================================================
echo "=== [7/15] OxfordPets ==="
mkdir -p "$DATA/oxford_pets"
cd "$DATA/oxford_pets"
if [ ! -d images ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    tar -xzf images.tar.gz
    rm -f images.tar.gz
fi
if [ ! -d annotations ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    tar -xzf annotations.tar.gz
    rm -f annotations.tar.gz
fi
if [ ! -f split_zhou_OxfordPets.json ]; then
    gdown "https://drive.google.com/uc?id=1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" -O split_zhou_OxfordPets.json
fi

# ============================================================
# 8. StanfordCars
# ============================================================
echo "=== [8/15] StanfordCars ==="
mkdir -p "$DATA/stanford_cars"
cd "$DATA/stanford_cars"
if [ ! -d cars_test ]; then
    wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
    tar -xzf cars_test.tgz
    rm -f cars_test.tgz
fi
if [ ! -f cars_test_annos_withlabels.mat ]; then
    wget http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat
fi
if [ ! -f split_zhou_StanfordCars.json ]; then
    gdown "https://drive.google.com/uc?id=1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT" -O split_zhou_StanfordCars.json
fi

# ============================================================
# 9. OxfordFlowers
# ============================================================
echo "=== [9/15] OxfordFlowers ==="
mkdir -p "$DATA/oxford_flowers"
cd "$DATA/oxford_flowers"
if [ ! -d jpg ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    tar -xzf 102flowers.tgz
    rm -f 102flowers.tgz
fi
if [ ! -f imagelabels.mat ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
fi
if [ ! -f cat_to_name.json ]; then
    gdown "https://drive.google.com/uc?id=1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" -O cat_to_name.json
fi
if [ ! -f split_zhou_OxfordFlowers.json ]; then
    gdown "https://drive.google.com/uc?id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" -O split_zhou_OxfordFlowers.json
fi

# ============================================================
# 10. Food101
# ============================================================
echo "=== [10/15] Food101 ==="
cd "$DATA"
if [ ! -d food-101 ]; then
    wget https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz
    tar -xzf food-101.tar.gz
    rm -f food-101.tar.gz
fi
cd "$DATA/food-101"
if [ ! -f split_zhou_Food101.json ]; then
    gdown "https://drive.google.com/uc?id=1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl" -O split_zhou_Food101.json
fi

# ============================================================
# 11. FGVCAircraft
# ============================================================
echo "=== [11/15] FGVCAircraft ==="
cd "$DATA"
if [ ! -d fgvc_aircraft ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
    tar -xzf fgvc-aircraft-2013b.tar.gz
    mv fgvc-aircraft-2013b/data fgvc_aircraft
    rm -rf fgvc-aircraft-2013b fgvc-aircraft-2013b.tar.gz
fi

# ============================================================
# 12. SUN397
# ============================================================
echo "=== [12/15] SUN397 ==="
mkdir -p "$DATA/sun397"
cd "$DATA/sun397"
if [ ! -d SUN397 ]; then
    wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
    tar -xzf SUN397.tar.gz
    rm -f SUN397.tar.gz
fi
if [ ! -f split_zhou_SUN397.json ]; then
    wget https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip
    unzip -o Partitions.zip
    rm -f Partitions.zip
    gdown "https://drive.google.com/uc?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq" -O split_zhou_SUN397.json
fi

# ============================================================
# 13. DTD
# ============================================================
echo "=== [13/15] DTD ==="
cd "$DATA"
if [ ! -d dtd ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar -xzf dtd-r1.0.1.tar.gz
    rm -f dtd-r1.0.1.tar.gz
fi
cd "$DATA/dtd"
if [ ! -f split_zhou_DescribableTextures.json ]; then
    gdown "https://drive.google.com/uc?id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" -O split_zhou_DescribableTextures.json
fi

# ============================================================
# 14. EuroSAT
# ============================================================
echo "=== [14/15] EuroSAT ==="
mkdir -p "$DATA/eurosat"
cd "$DATA/eurosat"
if [ ! -d 2750 ]; then
    wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
    unzip -o EuroSAT.zip
    rm -f EuroSAT.zip
fi
if [ ! -f split_zhou_EuroSAT.json ]; then
    gdown "https://drive.google.com/uc?id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o" -O split_zhou_EuroSAT.json
fi

# ============================================================
# 15. UCF101
# ============================================================
echo "=== [15/15] UCF101 ==="
mkdir -p "$DATA/ucf101"
cd "$DATA/ucf101"
if [ ! -d UCF-101-midframes ]; then
    gdown "https://drive.google.com/uc?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" -O UCF-101-midframes.zip
    unzip -o UCF-101-midframes.zip
    rm -f UCF-101-midframes.zip
fi
if [ ! -f split_zhou_UCF101.json ]; then
    gdown "https://drive.google.com/uc?id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" -O split_zhou_UCF101.json
fi

echo ""
echo "=============================================="
echo "  自动下载完成！"
echo "  以下数据集需要手动下载:"
echo "  - ImageNet 验证集 -> $DATA/imagenet/images/val/"
echo "  - ImageNet-A       -> $DATA/imagenet-adversarial/imagenet-a/"
echo "  - ImageNet-R       -> $DATA/imagenet-rendition/imagenet-r/"
echo "  - ImageNet-Sketch  -> $DATA/imagenet-sketch/images/"
echo "=============================================="
