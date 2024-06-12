# Multi-label Prototype-aware Structured Contrastive Distillation

Knowledge distillation (KD) has demonstrated considerable success in scenarios involving multi-class single-label learning. However, its direct application to multi-label learning proves challenging due to complex correlations in multi-label structures, causing student models to overlook more finely structured semantic relations present in the teacher model. In this paper, we present a solution called Multi-Label Prototype-Aware Structured Contrastive Distillation (MPSCD), comprising two modules: Prototype-Aware Contrastive Representation Distillation (PCRD) and Prototype-Aware Cross-image Structure Distillation (PCSD). The PCRD module maximizes the mutual information of prototype-aware representation between the student and teacher, ensuring semantic representation structure consistency to improve the compactness of intra-class and dispersion of inter-class representations. In the PCSD module, we introduce sample-to-sample and sample-to-prototype structured contrastive distillation to model prototype-aware cross-image structure consistency, guiding the student model to maintain a coherent label semantic structure with the teacher across multiple instances. To enhance prototype guidance stability, we introduce batch-wise dynamic prototype correction for updating class prototypes. Experimental results on three public benchmark datasets validate the effectiveness of our proposed method, demonstrating its superiority over state-of-the-art methods.

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
pip install pytorch-gpu
pip install scikit-learn
pip install numpy
pip install scipy

###  Use your own dataset

[Any name you want]
  |--VOCtrainval2007
    |--VOCdevkit
      |--VOC2007
        |--JPEGImages
          |--000005.jpg
          |--...
        |--ImageSets
          |--Main
            |--trainval.txt
  |--VOCtest2007
    |--VOCdevkit
      |--VOC2007
        |--JPEGImages
          |--000001.jpg
          |--...
        |--ImageSets
          |--Main
            |--test.txt

### Quick start
You can train on voc2007 with default settings stored in ./configs/voc/, Run distillation by following commands
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_file ./configs/voc/resnet50_to_resnet18_mpscd.py --data_root ./dataset/
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_file ./configs/voc/wrn101_to_wrn50_mpscd.py --data_root ./dataset/
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_file ./configs/voc/repvgg_a2_to_repvgg_a0_mpscd.py --data_root ./dataset/
CUDA_VISIBLE_DEVICES=1 python main.py --cfg_file ./configs/voc/swin_t_to_mobilenetv2_mpscd.py --data_root ./dataset/
```




