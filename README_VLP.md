## Table of Contents

* [Image-based VL-PTMs](#image-based-vl-ptms)
  * [Representation Learning](#representation-learning)
  * [Task-specific](#task-specific)
  * [Other Analysis](#other-analysis)
* [Video-based VL-PTMs](#video-based-vl-ptms)
* [Speech-based VL-PTMs](#speech-based-vl-ptms)
* [Other Transformer-based multimodal networks](#other-transformer-based-multimodal-networks)
* [Other Resources](#other-resources)


# Image-based VL-PTMs

## Representation Learning

[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265), NeurIPS 2019 [[code]](https://github.com/jiasenlu/vilbert_beta)
    Conceptual Captions 数据集在预训练中的比例越高，图文任务的指标单调升高，包含图文检索任务。 "the percentage of the Conceptual Captions dataset used during pre-training. We see monotonic gains as the pretraining dataset size grows.”
[LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490), EMNLP 2019 [[code]](https://github.com/airsplay/lxmert)
    北卡，任务: VQA, GQA, NLVR, 三个transformer，Cross-Modality
[VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530), ICLR 2020 [[code]](https://github.com/jackroos/VL-BERT)
    微软：segment embedding, position embedding, masked object prediction, 任务: VQA, VCR, RefCOCO+, 
[VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557), arXiv 2019/08, ACL 2020 [[code]](https://github.com/uclanlp/visualbert)
    UCLA: 主打简化VL-BERT，隐式对句法关系敏感，跟踪动词与对应其自变量的图像区域之间的关系
[Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066), AAAI 2020
    北大+ MSRA, 与VL-BERT思想类似，有位置编码，图文分割编码，一个transformer encoder，任务： Masked Language Modeling (MLM), Masked Object Classification (MOC) and Visual-linguistic Matching (VLM)
[Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/pdf/1909.11059.pdf), AAAI 2020, [[code]](https://github.com/LuoweiZhou/VLP), (**VLP**)
    密西根 + 微软： 经典VLP项目, 之前哪个任务用VLP提的特征？？？ (DSRAN)   transformer encoder decoder unified in one transformer.
[UNITER: Learning Universal Image-text Representations](https://arxiv.org/abs/1909.11740), ECCV 2020, [[code]](https://github.com/ChenRocks/UNITER)
    微软： 4个数据集预训练(COCO,Visual Genome, Conceptual Captions, and SBU Captions), 区别于随机掩码，采用条件掩码(i.e., masked language/region modeling is conditioned on full observation of image/text), 
    全局图文对齐(global image-text alignment), 词与图像检测区域对齐。 结构与Oscar类似，区别text encoder与 image encoder分开，通过掩码文预测文，掩码图预测图。
[Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063), arXiv 2019/12
    Word-Object Alignment, 弱监督，辅助训练，观点： 学习模态间的的对齐没必要依靠自动，可以通过弱监督Word-Object Alignment改善模型质量。
[InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining](https://arxiv.org/abs/2003.13198), arXiv 2020/03, KDD'20, August
    北大，计算语言学实验室+阿里： 和VL-BERT 很像，有位置，分割编码，区别是i与t分开encoding,之后再进单流交互模块(Single-Stream Interaction Module),再分开提取输出特征
    效果整体和VL-BERT差不多，比VilBERT好一点。
[Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/pdf/2004.06165.pdf), arXiv 2020/04, ECCV 2020
    微软Oscar, 特点：利用object tags信息，动机：发现图像检测的静态物品的识别准确率高，并且该物品往往在caption中也有对应的文本。
    指标高于UNITER, Unicoder-VL
[Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers](https://arxiv.org/abs/2004.00849), arXiv 2020/04
    中科大，微软：两个数据集(Visual Genome, MS-COCO)，random pixel sampling 掩码方式，图像像素与文本对齐。
    文本编码包含：tokens，位置，语义。 不需要检测框，全局像素级对齐。 无代码，指标高于Visual BERT, VLBERT, UNITER, LXMBERT.
[ERNIE-VIL: KNOWLEDGE ENHANCED VISION-LANGUAGE REPRESENTATIONS THROUGH SCENE GRAPH](https://arxiv.org/abs/2006.16934), arXiv 2020/06
    百度, Scene Graph 知识图谱 + transformer; Object, attribute, relationship prediction, 指标高于Oscar
[DeVLBert: Learning Deconfounded Visio-Linguistic Representations](https://arxiv.org/abs/2008.06884), ACM MM 2020, [[code]](https://github.com/shengyuzhang/DeVLBert)
    浙大，阿里：De-confounded(去模糊的)。 在跨域的预训练模型中，现有基于似然概率的图文模型中，因为数据集的bias导致条件概率可以很高带来错误，误导文本或图像之间的强关联。
    主要做文检索图的任务。
    out-of domain visio-linguistic pretraining：where the pretraining data distribution differs from that of downstream data on which the pretrained model will be fine-tuned.
[SEMVLP: VISION-LANGUAGE PRE-TRAINING BY ALIGNING SEMANTICS AT MULTIPLE LEVELS](https://openreview.net/forum?id=Wg2PSpLZiH), ICLR 2021 submission（withdrawn submission）
    阿里，文章暂时被拒。 multi levels: 图文特征加和的 single stream mode 和 图文分开 masked prediction的 two stream mode 
[CAPT: Contrastive Pre-Training for Learning Denoised Sequence Representations](https://arxiv.org/pdf/2010.06351.pdf), arXiv 2020/10
    北大： 消除在transformer预训练模型训练过程中加入的随机噪声协变效应，消除噪音影响带来的预训练与微调结果之间差异。
    提出： 学习噪音不变量序列表达，保持加入噪音干扰前后序列的一致性，通过无监督的instance-wise 的训练信号(二分类)。
    结果 略高于LXMBERT,高于VL-BERT, VisualBERT，ViLBERT
[Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/pdf/2011.15124.pdf), arXiv 2020/11
    ETH,剑桥: 讨论 dual-stream 和 single stream 两种不同形式的encoder 的区别，以及如何统一两者。
    一个重要结论： “Our experiments show that training data and hyperparameters are responsible for most of
                the differences between the reported results, but they also reveal that the embedding layer
                plays a crucial role in these massive models.”  
[LAMP: Label Augmented Multimodal Pretraining](https://arxiv.org/pdf/2012.04446.pdf), arXiv 2020/12
    浙大+阿里：对图像物体生成更多标签，达到丰富图文对数据、微调图文对齐关系的目的。 结构框架沿用BERT。 工程经验类论文，全篇无公式。
[Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network](https://arxiv.org/pdf/2101.11562.pdf), AAAI 2021,[[code]](https://github.com/YehLi/TDEN)
    京东+中山： 研究编解码器 Encoder-Decoder Network 对图文理解和生成的，之前的大部分工作都只有编码器。
    作者提出之所以解码器未使用是因为 VL 两个领域学科的跨度太大。同时提出一种新的掩码策略： scheduled sampling strategy ，
    代码已有github但还未准备就绪。
[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/pdf/2102.03334.pdf), arXiv 2021
    提出无卷积，无fasterRCNN检测框的图文编码transformer，图像仅用线性编码器，并对现有使用不同视觉编码方式的方法进行对比。
    打造最小的VLP模型，实验结果一般，想打脸装备多种图像预处理的方法，认为跨模态认为应该多探索不同模态在transformer内部的交互。
    结论和未来工作的部分可以参考一下。提出探索适用于文本与图像的增强策略比较 promising，比如高斯模糊，比如上面的LAMP文章。
## Task-specific

**VCR**: [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054), EMNLP 2019, [[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2), (**B2T2**)

**TextVQA**: [Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258), CVPR 2020, [[code]](https://github.com/ronghanghu/pythia/tree/project/m4c/projects/M4C), (**M4C**)

**VisDial**: [VD-BERT: A Unified Vision and Dialog Transformer with BERT](https://arxiv.org/abs/2004.13278), EMNLP 2020 [[code]](https://github.com/salesforce/VD-BERT), (**VD-BERT**)

**VisDial**: [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379), ECCV 2020 [[code]](https://github.com/vmurahari3/visdial-bert), (**VisDial-BERT**)

**VLN**: [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638), CVPR 2020, [[code]](https://github.com/weituo12321/PREVALENT), (**PREVALENT**)

**Text-image retrieval**: [ImageBERT: Cross-Modal Pre-training with Large-scale Weak-supervised Image-text Data](https://arxiv.org/abs/2001.07966), arXiv 2020/01
    微软：专门做图文检索的BERT预训练。很多数据集： 自己从网络采的大规模图文数据集LAIT,Conceptual Captions 和 SBU Captions数据集。多阶段不同数据集训练。
    2020年初的最好，超过Unicoder，UNITER
**Image captioning**: [XGPT: Cross-modal Generative Pre-Training for Image Captioning](https://arxiv.org/abs/2003.01473), arXiv 2020/03

**Visual Question Generation**: [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations](https://arxiv.org/abs/2002.10832), arXiv 2020/02

**Text-image retrieval**: [CROSS-PROBE BERT FOR EFFICIENT AND EFFECTIVE CROSS-MODAL SEARCH](https://openreview.net/forum?id=bW9SYKHcZiz), ICLR 2021 submission. 
    已被ICLR2021 拒绝。做搜索的。
**Chart VQA**: [STL-CQA: Structure-based Transformers with Localization and Encoding for Chart Question Answering](https://www.aclweb.org/anthology/2020.emnlp-main.264.pdf), EMNLP 2020.

## Other Analysis

**Multi-task Learning**, [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020, [[code]](https://github.com/facebookresearch/vilbert-multi-task) 

**Multi-task Learning**, [Unifying Vision-and-Language Tasks via Text Generation](https://arxiv.org/abs/2102.02779), arXiv 2021/02

**Social Bias in VL Embedding**, [Measuring Social Biases in Grounded Vision and Language Embeddings](https://arxiv.org/abs/2002.08911), arXiv 2020/02, [[code]](https://github.com/candacelax/bias-in-vision-and-language)

**In-depth Analysis**, [Are we pretraining it right? Digging deeper into visio-linguistic pretraining](https://arxiv.org/abs/2004.08744),

**In-depth Analysis**, [Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models](https://arxiv.org/abs/2005.07310), ECCV 2020 Spotlight

**Adversarial Training**, [Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/abs/2006.06195), NeurIPS 2020 Spotlight

**Adaptive Analysis**, [Adaptive Transformers for Learning Multimodal Representations](https://arxiv.org/abs/2005.07486), ACL SRW 2020

**Neural Architecture Search**, [Deep Multimodal Neural Architecture Search](https://arxiv.org/abs/2004.12070), arXiv 2020/04

**Dataset perspective**, [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918), arXiv 2021/02
   谷歌，暂时榜单的第一，从数据角度探索图文表示的下游任务。
  
# Video-based VL-PTMs

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766), ICCV 2019

[Learning Video Representations Using Contrastive Bidirectional Transformers](https://arxiv.org/abs/1906.05743), arXiv 2019/06, (**CBT**)

[M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787), arXiv 2019/08

[BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127), 	ICCV 2019 YouTube8M workshop, [[code]](https://github.com/hughshaoqz/3rd-Youtube8M-TM)

[Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog](https://arxiv.org/abs/2002.00163), AAAI2020 DSTC8 workshop

[Learning Spatiotemporal Features via Video and Text Pair Discrimination](https://arxiv.org/abs/2001.05691), arXiv 2020/01, (**CPD**), [[code]](https://github.com/MCG-NJU/CPD-Video)

[UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2002.06353), arXiv 2020/02

[ActBERT: Learning Global-Local Video-Text Representations](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.html), CVPR 2020

[HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training](https://arxiv.org/abs/2005.00200), EMNLP 2020

[Video-Grounded Dialogues with Pretrained Generation Language Models](https://arxiv.org/abs/2006.15319), ACL 2020

[Auto-captions on GIF: A Large-scale Video-sentence Dataset for Vision-language Pre-training](https://arxiv.org/abs/2007.02375), arXiv 2020/07

[Multimodal Pretraining for Dense Video Captioning](https://arxiv.org/pdf/2011.11760.pdf), arXiv 2020/11

[PARAMETER EFFICIENT MULTIMODAL TRANSFORMERS FOR VIDEO REPRESENTATION LEARNING](https://arxiv.org/pdf/2012.04124.pdf), arXiv 2020/12



# Speech-based VL-PTMs

[Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307), arXiv 2019/06

[Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924), arXiv 2019/09

[SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559), arXiv 2019/10

[vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453),  arXiv 2019/10

[Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912),  arXiv 2019/11

# Other Transformer-based multimodal networks

[Multi-Modality Cross Attention Network for Image and Sentence Matching](http://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.html), ICCV 2020
    快手，中科大 模型间cross-attention，一共四个transformers，

[MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning](https://arxiv.org/abs/2005.05402), ACL 2020
    USA, 腾讯ai
[History for Visual Dialog: Do we really need it?](https://arxiv.org/pdf/2005.07493.pdf), ACL 2020
    VQA任务 
[Cross-Modality Relevance for Reasoning on Language and Vision](https://arxiv.org/abs/2005.06035), ACL 2020
   密西根州立， Cross-Modality + transformer, 一共三个 transofmers, VQA任务， NLVR 任务

[ERNIE-ViL: Knowledge Enhanced Vision-Language Representations through Scene Graphs](https://arxiv.org/pdf/2006.16934.pdf), Mar 19, 2021
  百度, Scene Graph 知识图谱 + transformer; Object, attribute, relationship prediction, 

# Backbone research of Image-Language task

[Image Retrieval using Scene Graphs](https://openaccess.thecvf.com/content_cvpr_2015/papers/Johnson_Image_Retrieval_Using_2015_CVPR_paper.pdf),CVPR 2015
  斯坦福, Johnson



# Other Resources

* Two recent surveys on pretrained language models
  * [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271), arXiv 2020/03
  * [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278), arXiv 2020/03
* Other surveys about multimodal research
  * [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/abs/1907.09358), arXiv 2019
  * [Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019 
  * [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2018
  * [A Comprehensive Survey of Deep Learning for Image Captioning](https://arxiv.org/abs/1810.04020), ACM Computing Surveys 2018
* Other repositories of relevant reading list
  * [Pre-trained Languge Model Papers from THU-NLP](https://github.com/thunlp/PLMpapers)
  * [BERT-related Papers](https://github.com/tomohideshibata/BERT-related-papers)
  * [Reading List for Topics in Multimodal Machine Learning](https://github.com/pliang279/awesome-multimodal-ml)
  * [A repository of vision and language papers](https://github.com/sangminwoo/awesome-vision-and-language-papers)

