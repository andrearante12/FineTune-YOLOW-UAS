# Reading List: Open-Vocabulary Obstruction Detection for UAS Landing via Fine-Tuned YOLO-World

Compiled 2026-11-23. I checked every paper against its arXiv metadata or its publisher page. PDFs come only from arxiv.org and openaccess.thecvf.com. Each entry gives its filename, then the title, authors, venue and link, and then the section or claim of this paper it supports. BibTeX keys are in `reading_list.bib`.

## Read first

1. **Weng et al. 2025 (11)**: This paper fine-tunes YOLO-World v2 for UAV open-vocabulary detection on VisDrone and deploys it on a Jetson Orin NX. It is the closest prior work to ours, so we have to position against it in the Related Work section.
2. **Kumar et al. 2022, LP-FT (13)**: This is the theory behind our main finding. Full fine-tuning distorts pretrained features even when part of the model is frozen. It also motivates the lower-LR and partial-freezing future work.
3. **Wang et al. 2025, YOLOE (01)**: This is the direct successor to YOLO-World. Reviewers will ask why we did not use it.
4. **Ruis et al. 2025 (17)**: This paper shows that plain fine-tuning of an OV detector shrinks its vocabulary to the fine-tuned classes. It proposes a no-forgetting alternative (frozen weights, learned tokens). It supports our collapse result and our future work.
5. **Gupta et al. 2019, LVIS (24) + Gu et al. 2022, ViLD (25)**: These two set the standard OV evaluation protocol (LVIS AP_r). We need them for the planned LVIS evaluation and to explain why our 195-image benchmark is a complementary test.

## 1. Real-time / successor open-vocabulary detectors

- `01_ovdet_wang2025_yoloe.pdf`: **YOLOE: Real-Time Seeing Anything**. Wang, A. et al. ICCV 2025. https://arxiv.org/abs/2503.07465. Supports: Related Work and Future Work. YOLOE is the main YOLO-World successor and supports text, visual and prompt-free modes. Its re-parameterizable text prompts are an alternative that may forget less.
- `02_ovdet_ren2024_groundingdino15.pdf`: **Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection**. Ren, T. et al. arXiv 2024. https://arxiv.org/abs/2405.10300. Supports: Related Work and Deployment. Its Edge model reports more than 10 FPS on a Jetson Orin NX at 640x640 (75.2 FPS with TensorRT on a server GPU). That is a direct comparison point for our 9.4/12.5 FPS Jetson results.
- `03_ovdet_ren2024_dinox.pdf`: **DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding**. Ren, T. et al. arXiv 2024 (tech report). https://arxiv.org/abs/2411.14347. Supports: Related Work. It is the current high-accuracy end of open-world detection, which sets up the accuracy-versus-onboard-speed tradeoff.
- `04_ovdet_minderer2023_owlv2.pdf`: **Scaling Open-Vocabulary Object Detection (OWLv2)**. Minderer, M. et al. NeurIPS 2023. https://arxiv.org/abs/2306.09683. Supports: Related Work. It is a widely used ViT-based OV detector baseline. Its NanoOWL variants are what other papers run on Jetson.
- `05_ovdet_zhao2024_omdetturbo.pdf`: **Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head (OmDet-Turbo)**. Zhao, T. et al. arXiv 2024. https://arxiv.org/abs/2403.06892. Supports: Related Work. It is a real-time transformer competitor to YOLO-World.
- `06_ovdet_liu2024_yolouniow.pdf`: **YOLO-UniOW: Efficient Universal Open-World Object Detection**. Liu, L. et al. arXiv 2024. https://arxiv.org/abs/2412.20645. Supports: Related Work and Future Work. It extends YOLO-World to open-world detection, where unknown objects get flagged. That matches the goal of catching unlisted obstructions.
- `07_ovdet_liu2024_groundingdino.pdf`: **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**. Liu, S. et al. ECCV 2024. https://arxiv.org/abs/2303.05499. Supports: Related Work. It is the canonical open-set grounding detector. Most aerial OV work (LAE-DINO, MR-GDINO) builds on it.

## 2. Open-vocabulary detection on aerial / UAV imagery

- `08_aerial_wei2024_rtovad.pdf`: **RT-OVAD: Real-Time Open-Vocabulary Aerial Object Detection via Image-Text Collaboration** (earlier titled OVA-DETR). Wei, G. et al. arXiv 2024. https://arxiv.org/abs/2408.12246. Supports: Related Work. It is a real-time OV detector built for aerial imagery.
- `09_aerial_pan2025_laedino.pdf`: **Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community (LAE-DINO / LAE-1M)**. Pan, J. et al. AAAI 2025. https://arxiv.org/abs/2408.09110. Supports: Related Work. It provides large-scale aerial OV pretraining data. That is an alternative to fine-tuning on VisDrone's 10 classes alone.
- `10_aerial_tsourveloudis2026_ovd-aerial-transfer.pdf`: **Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation**. Tsourveloudis, C. arXiv 2026 (single-author preprint). https://arxiv.org/abs/2601.22164. Supports: Introduction and Motivation. Zero-shot OV detectors degrade badly on aerial data (F1 0.53 on DIOR, 0.12 on FAIR1M), which is why domain fine-tuning is needed at all.
- `11_aerial_weng2025_uav-ovd-lightweight.pdf`: **Light-Weight Cross-Modal Enhancement Method with Benchmark Construction for UAV-based Open-Vocabulary Object Detection**. Weng, Z. et al. arXiv 2025. https://arxiv.org/abs/2509.06011. Supports: Related Work and Deployment. It builds on YOLO-World v2, improves zero-shot VisDrone mAP by 5.3, introduces the UAVDE-2M/UAVCAP-15K benchmarks, and runs on a Jetson Orin NX.
- `12_aerial_chhipa2024_ovd-robustness-shift.pdf`: **Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts**. Chhipa, P. C. et al. ECCV Workshops 2024. https://arxiv.org/abs/2405.14874. Supports: Discussion. OV detectors such as OWL-ViT, YOLO-World and Grounding DINO are fragile under distribution shift.

## 3. Forgetting and robust fine-tuning of vision-language / OV models

- `13_ft_kumar2022_lpft.pdf`: **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (LP-FT)**. Kumar, A. et al. ICLR 2022. https://arxiv.org/abs/2202.10054. Supports: Discussion of why open-vocab ability collapses with more epochs. Full fine-tuning distorts features, and linear probing before fine-tuning mitigates it. Supports the lower-LR and staged-freezing future work.
- `14_ft_goyal2023_flyp.pdf`: **Finetune like you pretrain: Improved finetuning of zero-shot vision models (FLYP)**. Goyal, S. et al. CVPR 2023. https://arxiv.org/abs/2212.00638. Supports: Future Work. Fine-tuning with the contrastive pretraining loss preserves zero-shot robustness better.
- `15_ft_zheng2023_zscl.pdf`: **Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models (ZSCL)**. Zheng, Z. et al. ICCV 2023. https://arxiv.org/abs/2303.06628. Supports: Discussion. Zero-shot transfer degrades with downstream training. Distillation and weight-ensemble remedies exist.
- `16_ft_hu2022_lora.pdf`: **LoRA: Low-Rank Adaptation of Large Language Models**. Hu, E. J. et al. ICLR 2022. https://arxiv.org/abs/2106.09685. Supports: Future Work (partial freezing / parameter-efficient fine-tuning). It is the standard PEFT reference.
- `17_ft_ruis2025_textual-inversion-ovd.pdf`: **Textual Inversion for Efficient Adaptation of Open-Vocabulary Object Detectors Without Forgetting**. Ruis, F. et al. arXiv 2025. https://arxiv.org/abs/2508.05323. Supports: Results and Discussion. After plain fine-tuning an OV detector is limited to the fine-tuned classes, which matches what we observe. It also offers a frozen-weights alternative.
- `18_ft_dong2024_mrgdino.pdf`: **MR-GDINO: Efficient Open-World Continual Object Detection**. Dong, B. et al. arXiv 2024. https://arxiv.org/abs/2412.15979. Supports: Future Work (separate detection head / memory). It keeps unseen-class ability while adapting to new classes.

## 4. Vision-based safe landing / landing-zone hazard detection (2023-2026)

- `19_land_bong2023_dovesei.pdf`: **Dynamic Open Vocabulary Enhanced Safe-landing with Intelligence (DOVESEI)**. Bong, H. M. et al. IROS 2023 Workshop (Last-Mile Robotics). https://arxiv.org/abs/2308.11471. Supports: Related Work. It is the earliest open-vocabulary (CLIPSeg) safe-landing system, which makes it a direct precedent for our approach.
- `20_land_bong2025_peace.pdf`: **PEACE: Prompt Engineering Automation for CLIPSeg Enhancement for Safe-Landing Zone Segmentation**. Bong, H. M. et al. IROS 2025. https://arxiv.org/abs/2310.00085. Supports: Related Work and Future Work. Automated prompt engineering for landing-zone segmentation is a precedent for VLM-generated prompts.
- `21_land_cai2025_llmland.pdf`: **LLM-Land: Large Language Models for Context-Aware Drone Landing**. Cai, S. et al. arXiv 2025. https://arxiv.org/abs/2505.06399. Supports: Related Work and Future Work. It uses LLM/VLM reasoning for landing decisions.
- `22_land_delatorre2025_risk-aware-landing.pdf`: **Vision-Based Risk Aware Emergency Landing for UAVs in Complex Urban Environments**. de la Torre-Vanegas, J. et al. arXiv 2025. https://arxiv.org/abs/2505.20423. Supports: Related Work. It is a closed-set semantic-segmentation baseline for urban landing-risk assessment, the approach that open-vocab methods generalize.
- `23_land_tan2025_vislanding.pdf`: **VisLanding: Monocular 3D Perception for UAV Safe Landing via Depth-Normal Synergy**. Tan, Z. et al. IROS 2025. https://arxiv.org/abs/2506.14525. Supports: Related Work. It is a geometric (depth/normal) safe-landing method that complements semantic obstruction detection.

## 5. LVIS and the standard OV evaluation protocol

- `24_eval_gupta2019_lvis.pdf`: **LVIS: A Dataset for Large Vocabulary Instance Segmentation**. Gupta, A. et al. CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.html. Supports: Future Work (LVIS evaluation). This PDF is the CVPR version.
- `25_eval_gu2022_vild.pdf`: **Open-vocabulary Object Detection via Vision and Language Knowledge Distillation (ViLD)**. Gu, X. et al. ICLR 2022. https://arxiv.org/abs/2104.13921. Supports: Evaluation methodology. It defined the LVIS base/novel (AP_r) protocol that YOLO-World and its successors report.

## 6. Edge deployment of (OV) detectors on Jetson

(See also entries 02 and 11, which both report Jetson Orin NX speeds.)

- `26_edge_alqahtani2024_edge-detector-eval.pdf`: **A Comprehensive Evaluation of Deep Learning Object Detection Models on Heterogeneous Edge Devices**. Alqahtani, D. K. et al. arXiv 2024. https://arxiv.org/abs/2409.16808. Supports: Deployment. It benchmarks YOLOv8 accuracy, latency and energy on a Jetson Orin Nano and other devices, which gives context for our PyTorch FPS and the expected TensorRT speedup.
- *(not downloaded: the publisher's domain is not on the allowed list, but it is open access)* **Real-time open-vocabulary perception for mobile robots on edge devices: a systematic analysis of the accuracy-latency trade-off**. Park, J. et al. Frontiers in Robotics and AI 12:1693988, 2025. https://doi.org/10.3389/frobt.2025.1693988. Supports: Deployment. It runs YOLO-World and NanoOWL with TensorRT FP32/FP16 on a Jetson AGX Orin, and YOLO-World was slower than NanoOWL. That is directly relevant to our FPS discussion.

## Notes
- Paywalled papers: none were skipped. The Frontiers paper above is open access but was not downloaded because of the domain restriction. Get it at the DOI if you want it.
- Venue caveats: entries 02, 03, 05, 06, 08, 11, 17, 18, 21, 22 and 26 are arXiv preprints, and I found no confirmed venue. Check for a published version before you cite them.
