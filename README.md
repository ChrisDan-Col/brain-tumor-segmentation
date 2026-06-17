# Brain Tumor Segmentation with SegFormer (Transformer-based Semantic Segmentation)

This project fine-tunes a **SegFormer** vision transformer for binary semantic segmentation of brain tumors in MRI scans, using an H5 dataset of paired MRI images and ground-truth tumor masks.

## What It Does

All work is in `Brain_Tumor_Semantic_Segmentation_v5.ipynb`:

### Data Loading and Preprocessing
Reads MRI images and corresponding tumor masks from `brain_tumor_dataset.h5` (HDF5 format). Preprocessing per sample:
- **Percentile normalisation** (1st–99th percentile, computed on non-zero/brain tissue pixels only) to suppress background bias.
- **Median blur denoising** for salt-and-pepper noise reduction.
- Resize to 512×512 pixels.
- **ImageNet normalisation** (mean/std) to match SegFormer pretraining.
- **Data augmentation** during training: random horizontal flips and 90°/180°/270° rotations.

### Model
**SegFormer (nvidia/mit-b5)** — a hierarchical vision transformer with a lightweight MLP decoder, fine-tuned for binary segmentation (background vs. tumor). Uses pretrained Mix Transformer (MiT-B5) weights and a newly initialised decode head with 2 output classes.

### Training
- **Loss**: Combo loss = 0.5 × Cross-Entropy + 0.5 × Dice Loss (handles class imbalance between background and tumor pixels).
- **Optimiser**: AdamW (lr = 6e-5, weight decay = 1e-4).
- **Scheduler**: ReduceLROnPlateau (halves LR on plateau, patience = 3 epochs).
- **Checkpointing**: saves `best_model.pth` (best validation Dice) and `last_checkpoint.pth` for resuming.
- 70/20/10 train/val/test split with stratified random partitioning.

### Evaluation
Reports Dice Score (F1), IoU (Jaccard), Precision, and Recall on the test set. Final test performance: **Dice ≈ 0.816, IoU ≈ 0.694**.

Generates side-by-side visual comparisons: original MRI / ground truth mask / predicted mask.

## Data

- **`brain_tumor_dataset.h5`** — HDF5 file containing 3064 brain MRI images with paired tumor segmentation masks. Each record structured as `patient_id/cjdata/{image, tumorMask}`.

## Libraries

`torch`, `transformers` (Hugging Face SegFormer), `h5py`, `opencv-python`, `sklearn`, `tqdm`, `matplotlib`

---

**Module:** Signal and Image Processing (CMP9780) — University of Lincoln, School of Computer Science  
**Submitted as:** Assessment 2 | Academic Year 2025/2026
