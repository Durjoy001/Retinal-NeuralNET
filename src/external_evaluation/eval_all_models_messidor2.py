# -*- coding: utf-8 -*-
"""
Batch evaluation script: Evaluate all trained models on Messidor-2 dataset.

This script runs eval_messidor2.py for all available models
(CNN, ViT, Hybrid, VLM) in a single execution.

All models use:
- DR head from multilabel classifier
- Fixed RFMiD validation threshold (no threshold adaptation)

Usage:
    python -m src.external_evaluation.eval_all_models_messidor2 \
        --rfmid_train_csv data/RFMiD_Challenge_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv \
        --messidor_csv /path/to/messidor2_labels.csv \
        --images_dir /path/to/messidor2/images \
        --results_base_dir results/External/Messidor2 \
        [--skip_models model1 model2] \
        [--only_models model1 model2]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Model configurations: (model_name, checkpoint_path, thresholds_path, results_subdir)
MODEL_CONFIGS = [
    # CNNs
    ("densenet121", "results/CNN/Densenet121/densenet121_rfmid_best.pth", 
     "results/CNN/Densenet121/optimal_thresholds.npy", "Densenet121"),
    ("resnet50", "results/CNN/ResNet50/resnet50_rfmid_best.pth",
     "results/CNN/ResNet50/optimal_thresholds.npy", "ResNet50"),
    ("efficientnet_b3", "results/CNN/EfficientNetB3/efficientnet_b3_rfmid_best.pth",
     "results/CNN/EfficientNetB3/optimal_thresholds.npy", "EfficientNetB3"),
    ("inception_v3", "results/CNN/InceptionV3/inception_v3_rfmid_best.pth",
     "results/CNN/InceptionV3/optimal_thresholds.npy", "InceptionV3"),
    
    # ViTs
    ("swin_tiny", "results/ViT/SwinTiny/swin_tiny_rfmid_best.pth",
     "results/ViT/SwinTiny/optimal_thresholds.npy", "SwinTiny"),
    ("vit_small", "results/ViT/ViTSmall/vit_small_rfmid_best.pth",
     "results/ViT/ViTSmall/optimal_thresholds.npy", "ViTSmall"),
    ("deit_small", "results/ViT/DeiTSmall/deit_small_rfmid_best.pth",
     "results/ViT/DeiTSmall/optimal_thresholds.npy", "DeiTSmall"),
    ("crossvit_small", "results/ViT/CrossViTSmall/crossvit_small_rfmid_best.pth",
     "results/ViT/CrossViTSmall/optimal_thresholds.npy", "CrossViTSmall"),
    
    # Hybrids
    ("coatnet0", "results/Hybrid/CoAtNet0/coatnet0_rfmid_best.pth",
     "results/Hybrid/CoAtNet0/optimal_thresholds.npy", "CoAtNet0"),
    ("maxvit_tiny", "results/Hybrid/MaxViTTiny/maxvit_tiny_rfmid_best.pth",
     "results/Hybrid/MaxViTTiny/optimal_thresholds.npy", "MaxViTTiny"),
    
    # VLMs (optional - may not exist)
    ("clip_vit_b16", "results/VLM/CLIPViTB16/clip_vit_b16_rfmid_best.pth",
     "results/VLM/CLIPViTB16/optimal_thresholds.npy", "CLIPViTB16"),
    ("siglip_base_384", "results/VLM/SigLIPBase384/siglip_base_384_rfmid_best.pth",
     "results/VLM/SigLIPBase384/optimal_thresholds.npy", "SigLIPBase384"),
]

def main():
    ap = argparse.ArgumentParser(description="Batch evaluate all models on Messidor-2")
    ap.add_argument("--rfmid_train_csv", required=True,
                    help="RFMiD training labels CSV")
    ap.add_argument("--messidor_csv", required=True,
                    help="Messidor-2 labels CSV")
    ap.add_argument("--images_dir", required=True,
                    help="Messidor-2 images directory")
    ap.add_argument("--results_base_dir", required=True,
                    help="Base directory for results (e.g., results/External/Messidor2)")
    ap.add_argument("--skip_models", nargs="+", default=[],
                    help="Model names to skip (e.g., --skip_models inception_v3 clipvitb16)")
    ap.add_argument("--only_models", nargs="+", default=[],
                    help="Only evaluate these models (e.g., --only_models resnet50 vit_small)")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Batch size for inference")
    ap.add_argument("--use_temperature_scaling", action="store_true",
                    help="Apply temperature scaling calibration on validation split")
    ap.add_argument("--temp_scaling_subset", type=int, default=500,
                    help="Use subset of training data for temperature scaling (faster, default: 500)")
    ap.add_argument("--adapt_bn", action="store_true",
                    help="Adapt BatchNorm statistics on Messidor-2 (no labels needed)")
    ap.add_argument("--disable_clahe", action="store_true",
                    help="Disable CLAHE preprocessing (not used - models use training-aligned transforms)")
    ap.add_argument("--no_referable_dr", action="store_true",
                    help="Disable referable DR metrics for all models")
    args = ap.parse_args()
    
    # Resolve paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    rfmid_csv = root_dir / args.rfmid_train_csv
    messidor_csv = Path(args.messidor_csv).resolve()
    images_dir = Path(args.images_dir).resolve()
    results_base = root_dir / args.results_base_dir
    
    # Validate inputs
    if not rfmid_csv.exists():
        print(f"❌ Error: RFMiD CSV not found: {rfmid_csv}")
        sys.exit(1)
    if not messidor_csv.exists():
        print(f"❌ Error: Messidor CSV not found: {messidor_csv}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Filter models
    models_to_run = []
    for model_name, ckpt_path, thr_path, subdir in MODEL_CONFIGS:
        # Check if model should be skipped
        if args.skip_models and model_name.lower() in [m.lower() for m in args.skip_models]:
            print(f"⏭️  Skipping {model_name} (in --skip_models)")
            continue
        
        # Check if only specific models requested
        if args.only_models and model_name.lower() not in [m.lower() for m in args.only_models]:
            continue
        
        # Check if checkpoint exists
        full_ckpt = root_dir / ckpt_path
        full_thr = root_dir / thr_path
        if not full_ckpt.exists():
            print(f"⚠️  Warning: Checkpoint not found for {model_name}: {full_ckpt}")
            print(f"   Skipping {model_name}")
            continue
        if not full_thr.exists():
            print(f"⚠️  Warning: Thresholds not found for {model_name}: {full_thr}")
            print(f"   Skipping {model_name}")
            continue
        
        models_to_run.append((model_name, ckpt_path, thr_path, subdir))
    
    if not models_to_run:
        print("❌ Error: No models to evaluate. Check your --skip_models/--only_models filters and checkpoint paths.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Batch Evaluation on Messidor-2 Dataset")
    print(f"{'='*70}")
    print(f"Total models to evaluate: {len(models_to_run)}")
    print(f"RFMiD CSV: {rfmid_csv}")
    print(f"Messidor CSV: {messidor_csv}")
    print(f"Images dir: {images_dir}")
    print(f"Results base: {results_base}")
    print(f"{'='*70}\n")
    
    # Run evaluation for each model
    results = []
    for idx, (model_name, ckpt_path, thr_path, subdir) in enumerate(models_to_run, 1):
        print(f"\n[{idx}/{len(models_to_run)}] Evaluating {model_name}...")
        print("-" * 70)
        
        results_dir = results_base / subdir
        full_ckpt = root_dir / ckpt_path
        full_thr = root_dir / thr_path
        
        # Build command
        cmd = [
            sys.executable, "-m", "src.external_evaluation.eval_messidor2",
            "--model_name", model_name,
            "--checkpoint", str(full_ckpt),
            "--thresholds", str(full_thr),
            "--rfmid_train_csv", str(rfmid_csv),
            "--messidor_csv", str(messidor_csv),
            "--images_dir", str(images_dir),
            "--results_dir", str(results_dir),
            "--batch_size", str(args.batch_size),
        ]
        
        if args.use_temperature_scaling:
            cmd.append("--use_temperature_scaling")
            if args.temp_scaling_subset:
                cmd.extend(["--temp_scaling_subset", str(args.temp_scaling_subset)])
        
        if args.adapt_bn:
            cmd.append("--adapt_bn")
        
        if args.disable_clahe:
            cmd.append("--disable_clahe")
        
        if args.no_referable_dr:
            cmd.append("--no_referable_dr")
        
        # Run evaluation (stream output in real-time)
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"\n✅ {model_name} completed successfully")
            results.append((model_name, True, None))
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {model_name} failed with error code {e.returncode}")
            results.append((model_name, False, f"Exit code {e.returncode}"))
        except Exception as e:
            print(f"\n❌ {model_name} failed with exception: {e}")
            results.append((model_name, False, str(e)))
    
    # Summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    successful = [m for m, success, _ in results if success]
    failed = [m for m, success, _ in results if not success]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for model in successful:
        print(f"   - {model}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for model, _, error in results:
            if not any(m == model for m, s, _ in results if s):
                print(f"   - {model}: {error}")
    
    print(f"\n{'='*70}")
    print(f"Results saved under: {results_base}")
    print(f"{'='*70}\n")
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()

