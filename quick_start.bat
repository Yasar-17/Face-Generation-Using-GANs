@echo off
echo ========================================
echo  DCGAN Face Generation - Quick Start
echo ========================================
echo.

echo [1] Visualize real dataset (sanity check)
echo     python src\visualize_batch.py --data-dir "path\to\faces" --save data_preview.png
echo.

echo [2] Train the model
echo     python train.py --data-dir "path\to\faces" --epochs 25
echo.
echo     Outputs saved to:
echo       outputs\samples\epoch_XXXX.png    ^<-- Generated face grids per epoch
echo       outputs\checkpoints\*.pth         ^<-- Model checkpoints
echo.

echo [3] Evaluate trained model
echo     python eval.py --ckpt outputs\checkpoints\checkpoint_epoch_0024.pth --real-dir "path\to\faces"
echo.
echo     Outputs:
echo       interpolation.png                 ^<-- Latent space interpolation
echo       eval_outputs\generated\*.png      ^<-- Generated faces for FID
echo       FID score printed to console
echo.

echo [4] View training samples
echo     Open outputs\samples\ in File Explorer to see generated faces per epoch
echo.

echo [5] View a single checkpoint's output
echo     python eval.py --ckpt outputs\checkpoints\checkpoint_epoch_0024.pth --real-dir "path\to\faces" --num-samples 0 --save-interpolation interp.png
echo     Then open interp.png
echo.
pause
