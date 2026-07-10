# Exp: prune NIST 5k + PFAS ext 5k (requires conda env `graff` with CUDA PyG)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$Pkl = if ($env:PKL) { $env:PKL } else { "data/pfas/nist_标注/nist_pfas_annot.pkl" }
$NistCkpt = if ($env:NIST_CKPT) { $env:NIST_CKPT } else { "lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt" }
$OutDir = if ($env:OUT_DIR) { $env:OUT_DIR } else { "output/exp_pruned_union_5k5k" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "64" }
$MaxEpochs = if ($env:MAX_EPOCHS) { $env:MAX_EPOCHS } else { "50" }
$Gpus = if ($env:GPUS) { $env:GPUS } else { "1" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== [0/4] vocab preview ==="
python scripts/preview_pruned_vocab.py $Pkl --checkpoint $NistCkpt `
  --nist_vocab_keep 5000 --pfas_extension_size 5000 | Tee-Object -FilePath "$OutDir/vocab_preview.log"

if ($env:SKIP_PREPROCESS -ne "1") {
  Write-Host "=== [1/4] preprocess ==="
  python preprocess-pfas.py --no-parallel
}

Write-Host "=== [2/4] train ==="
python train-graff-ms.py $Pkl `
  --dataset pfas `
  --checkpoint $NistCkpt `
  --vocab_mode union `
  --nist_vocab_keep 5000 `
  --pfas_extension_size 5000 `
  --transfer_mode clf_map `
  --batch_size $BatchSize `
  --max_epochs $MaxEpochs `
  --gpus $Gpus `
  --num_workers 8 2>&1 | Tee-Object -FilePath "$OutDir/train.log"

$Ckpt = Get-ChildItem -Path "lightning_logs/graff/version_*/checkpoints/*.ckpt" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 -ExpandProperty FullName
Write-Host "Best checkpoint: $Ckpt"
Set-Content -Path "$OutDir/best_ckpt.txt" -Value $Ckpt

Write-Host "=== [3/4] predict ==="
python run-graff-ms.py $Ckpt "data/pfas/nist_标注/test_queries.tsv" "$OutDir/pred.msp" `
  --has_isotopes 1 --gpus $Gpus 2>&1 | Tee-Object -FilePath "$OutDir/predict.log"

Write-Host "=== [4/4] cosine ==="
python cosine-similarity.py "$OutDir/pred.msp" "data/pfas/nist_标注/test.msp" --matchms_tol 0.1 `
  2>&1 | Tee-Object -FilePath "$OutDir/cosine.log"

Write-Host "Done. Results in $OutDir/"
