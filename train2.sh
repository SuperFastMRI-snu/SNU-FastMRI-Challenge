python train.py \
  -b 1 \
  -e 1 \
  -l 0.001 \
  -r 50 \
  -i 200 \
  -n 'test_varnet2' \
  -t '/content/drive/MyDrive/Data/train/' \
  -v '/content/drive/MyDrive/Data/val/' \
  --cascade 1 \
  --chans 9 \
  --sens_chans 4 \
  --input-key 'kspace' \
  --target-key 'image_label' \
  --max-key 'max' \
  --seed 430


