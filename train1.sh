python train.py \
  -b 1 \
  -e 10 \
  -l 0.001 \
  -r 20 \
  -i 100 \
  -n 'test_varnet_mraug' \
  -t '/content/drive/MyDrive/Data/train/' \
  -v '/content/drive/MyDrive/Data/val/' \
  --cascade 4 \
  --chans 8 \
  --sens_chans 6 \
  --input-key 'kspace' \
  --target-key 'image_label' \
  --max-key 'max' \
  --seed 430


