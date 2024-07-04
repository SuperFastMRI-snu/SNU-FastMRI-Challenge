python train.py \
  -b 1 \
  -e 5 \
  -l 0.001 \
  -p 3 \
  -f 0.5 \
  -r 50 \
  -i 200 \
  -n 'test_varnet2' \
  -t '/content/drive/MyDrive/Data/train/' \
  -v '/content/drive/MyDrive/Data/val/' \
  --cascade 4 \
  --chans 9 \
  --sens_chans 4 \
  --input-key 'kspace' \
  --target-key 'image_label' \
  --max-key 'max' \
  --seed 430


