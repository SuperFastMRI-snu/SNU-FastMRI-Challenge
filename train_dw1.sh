python train.py \
  -b 1 \
  -e 10 \
  -l 0.001 \
  -p 3 \
  -f 0.5 \
  -r 20 \
  -i 100 \
  -n 'test_varnet' \
  -t '/content/drive/MyDrive/Data/val/' \
  -v '/content/drive/MyDrive/Data/val/' \
  --cascade 1 \
  --chans 9 \
  --sens_chans 4 \
  --input-key 'kspace' \
  --target-key 'image_label' \
  --max-key 'max' \
  --seed 430


