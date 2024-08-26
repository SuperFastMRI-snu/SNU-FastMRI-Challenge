import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import time
import torch
import random
from utils.mraugment.data_augment import DataAugmentor
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import apply_mask

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, DataAugmentor, args, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        # For MRAugment
        self.DataAugmentor = DataAugmentor
        # For random mask
        # if not forward:
        self.mask_type = args.mask_type
        self.center_fractions = args.center_fractions

        #if not forward:
        # mask list 만들기
        mask_acc = [args.acc] # [4, 5, 6, 7, 8, 9]
        mask_list = {}

        data = torch.randn(16, 768, 396, 2)
        mask_func = create_mask_for_mask_type("equispaced", [0.08], [5])
        _, mask = apply_mask(data, mask_func, None) # mask.shape = [1, 1, ?, 1]
        mask = np.array(torch.squeeze(mask))
        self.mask = mask
        print(mask)

        """# data에서 중요한 것은 [-2] 위치의 원소. mask 벡터의 크기가 바로 [-2] 원소이다.
        data_list = [torch.randn(16, 768, 396, 2), torch.randn(16, 396, 768, 2), torch.randn(4, 768, 392, 2)]
        for acc in mask_acc:
          for data in data_list:
            mask_func = create_mask_for_mask_type(self.mask_type, self.center_fractions, [acc])
            _, mask = apply_mask(data, mask_func, None) # mask.shape = [1, 1, ?, 1]
            print(f"{acc},{mask.shape}")
            mask = np.array(torch.squeeze(mask))
            #print(f"{acc},{mask.shape}")
            print(mask)
            mask_list[(acc, data.shape[-2])] = mask # 마스크를 찾을 때에는 acc와 input의 열의 개수로 찾아야 함
        self.mask_list = mask_list"""

        # @@@데이터 절반 만들기 시작(다시 바꿔줄 경우, 그냥 주석 붙여 놓은 코드 지우기
        if not forward:
            image_files = list(Path(root / "image").iterdir())

            # train data의 image_examples는 txt파일에서 가지고 오도록 함(google colab용)
            files_num = len(image_files)
            if files_num==51:
                # val data인 경우 원래처럼 image_examples 가지고 오기
                for fname in sorted(image_files):
                    num_slices = self._get_metadata(fname)
                    self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            else:
                self.image_examples = self._get_metadata2('image')


            """
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                if self.DataAugmentor != None: # train 하는 경우
                    self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
                    #@@@데이터 절반 만들기 self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
                else: # val 하는 경우
                    self.image_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            """

        kspace_files = list(Path(root / "kspace").iterdir())
        
        for fname in sorted(kspace_files):
            # train data의 kspace_examples는 txt파일에서 가지고 오도록 함(google colab용)
            files_num = len(kspace_files)
            if files_num==51:
                # val data인 경우 원래처럼 kspace_examples 가지고 오기
                num_slices = self._get_metadata(fname)
                # @@@아래 코드는 NAFNet 실험을 위해 쓴 것으로 지우고 맨 아래 주석 해제해야 함
                if self.DataAugmentor != None:
                    for slice_ind in range(num_slices):
                        if slice_ind < 51/2: # @@@데이터 절반 만들기
                            self.kspace_examples.append(tuple((Path(fname), int(slice_ind), args.acc[0])))
                        else: # @@@데이터 절반 만들기
                            self.kspace_examples.append(tuple((Path(fname), int(slice_ind), args.acc[1])))
                else: 
                    self.kspace_examples += [(fname, slice_ind, args.acc) for slice_ind in range(num_slices)]
                # self.kspace_examples += [(fname, slice_ind, args.acc) for slice_ind in range(num_slices)]
            elif self.DataAugmentor != None:
                # train data인 경우
                self.kspace_examples = self._get_metadata2('kspace', args)
            else:
                # evaluation인 경우
                num_slices = self._get_metadata(fname)
                # acc 6,7 test
                self.kspace_examples += [(fname, slice_ind, args.acc) for slice_ind in range(num_slices)]
                #self.kspace_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            """
            num_slices = self._get_metadata(fname)
            if not self.forward:
                if self.DataAugmentor != None: # train 하는 경우
                    if file_idx < len_kspace_files/2: # @@@데이터 절반 만들기
                        self.kspace_examples += [(fname, slice_ind, args.acc[0]) for slice_ind in range(num_slices)]
                    else: # @@@데이터 절반 만들기
                        self.kspace_examples += [(fname, slice_ind, args.acc[1]) for slice_ind in range(num_slices)]
                else: # val 하는 경우
                    self.kspace_examples += [(fname, slice_ind, args.acc) for slice_ind in range(num_slices)]
            else: # eval 하는 경우
                self.kspace_examples += [(fname, slice_ind) for slice_ind in range(num_slices)]
            file_idx+=1 # @@@데이터 절반 만들기
            """

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def _get_metadata2(self, data_type, args=None):
        examples = []
        if data_type == 'image':
            with open("/content/drive/MyDrive/Data/train_image_examples_mini.txt", "r") as f:
                image_examples = f.read()
                for line in image_examples.split('\n'):
                    fname, dataslice = line.split()
                    examples.append(tuple((Path(fname), int(dataslice))))
        elif data_type == 'kspace':
            with open("/content/drive/MyDrive/Data/train_kspace_examples_mini.txt", "r") as f:
                kspace_examples = f.read()
                file_idx = 0
                len_kspace_files = 120
                for line in kspace_examples.split('\n'):
                    fname, dataslice = line.split()
                    if file_idx < len_kspace_files/2: # @@@데이터 절반 만들기
                        examples.append(tuple((Path(fname), int(dataslice), args.acc[0])))
                    else: # @@@데이터 절반 만들기
                        examples.append(tuple((Path(fname), int(dataslice), args.acc[1])))
                    file_idx += 1
        return examples

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward: # train, val 하는 경우
            image_fname, _ = self.image_examples[i]

        if self.forward: # eval 하는 경우
            kspace_fname, dataslice, acc = self.kspace_examples[i]
            # kspace_fname, dataslice = self.kspace_examples[i]
        else:
            if self.DataAugmentor != None: # train 하는 경우
                kspace_fname, dataslice, args_acc = self.kspace_examples[i]
            else: # val 하는 경우
                kspace_fname, dataslice, args_acc_list = self.kspace_examples[i]
                args_acc = args_acc_list[round(torch.rand(1).item())]
                del args_acc_list
        
        if not self.forward:
            # image_file을 열어서 target_size 가져오기
            with h5py.File(image_fname, "r") as hf:
                target_size = hf[self.target_key][dataslice].shape

            # kspace_fname에서 acc 정보 가져오기(mask 만들 때 사용)
            str_kspace_fname = str(kspace_fname)
            acc = int(str_kspace_fname.split('_')[1][-1])

        # 파일 열어서 kspace, image 가져온 후 augment하기
        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]

            # DataAugmentor에 들어가는 input은 마지막 차원이 실수부와 허수부로 나뉘어져 있어야 한다.
            input = torch.from_numpy(input)
            input = torch.stack((input.real, input.imag), dim=-1)

            # augment된 kspace를 input으로 받기 / 그에 대응되는 target도 미리 받아두기 / random mask를 위한 p 설정
            target = None
            if self.DataAugmentor != None:
              input, target = self.DataAugmentor(input, [target_size[-2],target_size[-1]]) # return 된 input.shape[-1]는 2이다. 실수부와 허수부로 나뉘어져 있다.

            # random mask 항상 적용. Test 때는 적용 x
            if not self.forward: # train, val 하는 경우
                if args_acc == acc and input.shape[-2] != 768:
                    mask =  np.array(hf["mask"])
                else:
                    mask = self.mask_list[(args_acc, input.shape[-2])]
            else: # eval 하는 경우
                mask = self.mask_list[(acc, input.shape[-2])]
                #mask =  np.array(hf["mask"])

            if self.forward:
                target = -1
                attrs = -1
            else:
                with h5py.File(image_fname, "r") as hf:
                    attrs = dict(hf.attrs)
                    if target == None:
                      target = hf[self.target_key][dataslice]
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, DataAugmentor=None, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        DataAugmentor =  DataAugmentor,
        args = args
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
