import pathlib

import lmdb
from torch.utils import data


class LMDBDataset(data.Dataset):
    def __init__(self, lmdb_path, name="", train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return num_samples(self.name, self.train)
