from dataclasses import dataclass
import numpy as np

def ohc_decode(ohc):
    assert len(ohc) == 10
    for i in range(10):
        if ohc[i] == 1:
            return i
    raise ValueError

@dataclass
class MNIST:
    labels: np.ndarray
    images: np.ndarray

    def load(image_path: str, label_path: str, test=False):
        img_magic = [0,0,8,3,0,0,234,96,0,0,0,28,0,0,0,28]
        lbl_magic = [0,0,8,1,0,0,234,96]
        img_count = 60_000
        if test:
            img_magic = [0,0,8,3,0,0,39,16,0,0,0,28,0,0,0,28]
            lbl_magic = [0,0,8,1,0,0,39,16]
            img_count = 10_000

        image_idx = open(image_path, "rb").read()
        assert image_idx[:16] == bytes(img_magic)
        image_idx = image_idx[16:]
        pixel_data = np.copy(np.frombuffer(image_idx, dtype=np.uint8)).astype('f')
        pixel_data = pixel_data/255
        images = np.array_split(pixel_data, img_count)
        assert len(images[0]) == 28*28
        
        label_idx = open(label_path, "rb").read()
        assert label_idx[:8] == bytes(lbl_magic)
        labels = np.array(list(label_idx[8:]))
        ohc_enc = np.zeros((img_count, 10), dtype=np.float32)
        ohc_enc[np.arange(img_count), labels] = 1
        labels = ohc_enc

        assert len(labels) == len(images)
        return MNIST(labels, np.array(images))
    
def debug_print_ndarray(array: np.ndarray):
    print("Array Metadata:")
    print("---------------")
    print("Shape:          ", array.shape)
    print("Number of dims: ", array.ndim)
    print("Size:           ", array.size)
    print("Data type:      ", array.dtype)
    print("Item size:      ", array.itemsize, "bytes")
    print("Memory layout:  ", 'C_CONTIGUOUS' if array.flags['C_CONTIGUOUS'] else 'F_CONTIGUOUS' if array.flags['F_CONTIGUOUS'] else 'NON_CONTIGUOUS')
    print("Total bytes:    ", array.nbytes, "bytes")

TRAIN_DATA = MNIST.load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte")
TEST_DATA = MNIST.load("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", True)