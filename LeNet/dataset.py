import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype


class Mnist_data:
    def __init__(self):
        super(Mnist_data, self).__init__()

    def create_dataset(data_path, batch_size=32, repeat_size=1,
                       num_parallel_workers=1):
        # 定义数据集
        mnist_ds = ds.MnistDataset(data_path)
        resize_height, resize_width = 32, 32
        rescale = 1.0 / 255.0
        shift = 0.0
        rescale_nml = 1 / 0.3081
        shift_nml = -1 * 0.1307 / 0.3081

        # 定义所需要操作的map映射
        resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
        rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
        rescale_op = CV.Rescale(rescale, shift)
        hwc2chw_op = CV.HWC2CHW()
        type_cast_op = C.TypeCast(mstype.int32)

        # 使用map映射函数，将数据操作应用到数据集
        mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
        mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
        mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
        mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
        mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

        # 进行shuffle、batch操作
        buffer_size = 10000
        mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
        mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

        return mnist_ds