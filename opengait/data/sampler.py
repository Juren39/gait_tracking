import math
import random
import torch
import torch.utils.data as tordata

def random_sample_list(obj_list, k, with_replacement=False):
    """
    从 obj_list 中随机抽取 k 个元素，单机模式。
    如果 with_replacement=True，采用有放回采样，否则无放回采样。
    如果 obj_list 长度 < k 且无放回，则只返回 obj_list 的全部元素。
    """
    n = len(obj_list)
    if n == 0:
        return []
    if with_replacement:
        indices = [random.randint(0, n - 1) for _ in range(k)]
    else:
        if k >= n:
            indices = random.sample(range(n), n)
        else:
            indices = random.sample(range(n), k)
    return [obj_list[i] for i in indices]


class TripletSampler(tordata.sampler.Sampler):
    """
    原先带有分布式逻辑的 P x K 采样，现在只保留单机下的采样流程：
      - batch_size = (P, K)
      - 每次从 label_set 中随机抽 P 个 label，每个 label 再随机抽 K 个样本索引。
      - 可选地 shuffle 整个 batch。
    """

    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size  # (P, K)
        if len(self.batch_size) != 2:
            raise ValueError("batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

    def __iter__(self):
        """
        不断产出一个 batch 的索引列表，总大小 = P*K，单机下不再分割到多卡。
        """
        P, K = self.batch_size
        while True:
            sample_indices = []
            # 1. 随机抽 P 个行人 ID
            pid_list = random_sample_list(self.dataset.label_set, P, with_replacement=False)

            # 2. 对每个 pid 再随机抽 K 个索引
            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                chosen = random_sample_list(indices, K, with_replacement=False)
                sample_indices.extend(chosen)

            # 3. 可选地再次打乱
            if self.batch_shuffle:
                random.shuffle(sample_indices)

            yield sample_indices

    def __len__(self):
        return len(self.dataset)


class InferenceSampler(tordata.sampler.Sampler):
    """
    推理/测试时的采样器，单机下仅做顺序采样或一次性采样。
    不再做任何分布式切分或整除处理。
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = len(dataset)
        self.indices = list(range(self.size))

        # 如果需要补齐以满足整除 batch_size，可在此处理；否则直接使用全部 indices
        if self.batch_size != 1:
            complement_size = math.ceil(self.size / self.batch_size) * self.batch_size
            self.indices += self.indices[: (complement_size - self.size)]
            self.size = complement_size

        # 将所有索引按照 batch_size 分组
        self.batches = []
        for i in range(0, self.size, self.batch_size):
            self.batches.append(self.indices[i : i + self.batch_size])

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    """
    通用随机采样器，每次从整个数据集中随机抽取 batch_size 个索引。
    单机情况下不进行任何分布式同步或切分。
    """

    def __init__(self, dataset, batch_size, batch_shuffle):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        if not isinstance(self.batch_size, int):
            raise ValueError("batch_size should be int, not {}".format(batch_size))

    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            # 从中随机抽取 batch_size 个索引 (有放回采样)
            sample_indices = random_sample_list(indices_list, self.batch_size, with_replacement=True)
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


class BilateralSampler(tordata.sampler.Sampler):
    """
    BilateralSampler：原先在 GaitSSB 中用的双路采样器，每次采 batch_size 帧后翻倍 (x2)。
    单机版本不做分布式处理。
    """

    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size  # 若为 (P, K)，表示 P*K
        self.batch_shuffle = batch_shuffle

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        if isinstance(self.batch_size, (list, tuple)):
            real_batch_size = self.batch_size[0] * self.batch_size[1]
        else:
            real_batch_size = self.batch_size

        while True:
            # 如果不足一个 batch，重新洗牌从头开始
            if (count + 1) * real_batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            start = count * real_batch_size
            end = (count + 1) * real_batch_size
            sampled_indices = self.total_indices[start : end]
            count += 1

            # 进一步随机打乱
            sampled_indices = random_sample_list(sampled_indices, len(sampled_indices), with_replacement=False)

            # 翻倍
            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)