import os
import pickle
import json
import os.path as osp
import torch.utils.data as tordata

class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training: bool):
        """
        Args:
            data_cfg (dict): 数据相关配置，包含 dataset_root, dataset_partition, cache, data_in_use 等
            training (bool): 是否为训练模式，决定加载训练集还是测试集
        """
        self.seqs_info = self.__dataset_parser(data_cfg, training)

        self.cache = data_cfg['cache']
        
        self.label_list = [seq[0] for seq in self.seqs_info]
        self.types_list = [seq[1] for seq in self.seqs_info]
        self.views_list = [seq[2] for seq in self.seqs_info]

        self.label_set = sorted(set(self.label_list))
        self.types_set = sorted(set(self.types_list))
        self.views_set = sorted(set(self.views_list))
        self.seqs_data = [None] * len(self)

        self.indices_dict = {label: [] for label in self.label_set}
        for idx, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(idx)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.seqs_info)

    def __getitem__(self, idx: int):
        """
        按索引获取单条序列数据和其 meta 信息
        Returns:
            tuple(data_list, seq_info)
        """
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        else:
            if self.seqs_data[idx] is None:
                data_list = self.__loader__(self.seqs_info[idx][-1])
                self.seqs_data[idx] = data_list
            else:
                data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        """
        把全部样本数据加载到内存
        """
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __loader__(self, paths):
        """
        从指定路径列表中加载 .pkl 文件，并检查文件内容长度一致性
        Args:
            paths (list): pkl 文件路径列表
        Returns:
            data_list (list): 所有 .pkl 文件的加载结果，按顺序组成的列表
        """
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if not pth.endswith('.pkl'):
                raise ValueError(f"只支持加载 .pkl 文件，但 {pth} 不是 .pkl 后缀！")
            with open(pth, 'rb') as f:
                content = pickle.load(f)
            data_list.append(content)
        first_len = len(data_list[0])
        for idx, data in enumerate(data_list):
            if len(data) != first_len:
                raise ValueError(
                    f'文件 {paths[idx]} 的长度 {len(data)} '
                    f'与 {paths[0]} 的长度 {first_len} 不一致！'
                )
            if len(data) == 0:
                raise ValueError(f'文件 {paths[idx]} 内容为空，至少需要一个元素！')
        return data_list

    def __dataset_parser(self, data_cfg, training: bool):
        """
        解析数据集分区文件 (JSON)，获取指定 (train/test) 的序列信息。
        Returns:
            list: 形如 [ [label, type, view, [pkl_paths]], ... ] 的列表
        """
        dataset_root = data_cfg['dataset_root']
        partition_path = data_cfg['dataset_partition']
        with open(partition_path, "rb") as f:
            dataset_partition = json.load(f)
        train_labels = dataset_partition["TRAIN_SET"]
        test_labels = dataset_partition["TEST_SET"]

        # 实际文件系统里的 label
        all_label_dirs = os.listdir(dataset_root)

        # 只保留在目录中实际存在的标签
        train_labels = [lab for lab in train_labels if lab in all_label_dirs]
        test_labels = [lab for lab in test_labels if lab in all_label_dirs]
        miss_labels = [lab for lab in all_label_dirs 
                          if lab not in (train_labels + test_labels)]

        # 根据 training 来决定加载哪个集合
        if training:
            selected_labels = train_labels
        else:
            selected_labels = test_labels

        seqs_info = self.__get_sequences_info(dataset_root, selected_labels, data_cfg)
        return seqs_info

    def __get_sequences_info(self, dataset_root, label_set, data_cfg):
        """
        遍历 label/type/view 三级目录，收集所有序列的 pkl 文件路径。
        """
        data_in_use = data_cfg.get('data_in_use', None)
        seqs_info_list = []
        for lab in label_set:
            label_path = osp.join(dataset_root, lab)
            for typ in sorted(os.listdir(label_path)):
                type_path = osp.join(label_path, typ)
                for vie in sorted(os.listdir(type_path)):
                    view_path = osp.join(type_path, vie)
                    pkl_files = sorted(os.listdir(view_path))
                    if not pkl_files:
                        print(f'[DEBUG] 在 {lab}-{typ}-{vie} 下未找到任何 pkl 文件')
                        continue
                    full_paths = [osp.join(view_path, fname) for fname in pkl_files]
                    if data_in_use is not None:
                        full_paths = [
                            p for p, use_flag in zip(full_paths, data_in_use) 
                            if use_flag
                        ]
                    if full_paths:
                        seqs_info_list.append([lab, typ, vie, full_paths])
                    else:
                        print(f'[DEBUG] 在 {lab}-{typ}-{vie} 下，没有符合 data_in_use 的 pkl 文件')
        return seqs_info_list