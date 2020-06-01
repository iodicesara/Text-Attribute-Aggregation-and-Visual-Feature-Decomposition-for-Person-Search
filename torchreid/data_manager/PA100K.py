from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import re
import os.path as osp
import numpy as np


class PA100K(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    dataset_dir = 'PA-100K'

    def __init__(self, root='data', verbose=True,random_label=1,self_attribute_path=None,attribute_path_bin=None,
                 attribute_path=None,arch=None,is_frame=False,test_attribute_path=None,tr_id_all=4000,**kwargs):

        super(PA100K, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'release_data')
        self.query_dir = osp.join(self.dataset_dir, 'release_data')
        self.gallery_dir = osp.join(self.dataset_dir, 'release_data')

        self.attribute_path_market=attribute_path
        self._check_before_run()
        self.random_label=random_label
        self.from_att_to_ID_tr = {}
        self.from_att_to_ID_test = {}

        attribute_data, attribute_data_self, attribute_data_bin = None, None, None

        if attribute_path is not None:
            attribute_data = np.load(attribute_path).item()

        if test_attribute_path is not None:
            test_attribute_data = np.load(test_attribute_path).item()

        if self_attribute_path is not None:
            attribute_data_self = np.load(self_attribute_path).item()

        if attribute_path_bin is not None:
            attribute_data_bin = np.load(attribute_path_bin).item()

        print('training')

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir,relabel=True,\
                                                                   attribute_data=attribute_data,\
                                                                   attribute_data_self=attribute_data_self,\
                                                                   attribute_data_bin=attribute_data_bin,
                                                                   arch=arch,is_frame=is_frame,start_pid=0,
                                                                   end_pid=80000,training_id=tr_id_all)

        print('testing')

        if arch=='resnetAttW2VAttributes':
            query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir,relabel=False,\
                                                                  attribute_data=attribute_data,\
                                                                  attribute_data_self=attribute_data_self,\
                                                                  attribute_data_bin=attribute_data_bin,
                                                                  arch=arch,from_att_to_ID=self.from_att_to_ID_test,
                                                                  is_frame=is_frame,start_pid=90001,
                                                                  end_pid=100000,training_id=10000)

            gallery, num_gallery_pids, num_gallery_imgs =self._process_dir(self.gallery_dir,relabel=False,\
                                                                  attribute_data=attribute_data,\
                                                                  attribute_data_self=attribute_data_self,\
                                                                  attribute_data_bin=attribute_data_bin,
                                                                  arch=arch,from_att_to_ID=self.from_att_to_ID_test,
                                                                  is_frame = is_frame,start_pid=90001,
                                                                  end_pid=100000,training_id=10000)
        else:
            query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False, \
                                                                      attribute_data=test_attribute_data, \
                                                                      attribute_data_self=test_attribute_data, \
                                                                      attribute_data_bin=attribute_data_bin,
                                                                      arch=arch,
                                                                      from_att_to_ID=self.from_att_to_ID_test,
                                                                      is_frame =False,start_pid=90001,
                                                                      end_pid=100000,training_id=10000)

            gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False, \
                                                                            attribute_data=test_attribute_data, \
                                                                            attribute_data_self=test_attribute_data, \
                                                                            attribute_data_bin=attribute_data_bin,
                                                                            arch=arch,
                                                                            from_att_to_ID=self.from_att_to_ID_test,
                                                                            is_frame =False,start_pid=90001,
                                                                            end_pid=100000,training_id=10000)


        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> PA100K loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))


    def get_attributes(self,attribute_data,key):
        if self.random_label == 0:
            current_attributes = attribute_data[key]

        return current_attributes


    def _process_dir(self, dir_path, relabel=False,attribute_data=None,attribute_data_self=None,
                     attribute_data_bin=None,arch='resnet50',from_att_to_ID=None,is_frame=False,
                     start_pid=0,end_pid=100,training_id=4000):

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid=int((img_path.split('/')[-1].split('.')[0]))
            if pid == -1: continue  # junk images are just ignored

            if pid < start_pid or pid > end_pid:
                continue

            pid_container.add(pid)

        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []

        iii=0

        for img_path in img_paths:
            pid =int((img_path.split('/')[-1].split('.')[0]))
            camid=1
            if pid == -1 or pid==0: continue  # junk images are just ignored

            if pid < start_pid or pid > end_pid:
                continue


            camid = 1 # index starts from 0

            key = str(pid)
            key_pid = str(pid)


            if relabel: pid = pid2label[pid]
            # img_path=img_path.split('/')

            if arch == 'resnetAttW2VAttributes' or arch == 'resnetAttW2VText':
                if is_frame:
                    current_attributes = self.get_attributes(attribute_data, key)
                    label_attributes=self.get_attributes(attribute_data_self,key)
                else:

                    current_attributes = self.get_attributes(attribute_data, key_pid)
                    label_attributes=self.get_attributes(attribute_data_self,key_pid)
                if from_att_to_ID is not None:
                    current_attributes_bin = self.get_attributes(attribute_data_bin, key_pid)
                    kkkk=''.join(map(str, current_attributes_bin))

                    if kkkk in from_att_to_ID.keys():
                        key_pid=from_att_to_ID[str(kkkk)]
                    else:
                        from_att_to_ID[str(kkkk)]=key_pid

                    if relabel:
                        pid = pid2label[int(key_pid)]
                    else:
                        pid=int(key_pid)
                dataset.append((tuple([img_path]), pid, camid, current_attributes,label_attributes))
            iii = iii + 1

            if iii==training_id:
                break

        if from_att_to_ID is not None:
            print(len(from_att_to_ID))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs









