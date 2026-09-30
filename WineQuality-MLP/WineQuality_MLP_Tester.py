
#
# Automatic tests for Coding test (ZH) #1, version B (27may2026) in ELTE IK, ANN BSc course part1, 2026 spring
#
# Authors: Imre Molnar, Viktor Varga
#

import json
import requests
import datetime
import platform
import getpass
import copy as copy_module
import os
import urllib
import numpy as np

import torch
import torch.nn as nn


class Tester:

    # Tests

    def __test_dataset_load(self, *args):
        '''
        Expected parameters:
            dataset_noisy: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_load() expects 1 parameters: dataset_noisy. "
        dataset_noisy, = args

        try:
            assert type(dataset_noisy) == np.ndarray, "Tester: 'dataset_noisy' must be a numpy array."
            assert dataset_noisy.dtype == np.float32, "Tester: 'dataset_noisy' array must have float32 data type."
            expected_shape = (4898, 16)
            assert dataset_noisy.shape == expected_shape, f"Tester: 'dataset_noisy' array must have a shape of {expected_shape}."
            assert not np.any(np.isnan(dataset_noisy)), "Tester: There should be no NaN values in the 'dataset_noisy' array."

            zeromask = dataset_noisy == 0.
            zeroidxs = np.argwhere(zeromask)
            expected_zerocount = 1976
            expected_zerocols = [0, 1]
            expected_zeropos50_60 = [[128, 1], [134, 1], [135, 0], [136, 0], [136, 1], [138, 1], [139, 0], [143, 0], [144, 1], [145, 1]]

            assert np.count_nonzero(zeromask) == expected_zerocount, "Tester: Incorrect zero (missing) value count in 'dataset_noisy' array."
            assert np.array_equal(zeroidxs[50:60], expected_zeropos50_60), "Tester: Incorrect zero (missing) value positions in 'dataset_noisy' array."

        finally:
            self.dataset_noisy = np.copy(dataset_noisy)

        print("Tester: Dataset loading OK")


    def __test_dataset_fill_missing(self, *args):
        '''
        Expected parameters:
            dataset: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_fill_missing() expects 1 parameters: dataset. "
        dataset, = args
        assert self.dataset_noisy is not None, "Tester error: Run tester for task 'A' first."

        try:
            assert type(dataset) == np.ndarray, "Tester: 'dataset' must be a numpy array."
            assert dataset.dtype == np.float32, "Tester: 'dataset' array must have float32 data type."

            expected_shape_op1 = (4898, 16)
            expected_shape_op23 = (4898, 18)
            if dataset.shape == expected_shape_op1:
                print("Tester: Dataset fill missing: Identified solution aiming for option#1...")

                assert np.array_equal(dataset, self.dataset_noisy), "Tester: In case you are trying to solve option#1, the 'dataset' array must stay the same as the 'dataset_noisy' array."

                print("Tester: Dataset fill missing: Option #1 OK")
                self.n_cols = 16
            elif dataset.shape == expected_shape_op23:
                print("Tester: Dataset fill missing: Identified solution aiming for option#2/3...")

                assert dataset.shape == expected_shape_op23, f"Tester: 'dataset' array must have a shape of {expected_shape_op23}."

                new_vars = dataset[:,[2,3]]
                assert set(np.unique(new_vars)) == {0, 1}, f"Tester: Incorrect placing of new variable columns of the 'dataset' array."

                assert np.allclose(np.sum(new_vars), 7839., atol=1e-2), "Tester: Incorrect values in new variable columns of the 'dataset' array."
                assert np.allclose(np.sum(self.dataset_noisy) + np.sum(new_vars), np.sum(dataset), atol=1e-2), "Tester: Values from the 'dataset_noisy' array may have been changed incorrectly."

                print("Tester: Dataset fill missing: Option #2 or 3 OK")
                self.n_cols = 18
            else:
                assert False, "The shape of the 'dataset' array is incorrect for either option#1 or option#2/3."
        
        finally:
            self.dataset = np.copy(dataset)


    def __test_dataset_split(self, *args):
        '''
        Expected parameters:
            dataset_split_train, dataset_split_val, dataset_split_test: ndarray(n_split_samples, n_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_dataset_split() expects 3 parameters: " +\
                                    "dataset_split_train, dataset_split_val, dataset_split_test. "
        assert (self.dataset is not None) and (self.n_cols is not None), "Tester error: Run tester for task 'A' & 'B' first."
        dataset_split_train, dataset_split_val, dataset_split_test = args

        dataset_split_arrs = [dataset_split_train, dataset_split_val, dataset_split_test]
        dataset_split_sizes = {'dataset_split_train': 3698, 'dataset_split_val': 600, 'dataset_split_test': 600}
        for dataset_split_arr, dataset_split_size_entry in zip(dataset_split_arrs, dataset_split_sizes.items()):
          dataset_split_name, dataset_split_size = dataset_split_size_entry
          assert type(dataset_split_arr) == np.ndarray, f"Tester: '{dataset_split_name}' must be a numpy array."
          assert dataset_split_arr.dtype == np.float32, f"Tester: '{dataset_split_name}' array must have float32 data type."

          assert dataset_split_arr.ndim == 2, f"Tester: '{dataset_split_name}' array has an incorrect shape."
          assert dataset_split_arr.shape[1] == self.dataset.shape[1], f"Tester: '{dataset_split_name}' array has an incorrect shape."
          assert np.fabs(dataset_split_arr.shape[0] - dataset_split_size) < 4, f"Tester: '{dataset_split_name}' array has an incorrect shape."

        dataset_split_concat = np.concatenate([dataset_split_train, dataset_split_val, dataset_split_test], axis=0)
        assert dataset_split_concat.shape == dataset_split_concat.shape, "Tester: concatenated dataset splits do not match original 'dataset' array shape."
        assert not np.array_equal(dataset_split_concat, self.dataset), "Tester: Samples in dataset were not shuffled randomly."
        sample_sums_concat = np.sort(np.sum(dataset_split_concat, axis=1))
        sample_sums_orig = np.sort(np.sum(self.dataset, axis=1))
        assert np.allclose(sample_sums_concat, sample_sums_orig, atol=1e-2), "Tester: Samples might have been shuffled incorrectly (?)."

        print("Tester: Dataset split OK")


    def __test_cl_model_iter(self, *args):
        '''
        Expected parameters:
            dataloader_bincl_train, dataloader_bincl_val, dataloader_bincl_test: ndarray(n_split_samples, n_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_cl_model_iter() expects 3 parameters: " +\
                                    "dataloader_cl_train, dataloader_cl_val, dataloader_cl_test. "
        dataloader_cl_train, dataloader_cl_val, dataloader_cl_test = args

        dataloader_iters = [dataloader_cl_train, dataloader_cl_val, dataloader_cl_test]
        dataloader_names = ['dataloader_cl_train', 'dataloader_cl_val', 'dataloader_cl_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
          dataloader_iter = copy_module.copy(dataloader_iter)
          for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
            break
          assert len(r) == 2, f"Tester: '{dataloader_name}' must return two tensors in each iteration."
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor, f"Tester: '{dataloader_name}' must return tensors."
          assert r0.dtype == torch.float32, f"Tester: '{dataloader_name}' must return an input feature tensor with float32 data type."
          assert r1.dtype == torch.int64, f"Tester: '{dataloader_name}' must return a target label tensor with int64 data type."
          assert r0.ndim == 2, f"Tester: Input feature tensor returned by '{dataloader_name}' have incorrect shape."
          assert r1.ndim == 1, f"Tester: Target label tensor returned by '{dataloader_name}' have incorrect shape."
          batch_size, n_feat = r0.shape
          assert r1.shape[0] == batch_size, f"Tester: Tensors returned by '{dataloader_name}' have inconsistent shape."
          assert n_feat == self.n_cols-5, f"Tester: Input feature tensor returned by '{dataloader_name}' have incorrect shape."

          if (r1.shape[0] >= 16) and (torch.unique(r1).shape[0] == 1):
            print("Tester Warning: Did you make sure that batches contain a mix of different wine categories (with a high probability)?")

        print("Tester: Dataset iterators for classification task OK")

    def __test_cl_model_architecture(self, *args):
        '''
        Expected parameters:
            multiclass_cl_model: PyTorch nn.Module
        '''
        assert len(args) == 1, "Tester error: __test_bincl_model_architecture() expects 1 parameter: multiclass_cl_model "
        multiclass_cl_model, = args
        
        assert isinstance(multiclass_cl_model, torch.nn.Module), f"Tester: The 'multiclass_cl_model' neural network model must be of a subtype of torch.nn.Module."
        n_params = sum([p.numel() for p in multiclass_cl_model.parameters(recurse=True) if p.requires_grad])
        assert n_params == ((self.n_cols-5)*20 + 20) + (20*3 + 3), f"Tester: Incorrect structure in 'multiclass_cl_model' network (incorrect parameter count)."

        print("Tester: Classification model architecture OK")

    def __test_cl_model_learning(self, *args):
        '''
        Expected parameters:
            test_ce, test_acc: float
        '''
        assert len(args) == 2, "Tester error: __test_cl_model_learning() expects 2 parameter: test_ce, test_acc. "
        test_ce, test_acc = args
        assert 0.25 < test_ce < 0.5, "Tester: A well trained binary classification model should provide a test loss (CE) between 0.25 and 0.5."
        assert 0.75 < test_acc < 0.95, "Tester: A well trained binary classification model should provide a test accuracy between 0.75 and 0.95."

        print("Tester: Classification model learning OK")
    #

    # Utilities

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}
        dataset_content: None OR str; dataset loaded into a string

        # STORED DATA FROM PREVIOUS TESTS
        self.dataset_noisy: ndarray(n_samples, n_features) fl32
        self.dataset: ndarray(n_samples, n_features) fl32; no NaNs

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''

    def __init__(self):

        self.LOAD_DATASET_FROM_LOCAL = True
        self.TESTS_DICT = {
              'dataset_load': self.__test_dataset_load,
              'dataset_fill_missing': self.__test_dataset_fill_missing,
              'dataset_split': self.__test_dataset_split,
              'cl_iter': self.__test_cl_model_iter,
              'cl_model_architecture': self.__test_cl_model_architecture,
              'cl_model_learning': self.__test_cl_model_learning}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}
        self.dataset_content = None

    def get_dataset_content(self):
        '''
        Sets self.dataset_content.
        Returns:
            dataset_content: str
        '''
        fname = "./zh_files/winequality26_zh1b.csv"
        with open(fname, 'r') as f:
            self.dataset_content = f.read()

        return self.dataset_content

    def test(self, test_name, *args):
        '''
        Parameters:
            test_name: str
            *args: varargs; the arguments for the selected test
        '''
        if test_name not in self.TESTS_DICT:
            self.print_tester_error_msg(f"Tester error: Invalid test name: {test_name}", "<pre-test>")

        self.test_results[test_name] = False
        succ = False
        test_func = self.TESTS_DICT[test_name]
        try:
            test_func(*args)
        except:
            raise
        else:
            succ = True
        finally:
            self.test_results[test_name] = succ

    def print_tester_error_msg(self, msg, test_name):
        print(f"\nTester error in test '{test_name}' ->  {msg}\n")

