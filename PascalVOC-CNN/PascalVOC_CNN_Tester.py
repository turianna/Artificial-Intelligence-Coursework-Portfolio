
#
# Automatic tests for Midterm software test (ZH) #2, version A (22may2026) in ELTE IK, ANN BSc course part1, 2026 spring
#
# Authors: Imre Molnar, Viktor Varga
#

import json
import requests
import datetime
import platform
import copy as copy_module
import os
import getpass
import numpy as np

import torch
import torch.nn as nn

class Tester:

    # Tests:


    def __test_A(self, *args):
        try:
            assert len(args) == 2, "Tester error: __test_A() expects 2 parameters:  imgs, labels. "
            imgs, labels = args

            assert type(imgs) == np.ndarray, "Tester: 'imgs' should be a numpy array."
            assert type(labels) == np.ndarray, "Tester: 'labels' should be a numpy array."

            assert imgs.ndim == 4, "Tester: The 'imgs' array should have 4 dimensions."
            assert imgs.shape[1:] == (240, 320, 3), "Tester: Incorrect shape for the 'imgs' array."

            assert labels.ndim == 1, "Tester: The 'labels' array should have 1 dimension."
            assert labels.shape[0] == imgs.shape[0], "Tester: The 'imgs' array and the 'labels' array should have consistent shapes."

            u_labels, c_labels = np.unique(labels, return_counts=True)
            assert set(u_labels) == set([0, 1]), "Tester: The 'labels' array should contain binary classification labels."
            assert c_labels[0] == 634, "Tester: The content of the 'labels' array might be incorrect."
            assert c_labels[1] == 614, "Tester: The content of the 'labels' array might be incorrect."
            assert imgs.shape[0] == 1248, "Tester: The shape of the 'imgs' array might be incorrect."

            print("Tester: Task A - OK")

        finally:
            self.imgs = imgs
            self.labels = labels

    def __test_B(self, *args):
        assert len(args) == 6, "Tester error: __test_B() expects 6 parameters:" +\
                                                        "imgs_train, imgs_val, imgs_test, \
                                                        labels_train, labels_val, labels_test."

        assert (self.imgs is not None) and (self.labels is not None), "Tester error: Run test A before running test B."
        imgs_train, imgs_val, imgs_test, labels_train, labels_val, labels_test = args

        for arr in [imgs_train, imgs_val, imgs_test]:
            assert type(arr) == np.ndarray, "Tester: All of the arrays passed to the tester should be numpy arrays."
            assert arr.ndim == 4, "Tester: All of the 'imgs_*' arrays passed to the tester should have 4 dimensions."

        for labels_arr in [labels_train, labels_val, labels_test]:
            assert type(labels_arr) == np.ndarray, "Tester: All of the arrays passed to the tester should be numpy arrays."
            assert labels_arr.ndim == 1, "Tester: All of the 'labels_*' arrays passed to the tester should have 1 dimension."
            
            u_labels, c_labels = np.unique(labels_arr, return_counts=True)
            c_total = c_labels[0] + c_labels[1]
            assert set(u_labels) == set([0, 1]), "Tester: All of the 'labels_*' arrays should contain both categories."
            assert (c_labels[0] / c_total) > 0.3, "Tester: In at least one of the 'labels_*' arrays, the distribution of categories is far from equal."
            assert (c_labels[1] / c_total) > 0.3, "Tester: In at least one of the 'labels_*' arrays, the distribution of categories is far from equal."

        assert imgs_val.shape[0] == imgs_test.shape[0] == 200, "Tester: At least one of the 'imgs_*' arrays have an incorrect shape."
        assert labels_val.shape[0] == labels_test.shape[0] == 200, "Tester: At least one of the 'labels_*' arrays have an incorrect shape."
        
        # assert combined shape unchanged
        concat_imgs = np.concatenate([imgs_train, imgs_val, imgs_test], axis=0)
        concat_labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)
        assert concat_imgs.shape == self.imgs.shape, "Tester: Concatenated 'imgs_*' splits do not match original 'imgs' array shape."
        assert concat_labels.shape == self.labels.shape, "Tester: Concatenated 'labels_*' splits do not match original 'labels' array shape."
        del concat_imgs
        del concat_labels

        print("Tester: Task B - OK")


    def __test_C(self, *args):
        assert len(args) == 3, "Tester error: __test_C() expects 3 parameters: "+\
                                        "dataloader_train, dataloader_val, dataloader_test. "
        dataloader_train, dataloader_val, dataloader_test = args

        dataloader_iters = [dataloader_train, dataloader_val, dataloader_test]
        dataloader_names = ['dataloader_train', 'dataloader_val', 'dataloader_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
            dataloader_iter = copy_module.copy(dataloader_iter)
            for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
                break
            assert len(r) == 2, f"Tester:  The iterator of '{dataloader_name}' should return two tensors (input images, labels) in each iteration."
            r0, r1 = r
            assert type(r0) == type(r1) == torch.Tensor, f"Tester: The iterator of '{dataloader_name}' should return tensors."
            assert r0.dtype == torch.float32, f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' should be of float32 data type."
            assert r1.dtype == torch.float32, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should be of float32 data type."
            assert r0.ndim == 4, f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' should be 4D."
            assert r1.ndim == 2, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should be 2D."

            batch_size, n_ch, sy, sx = r0.shape
            assert r0.shape[2:] == (192, 192), f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."
            assert r1.shape[0] == batch_size, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should have a consistent shape with the first (input images) tensor."
            assert r1.numel() == batch_size, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."
            assert n_ch == 3, f"Tester:  First (input images) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."

            assert torch.amin(r0) > -5., f"Tester: Images in the first tensor returned by the iterator of '{dataloader_name}' might have been normalized incorrectly."
            assert torch.amax(r0) < 5., f"Tester: Images in the first tensor returned by the iterator of '{dataloader_name}' might have been normalized incorrectly."

            del dataloader_iter

        print("Tester: Task C, 'Alapfeladat' - OK")


    def __test_C_extra(self, *args):
        assert len(args) == 3, "Tester error: __test_C_extra() expects 3 parameters: "+\
                                        "dataloader_train, dataloader_val, dataloader_test. "
        dataloader_train, dataloader_val, dataloader_test = args

        dataloader_iters = [dataloader_train, dataloader_val, dataloader_test]
        dataloader_names = ['dataloader_train', 'dataloader_val', 'dataloader_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
            dataloader_iter = copy_module.copy(dataloader_iter)
            for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
                break
            assert len(r) == 3, f"Tester:  The iterator of '{dataloader_name}' should return three tensors (input images, labels, sample_weights) in each iteration."
            r0, r1, r2 = r
            assert type(r0) == type(r1) == type(r2) == torch.Tensor, f"Tester: The iterator of '{dataloader_name}' should return tensors."
            assert r0.dtype == torch.float32, f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' should be of float32 data type."
            assert r1.dtype == torch.float32, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should be of float32 data type."
            assert r2.dtype == torch.float32, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should be of float32 data type."
            assert r0.ndim == 4, f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' should be 4D."
            assert r1.ndim == 2, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should be 2D."
            assert r2.ndim == 1, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should be 1D."

            batch_size, n_ch, sy, sx = r0.shape
            assert r0.shape[2:] == (192, 192), f"Tester: First (input images) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."
            assert r1.shape[0] == batch_size, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' should have a consistent shape with the first (input images) tensor."
            assert r2.shape[0] == batch_size, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should have a consistent shape with the first (input images) tensor."
            assert r1.numel() == batch_size, f"Tester: Second (labels) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."
            assert n_ch == 3, f"Tester:  First (input images) tensor returned by the iterator of '{dataloader_name}' has an incorrect shape."

            assert torch.amin(r0) > -5., f"Tester: Images in the first tensor returned by the iterator of '{dataloader_name}' might have been normalized incorrectly."
            assert torch.amax(r0) < 5., f"Tester: Images in the first tensor returned by the iterator of '{dataloader_name}' might have been normalized incorrectly."

            assert torch.min(r2) > 0.6, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should not contain values smaller than 0.6."
            assert torch.max(r2) < 1.5, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should not contain values larger than 1.5."
            assert torch.unique(r2).shape[0] <= 2, f"Tester: Third (sample_weights) tensor returned by the iterator of '{dataloader_name}' should contain up to two unique values only."

            del dataloader_iter

        print("Tester: Task C, 'Extra feladat' - OK")



    def __test_D(self, *args):
        assert len(args) == 1, "Tester error: __test_D() expects 1 parameter: finetuned_model "
        finetuned_model, = args

        assert isinstance(finetuned_model, torch.nn.Module), f"Tester: The 'finetuned_model' neural network model must be of a subtype of torch.nn.Module."

        n_params = sum([p.numel() for p in finetuned_model.parameters(recurse=True)])
        n_params_unfrozen = sum([p.numel() for p in finetuned_model.parameters(recurse=True) if p.requires_grad])
        assert n_params > 500000, f"Tester: The 'finetuned_model' network must include the MobileNet v2 network as a backbone (parameter count is too low)."
        assert n_params_unfrozen >= 1, f"Tester: All parameters of the 'finetuned_model' network are frozen."

        print("Tester: Task D might be OK. Note, that the exact architecture is not checked with this test.")


    def __test_F(self, *args):
        '''
        Expected parameters:
            test_acc: float
        '''
        assert len(args) == 1, "Tester error: __test_F() expects 1 parameter: test_acc. "
        test_acc, = args

        assert 0.88 < test_acc < 0.97, "A well trained classifier should produce a binary accuracy between 0.88 and 0.97."

        print("Tester: Task F  - achieved metrics are OK.")


    # Utilities

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}

        # STORED DATA FROM PREVIOUS TESTS

        imgs: ndarray(n_selected_imgs, 240, 320, 3) of uint8
        labels: ndarray(n_selected_imgs) of float32

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        
        test_results: dict{test_name - str: success - bool}

    '''
    def __init__(self):

        self.TESTS_DICT = {'A': self.__test_A,
                           'B': self.__test_B,
                           'C': self.__test_C,
                           'C_extra': self.__test_C_extra,
                           'D': self.__test_D,
                           'F': self.__test_F}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}
        self.imgs = None
        self.labels = None


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