
import copy as copy_module
import os
import urllib
import numpy as np

import torch
import torch.nn as nn

class Tester:

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''

    def __init__(self):

        self.TESTS_DICT = {'B': self.__test_B,
                           'C': self.__test_C,
                           'D': self.__test_D,
                           'E': self.__test_E,
                           'G': self.__test_G,
                           'H': self.__test_H,
                           'I': self.__test_I}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}

    def test(self, test_name, *args):
        '''
        Parameters:
            test_name: str
            *args: varargs; the arguments for the selected test
        '''
        if test_name not in self.TESTS_DICT:
            assert False, "Tester error: Invalid test name: " + str(test_name)

        self.test_results[test_name] = False
        test_func = self.TESTS_DICT[test_name]
        test_func(*args)
        self.test_results[test_name] = True    # only executed if no assert happened during test

    def print_all_tests_successful(self):
        if all(list(self.test_results.values())):
            print("\nTester: All tests were successful.")


    # TESTS

    def __test_B(self, *args):
        assert len(args) == 1, "Tester error: __test_B() expects 1 parameter: create_random_positive_sample. "
        create_random_positive_sample, = args

        puzzle_image = np.full((2000,3000,3), dtype=np.uint8, fill_value=[0,255,0])   # green background
        waldo_head_image = np.full((100,100,4), dtype=np.uint8, fill_value=[0,0,255,0])   # blue waldo, only center is not transparent
        head_visible_dim = 50
        waldo_head_image[25:25+head_visible_dim,25:25+head_visible_dim,3] = 255    # 50 x 50 pixels are visible and blue
        waldo_head_scale = 0.5                   # 25 x 25 pixels should remain visible and blue
        im_out_size_yx = (160, 160)
        try:
            result_crop = create_random_positive_sample(puzzle_image, waldo_head_image, waldo_head_scale, im_out_size_yx)
            assert result_crop.shape == im_out_size_yx + (3,)
            assert result_crop.dtype == np.uint8
            assert np.any(np.all(result_crop == [0,255,0], axis=2))   # assert green background is visible
            assert np.any(np.all(result_crop == [0,0,255], axis=2))   # assert blue waldo is visible
            assert np.count_nonzero(np.all(result_crop == [0,0,255], axis=2)) < 2*head_visible_dim*head_visible_dim*waldo_head_scale*waldo_head_scale   # assert blue waldo is downscaled
        except:
            print("\nTester: Test failed. Are you sure the function is correctly implemented? Try to visualize the results after verifying shape/dtype are correct!")
        else:
            print("\nTester: Task B solution seems OK")


    def __test_C(self, *args):
        assert len(args) == 1, "Tester error: __test_C() expects 1 parameter: create_random_negative_sample. "
        create_random_negative_sample, = args

        puzzle_image = np.full((300,400,3), dtype=np.uint8, fill_value=[0,255,0])   # green background
        puzzle_image[20:40,360:380,:] = [0,0,255]   # draw blue waldo
        im_out_size_yx = (200, 200)
        waldo_bbox = (355, 15, 30, 30)    # min_x, min_y, w, h; == puzzle_image[15:45,355:385,:]
        try:
            for trial_idx in range(10):
                result_crop = create_random_negative_sample(puzzle_image, waldo_bbox, im_out_size_yx)
                assert result_crop.shape == im_out_size_yx + (3,)
                assert result_crop.dtype == np.uint8
                assert not np.any(np.all(result_crop == [0,0,255], axis=2))   # assert blue waldo is not visible
        except:
            print("\nTester: Test failed. Are you sure the function is correctly implemented? Try to visualize the results after verifying shape/dtype are correct!")
        else:
            print("\nTester: Task C solution seems OK")


    def __test_D(self, *args):
        assert len(args) == 6, "Tester error: __test_D() expects 6 parameters:" +\
                                                        "puzzle_image_names_train, puzzle_image_names_val, puzzle_image_names_test, "+\
                                                        "waldo_head_image_names_train, waldo_head_image_names_val, waldo_head_image_names_test. "
        puzzle_image_names_train, puzzle_image_names_val, puzzle_image_names_test, \
                    waldo_head_image_names_train, waldo_head_image_names_val, waldo_head_image_names_test = args

        try:
            assert len(puzzle_image_names_train) == 8
            assert len(puzzle_image_names_val) == 4
            assert len(puzzle_image_names_test) == 10
            assert len(waldo_head_image_names_train) == 8
            assert len(waldo_head_image_names_val) == 2
            assert len(waldo_head_image_names_test) == 2

            assert type(puzzle_image_names_train[0]) is str
            assert type(puzzle_image_names_val[0]) is str
            assert type(puzzle_image_names_test[0]) is str
            assert type(waldo_head_image_names_train[0]) is str
            assert type(waldo_head_image_names_val[0]) is str
            assert type(waldo_head_image_names_test[0]) is str

            assert len(set(puzzle_image_names_train) | set(puzzle_image_names_val) | set(puzzle_image_names_test)) == 22
            assert len(set(waldo_head_image_names_train) | set(waldo_head_image_names_val) | set(waldo_head_image_names_test)) == 12
        except:
            print("\nTester: Test failed. Are you sure the splitting is correctly implemented? "+\
                      "Try to print out the results, their types and the type of their elements and compare it to the task decription.")
        else:
            print("\nTester: Task D solution seems OK")


    def __test_E(self, *args):
        assert len(args) == 3, "Tester error: __test_E() expects 3 parameters: "+\
                                        "dataloader_train, dataloader_val, dataloader_test. "
        dataloader_train, dataloader_val, dataloader_test = args

        dataloader_iters = [dataloader_train, dataloader_val, dataloader_test]
        dataloader_names = ['dataloader_train', 'dataloader_val', 'dataloader_test']
        try:
            for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
              dataloader_iter = copy_module.copy(dataloader_iter)
              for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
                break
              assert len(r) == 2
              r0, r1 = r
              assert type(r0) == type(r1) == torch.Tensor
              assert r0.dtype == r1.dtype == torch.float32
              assert r0.ndim == 4
              assert r1.ndim == 2
              
              batch_size, n_ch, sy, sx = r0.shape
              assert r1.shape == (batch_size, 1)
              assert n_ch == 3
              assert sy == sx == 128
              
              assert torch.amin(r0) > -4.
              assert torch.amax(r0) < 4.

              del dataloader_iter
        except:
            print("\nTester: Test failed. Are you sure the data iterators are correctly implemented? "+\
                      "Try to print out the shape/dtype/content of the tensors in the batches, then try to visualize them (task F is about this).")
        else:
            print("\nTester: Task E solution seems OK")

    def __test_G(self, *args):
        assert len(args) == 1, "Tester error: __test_G() expects 1 parameter: finetuned_model "
        finetuned_model, = args

        try:
            assert isinstance(finetuned_model, torch.nn.Module)

            n_params = sum([p.numel() for p in finetuned_model.parameters(recurse=True)])
            n_params_unfrozen = sum([p.numel() for p in finetuned_model.parameters(recurse=True) if p.requires_grad])
            assert n_params > 1000000
            assert n_params_unfrozen >= 1

        except:
            print("\nTester: Test failed. Are you sure the network is correctly implemented? "+\
                      "Try to verify that the MobileNet v2 pretrained network is actually connected to your new layers.")
        else:
            print("\nTester: Task G solution seems OK")


    def __test_H(self, *args):
        assert len(args) == 2, "Tester error: __test_H() expects 2 parameters: test_bce, test_acc. "
        test_bce, test_acc = args

        assert test_bce < 0.1, "A well trained classifier should produce a BCE loss less than 0.1."
        assert test_acc > 0.9, "A well trained classifier should produce an accuracy greather than 0.9."

        print("\nTester: Task H solution seems OK")


    def __test_I(self, *args):
        assert len(args) == 1, "Tester error: __test_I() expects 1 parameter: dataloader_puzzleimg_eval. "
        dataloader_puzzleimg_eval, = args

        try:
          dataloader_puzzleimg_eval = copy_module.copy(dataloader_puzzleimg_eval)
          for r in dataloader_puzzleimg_eval:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
            break
          assert len(r) == 2
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor
          assert r0.dtype == r1.dtype == torch.float32
          assert r0.ndim == 4
          assert r1.ndim == 2
          
          batch_size, n_ch, sy, sx = r0.shape
          assert r1.shape == (batch_size, 1)
          assert n_ch == 3
          assert sy == sx == 128
          
          assert torch.amin(r0) > -4.
          assert torch.amax(r0) < 4.

          del dataloader_puzzleimg_eval
        except:
            print("\nTester: Test failed. Are you sure the data iterators are correctly implemented? "+\
                      "Try to print out the shape/dtype/content of the tensors in the batches, then try to visualize them.")
        else:
            print("\nTester: Task I solution seems OK")

