import numpy as np
from sklearn.model_selection import ShuffleSplit
import PIL.Image

import os
from glob import glob
import csv, json
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

from src.utils_plot import read_flow


def image_files_from_folder(folder: str, pair: bool = True, uppercase: bool = True,
                            extensions: Tuple[str] = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'ppm')
                            ) -> List[str]:
    img_files = []
    if pair:  # Images as pair (take the first pair only)
        for ext in extensions:
            img_files += sorted(glob(os.path.join(folder, f'*_img1.{ext}')))
            if uppercase:
                img_files += sorted(glob(os.path.join(folder, f'*_img1.{ext.upper()}')))


    else:  # Images as a sequence of frames
        for ext in extensions:
            img_files += sorted(glob(os.path.join(folder, f'*.{ext}')))
            if uppercase:
                img_files += sorted(glob(os.path.join(folder, f'*.{ext.upper()}')))

    return img_files


def flo_files_from_folder(folder: str, uppercase: bool = True, extensions: Tuple[str] = '.flo') -> List[str]:
    flo_files = []
    for ext in extensions:
        flo_files += sorted(glob(os.path.join(folder, f'*.{ext}')))
        if uppercase:
            flo_files += sorted(glob(os.path.join(folder, f'*.{ext.upper()}')))

    return flo_files


def read_gen(file_name: str, im_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.ppm')):
    ext = os.path.splitext(file_name)[-1]
    if ext in im_extensions:
        im = PIL.Image.open(file_name).convert('RGB')
        return im
    elif ext in ('.bin', '.raw'):
        return np.load(file_name)
    elif ext == '.flo':
        return read_flow(file_name)
    else:
        return []


class ExtractDataset:
    def __init__(self, root: str,
                 extension: str,
                 training_size: float,
                 use_val: bool = True,
                 random_state: int = 69,
                 exception_files: Optional[List[str]] = None,
                 verbose: bool = True) -> None:
        """Class for creating csv files of train, validation, and test
        Parameters
        ----------
        root
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        use_val
            If True, separate the dataset into "train, val and test" else, "train and val" only!
        Returns
        -------
        None
        """

        self.root = root
        self.extension = extension
        self.training_size = training_size
        self.test_size = 1 - training_size
        self.random_state = random_state

        self.exception_files = [] if exception_files is None else exception_files

        self.use_val = use_val
        self.verbose = verbose

    def _create_dataset_array(self, rule: Optional[str] = None) -> np.ndarray:
        """Sklearn stratified sampling uses a whole array so we must build it first
        Parameters
        ----------
        None
        Returns
        -------
        Tuple of X and y
        """

        name = []
        files = sorted(glob(os.path.join(self.root, f'*{self.extension}')))
        for file in files:
            fname = os.path.basename(file)
            if fname in self.exception_files:
                continue

            frule = os.path.splitext(fname)[0].split('_')
            if rule is None:
                name.append(fname)
            else:
                if rule in frule:
                    name.append(fname)

        if self.verbose:
            print("Finished creating whole dataset array")

        return np.array(name)

    def _stratify_sampling(self, rule: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sklearn stratified sampling uses a whole array so we must build it first
        Parameters
        ----------
        None
        Returns
        -------
        Tuple of train(X, y), validation(X, y), and test(X, y)
        """

        # Init.
        label = self._create_dataset_array(rule=rule)
        label_validation_test = None
        label_train, label_validation, label_test = None, None, None

        # Start sampling
        shuffle = ShuffleSplit(n_splits=1, train_size=self.training_size, test_size=self.test_size,
                               random_state=self.random_state)
        for train_index, validation_test_index in shuffle.split(label):
            label_train, label_validation_test = label[train_index], label[validation_test_index]

        if self.use_val:
            shuffle = ShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=self.random_state)
            for validation_index, test_index in shuffle.split(label_validation_test):
                label_validation, label_test = label_validation_test[validation_index], label_validation_test[test_index]
        else:
            label_validation = label_validation_test

        if self.verbose:
            print("Finished splitting dataset into train, validation, and test")

        return label_train, label_validation, label_test

    def extract(self, rule: Optional[str] = None) -> Dict:
        """
        Extracting the dataset into a dictionary of each train, val and/or test set
        :return:
        """

        dataset = self._stratify_sampling(rule=rule)
        naming = ['train', 'val', 'test']

        extraction = {}
        for i, set in enumerate(dataset):  # train, val, test (test might be optional!)
            extraction[naming[i]] = set

        return extraction

    @staticmethod
    def write(naming: str, set: np.array, file_prefix: str = 'dataset', save_path: str = './', mode: str = 'txt',
              verbose=True) -> None:
        """Write csv or txt file of train, validation, and test
        Parameters
        ----------
        file_prefix
            The prefix of train, validation, and test file_prefix
            Have the format of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        save_path
            The parent_directory folder name of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        Returns
        -------
        train, validation, and test csv/txt file with the following names:
        file_prefix_train.ext, file_prefix_val.ext, and file_prefix_test.ext
        """

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if mode == 'csv':
            with open(os.path.join(save_path, f'{file_prefix}_{naming}.csv'), 'w') as writer:
                csv_writer = csv.writer(writer)
                for row in set:
                    csv_writer.writerow(row)

                writer.close()

        elif mode == 'txt':
            with open(os.path.join(save_path, f'{file_prefix}_{naming}.txt'), 'w') as writer:
                for row in set:
                    writer.write("%s\n" % row)

        elif mode == 'json':
            raise ValueError('Currently under development! Proceed with other method')

        else:
            raise ValueError(f'Unknown input (mode: {mode})!')

        if verbose:
            print(f'Finished writing {file_prefix}_{naming}.csv into {save_path}')

    @staticmethod
    def write_json(naming: str, set: Dict, file_prefix: str = 'dataset', save_path: str = './', verbose=True) -> None:
        """Write a single json file for all the train, validation, and test set
        Parameters
        ----------
        file_prefix
            The prefix of train, validation, and test file_prefix
            Have the format of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        save_path
            The parent_directory folder name of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        Returns
        -------
        train, validation, and test json with the following name:
        file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        """

        # file_prefix = os.path.basename(self.root) if file_prefix is None else file_prefix
        # save_path = self.root if save_path is None else save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, f'{file_prefix}_{naming}.json'), 'w') as writer:
            json.dump(set, writer, indent=4)

            writer.close()

        if verbose:
            print(f'Finished writing {file_prefix}_{naming}.json into {save_path}')
