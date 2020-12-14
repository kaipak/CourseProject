import os, sys
import re
import errno
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPrep:

    def __init__(self, train_path: str = '../data/train.jsonl',
                 sub_path: str = '../data/test.jsonl',
                 response_only: bool = True, rm_cap: bool=False,
                 rm_punc: bool=False):
        """

        :param train_path: Training data to be split into validation and test
        :param sub_path: Location of dataset used to submit results
        :param rm_cap: Remove captilization
        :param rm_punc:
        """
        self.df = pd.read_json(train_path, lines=True)
        self.df['label'] = (self.df['label'] == 'SARCASM').astype('int')
        self.df['concat'] = (
                self.df['response']
                + " "
                + self.df['context'].str.join(" ")
        )

        self.df_sub = pd.read_json(sub_path, lines=True)
        self.df_sub['concat'] = (
                self.df_sub['response']
                + " "
                + self.df_sub['context'].str.join(" ")
        )
        self.df_sub['label'] = 1
        self.df_split = False
        if response_only:
            self.df.rename(columns={'response': 'text'}, inplace=True)
            self.df_sub.rename(columns={'response': 'text'}, inplace=True)
        else:
            self.df.rename(columns={'concat': 'text'}, inplace=True)
            self.df_sub.rename(columns={'concat': 'text'}, inplace=True)
        self.df['text'] = self.df.text.apply(self.rm_non_alphanum)
        self.df_sub['text'] = self.df_sub.text.apply(self.rm_non_alphanum)

    def rm_non_alphanum(self, text):
        """

        :param text:
        :return:
        """
        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def train_test_split(self, test_size: float = 0.2, random_state: int=None,
                         shuffle: bool = True):
        """Split datasets
        :param train_size:
        :param random_state:
        :param shuffle:
        :return:
        """
        self.df_train, self.df_validation = (
            train_test_split(self.df, test_size=test_size,
                             random_state=random_state, shuffle=shuffle)
        )
        self.df_test, self.df_validation = (
            train_test_split(self.df_validation, test_size=0.5,
                             random_state=random_state)
        )
        self.df_split = True
        print(f"{test_size} test ratio results in {self.df_train.shape[0]} "
              f"training, {self.df_validation.shape[0]} "
              f"validation, and {self.df_test.shape[0]} "
              f"test observations.")

    def write_data(self, datapath: str = '../data/processed',
                   train_fname: str = 'train.csv',
                   valid_fname: str = 'validate.csv',
                   test_fname: str = 'test.csv',
                   sub_fname: str = 'sub.csv',
                   format: str = "CSV"):
        """
        :param datapath:
        :param train_fname:
        :param valid_fname:
        :param test_fname:
        :param format:
        :return:
        """
        rootpath = Path(datapath)

        try:
            os.makedirs(rootpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if self.df_split:
            self.df_train.to_csv(rootpath / train_fname, index=False,
                                 encoding=sys.getdefaultencoding())
            self.df_validation.to_csv(rootpath / valid_fname, index=False,
                                      encoding=sys.getdefaultencoding())
        self.df_test.to_csv(rootpath / test_fname, index=False,
                            encoding=sys.getdefaultencoding())
        self.df_sub[['label', 'text']].to_csv(rootpath / sub_fname, index=False,
                           encoding=sys.getdefaultencoding())



