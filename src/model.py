from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
import tensorflow as tf
from tensorflow import summary


class SarcasmDetector(object):

    def __init__(self, tokenizer_model: str = 'bert-base-uncased',
                 tokenizer_do_lc: bool = True, model_criterion=nn.BCELoss(),
                 input_dir=Path('../data/'),
                 output_dir=Path('../data/output/'),
                 train_log_dir=Path('../data/logs/tensorboard/train/'),
                 model_options_name: str = 'bert-base-uncased'):
        """
        :param tokenizer_model:
        :param tokenizer_do_lc:
        :param model_criterion:
        """
        self.tokenizer_model = tokenizer_model
        self.model_criterion = model_criterion
        self.INPUT_DIR = input_dir
        self.OUTPUT_DIR = output_dir
        self.TRAIN_LOG_DIR = train_log_dir
        self.tokenizer = (
            BertTokenizer.from_pretrained(self.tokenizer_model,
                                          do_lower_case=tokenizer_do_lc)
        )
        self.PAD_INDEX = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token)
        self.UNK_INDEX = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.unk_token)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"There is/are {torch.cuda.device_count()} GPU(s) available.")
            print(f"GPU {torch.cuda.get_device_name(0)} will be used.")
        else:
            print("No GPU available, falling back to CPU.")

        self.model = BERT(options_name=model_options_name).to(self.device)

    def tokenize_data(self, train_fname, validate_fname, test_fname,
                      batch_size: int = 4, max_seq_len: int = 128,
                      lf_sequential: bool = False, lf_use_vocab: bool = False,
                      lf_batch_first: bool = True, lf_dtype=torch.float,
                      tf_use_vocab: bool = False, tf_lower: bool = False,
                      tf_include_lengths: bool = False,
                      tf_batch_first: bool = True):
        """

        :param batch_size:
        :param max_seq_len:
        :param lf_sequential:
        :param lf_use_vocab:
        :param lf_batch_first:
        :param lf_dtype:
        :param tf_use_vocab:
        :param tf_lower:
        :param tf_include_lengths:
        :param tf_batch_first:
        :return:
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.label_field = Field(sequential=lf_sequential,
                                 use_vocab=lf_use_vocab,
                                 batch_first=lf_batch_first, dtype=lf_dtype)
        self.text_field = Field(use_vocab=tf_use_vocab,
                                tokenize=self.tokenizer.encode,
                                lower=tf_lower,
                                include_lengths=tf_include_lengths,
                                batch_first=tf_batch_first,
                                fix_length=self.max_seq_len,
                                pad_token=self.PAD_INDEX,
                                unk_token=self.UNK_INDEX)
        self.fields = [('label', self.label_field), ('text', self.text_field)]

        train, valid, test = TabularDataset.splits(path=self.OUTPUT_DIR,
                                                   train=train_fname,
                                                   validation=validate_fname,
                                                   test=test_fname,
                                                   format='CSV',
                                                   fields=self.fields,
                                                   skip_header=True)
        print("Created train, validation, and test datasets "
              "with max_seq_len={}".format(max_seq_len))
        # Iterators
        self.train_iter = BucketIterator(train, batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.text),
                                         device=self.device, train=True,
                                         sort=True, sort_within_batch=True)
        self.valid_iter = BucketIterator(valid, batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.text),
                                         device=self.device, train=True,
                                         sort=True, sort_within_batch=True)
        self.test_iter = Iterator(test, batch_size=self.batch_size,
                                  device=self.device, train=False,
                                  shuffle=False, sort=False)
        print("Created iterators with batch_size={}".format(batch_size))

    def train(self, lr: float = 1e-5, num_epochs: int = 3,
              eval_every: int = None, best_valid_loss=float("Inf")):
        """

        :param lr:
        :param num_epochs:
        :param eval_every:
        :param best_valid_loss:
        :return:
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # instantiate tensorboard writer
        pathstr = str(Path(self.TRAIN_LOG_DIR / f"lr={lr}-epochs={num_epochs}"))
        self.writer = summary.create_file_writer(pathstr)

        # initialize running values
        if eval_every is None:
            eval_every = len(self.train_iter) // 2
        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        # training loop
        self.model.train()
        for epoch in range(num_epochs):
            for (label, text), _ in self.train_iter:
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
                text = text.type(torch.LongTensor)
                text = text.to(self.device)
                output = self.model(text, label)
                loss, _ = output

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    self.model.eval()
                    with torch.no_grad():

                        # validation loop
                        for (label, text), _ in self.valid_iter:
                            label = label.type(torch.LongTensor)
                            label = label.to(self.device)
                            text = text.type(torch.LongTensor)
                            text = text.to(self.device)
                            output = self.model(text, label)
                            loss, _ = output

                            valid_running_loss += loss.item()
                            curr_val_loss = loss.item()

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(
                        self.valid_iter)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    curr_train_loss = loss.item()
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    self.model.train()

                    # print progress
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}"
                            .format(epoch + 1, num_epochs, global_step,
                                    num_epochs * len(self.train_iter),
                                    average_train_loss, average_valid_loss))

                    # write to tensorboard logs
                    with self.writer.as_default():
                        tf.summary.scalar('train loss', curr_train_loss, step=global_step)
                    with self.writer.as_default():
                        tf.summary.scalar('validation loss', curr_val_loss, step=global_step)

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        #print(self.OUTPUT_DIR / 'foo.pt')
                        #print(best_valid_loss)
                        self.save_checkpoint(self.OUTPUT_DIR / 'model.pt',
                                             best_valid_loss)
                        self.save_metrics(self.OUTPUT_DIR / 'metrics.pt',
                                          train_loss_list, valid_loss_list,
                                          global_steps_list)

        self.save_metrics(self.OUTPUT_DIR / 'metrics.pt', train_loss_list,
                          valid_loss_list, global_steps_list)
        print("Finished Training!")

    def evaluate(self):
        """

        :return:
        """
        self.load_checkpoint(self.OUTPUT_DIR / 'model.pt')

        y_pred = []
        y_true = []

        self.model.eval()
        with torch.no_grad():
            for (label, text), _ in self.test_iter:
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
                text = text.type(torch.LongTensor)
                text = text.to(self.device)
                output = self.model(text, label)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(label.tolist())

        print('Classification Report:')
        print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

        ax.set_title('Confusion Matrix')

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        ax.xaxis.set_ticklabels(['SARCASM', 'NOT_SARCASM'])
        ax.yaxis.set_ticklabels(['SARCASM', 'NOT_SARCASM'])

    def save_checkpoint(self, save_path, valid_loss):
        """
        :param valid_loss:
        :param save_path:
        :return:
        """
        state_dict = {'model_state_dict': self.model.state_dict(),
                      'valid_loss': valid_loss}

        torch.save(state_dict, save_path)
        print(f"Model saved to ==> {save_path}")

    def load_checkpoint(self, save_path):
        """
        :param save_path:
        :return:
        """
        state_dict = torch.load(save_path, map_location=self.device)
        print(f'Model loaded from <== {save_path}')

        self.model.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, save_path, train_loss_list, valid_loss_list,
                     global_steps_list):
        """

        :param train_loss_list:
        :param valid_loss_list:
        :param global_steps_list:
        :return:
        """
        state_dict = {'train_loss_list': train_loss_list,
                      'valid_loss_list': valid_loss_list,
                      'global_steps_list': global_steps_list}

        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_metrics(self, load_path):
        """

        :return:
        """
        state_dict = torch.load(load_path, map_location=self.device)
        print(f"Model loaded from {self.OUTPUT_DIR}.")

        return state_dict['train_loss_list'], state_dict['valid_loss_list'], \
               state_dict['global_steps_list']

    def tune(self, lr_list, num_epochs_list):
        """

        :param lr_list:
        :param num_epochs_list:
        :return:
        """
        # loop through hyperparams
        for lr in lr_list:
            for num_epochs in num_epochs_list:

                # print current params
                print('CURRENT ITERATION: lr=' + str(lr) + '; num_epochs=' +
                      str(num_epochs) + ';---------------------------------')
                
                # run train method on currrent params
                self.train(lr=lr, num_epochs=num_epochs)

    def predict(self, filepath=Path('../data/output/sub.csv')):
        """
        """

        preds = []
        sub_dataset = TabularDataset(filepath, format="CSV", fields=self.fields,
                                     skip_header=True)
        sub_iter = Iterator(sub_dataset, batch_size=self.batch_size,
                            device=self.device, train=False, shuffle=False,
                            sort=False)
        self.load_checkpoint(self.OUTPUT_DIR / 'model.pt')
        self.model.eval()
        with torch.no_grad():
            for (label, text), _ in sub_iter:
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
                text = text.type(torch.LongTensor)
                text = text.to(self.device)
                output = self.model(text, label)
                _, output = output
                preds.extend(torch.argmax(output, 1).tolist())

        id_list = ["twitter_" + str(n) for n in range(1, len(sub_dataset) + 1)]
        label_list = ["SARCASM" if pred == 1 else "NOT_SARCASM" for pred in preds]
        df_sub = pd.DataFrame(list(zip(id_list, label_list)),
                              columns=['id', 'label'])
        return df_sub


class BERT(nn.Module):

    def __init__(self, options_name: str = "bert-base-cased"):
        super(BERT, self).__init__()

        options_name = options_name
        self.encoder = BertForSequenceClassification.from_pretrained(
            options_name)

    def forward(self, text, label):
        loss, text_feat = self.encoder(text, labels=label)[:2]

        return loss, text_feat
