import os
import re
import numpy as np
import pickle

from functools import partial
from multiprocessing import Manager, Pool, Process, Queue
from typing import Callable, Union

from Affectiv_AI.io import EEGSignalIO, MetaInfoIO
from tqdm import tqdm

MAX_QUEUE_SIZE = 1024

def transform_producer(file_name: str, root_path: str,
                       num_channel: int,
                       before_trial: Union[None, Callable],
                       transform: Union[None,Callable],
                       after_trial: Union[Callable,None],
                       after_subject: Union[Callable,None],
                       queue: Queue):
    subject = int(
        os.path.basename(file_name).split('.')[0].split('_')[0])  # subject (15)

    data_npz = np.load(os.path.join(root_path, file_name))  # class list: ['data','label']
    samples = pickle.loads(data_npz['data']) # trial (45), channel (62), timestep(n*800)

    # label
    label = pickle.loads(data_npz['label']) # trial (45), timestep (n*800)

    trial_ids = [
        int(key) for key in samples.keys()
    ]  # [0,1,2,...44]
    session_ids = [int(i/15) + 1 for i in trial_ids] # [1,1,1,..2,2,2,...,3,3,3,...]


    write_pointer = 0
    # loop for each session
    for sess in range(3):
        # loop for each trial
        subject_queue = []  # recoding multiple trials for one session of one subject
        for trial_id in trial_ids:
            if int(sess+1) == session_ids[trial_id]:
                # extract baseline signals
                trial_samples = samples[trial_id].reshape(-1,num_channel,5)  # timestep(n), channel(62), features(5)

                if before_trial:
                    trial_samples = before_trial(trial_samples)

                # record the common meta info
                trial_meta_info = {
                    'subject_id': subject,
                    'trial_id': int((trial_id % 15)+1),
                    'emotion': int(list(set(label[trial_id]))[0]),
                    'date': (int(trial_id / 15)+1)*1000 + subject,
                    'session_id': int(trial_id / 15)
                } # session_id: 0, 1, 2 (3 session)

                trial_queue = []
                for i, clip_sample in enumerate(trial_samples):
                    t_eeg = clip_sample
                    if not transform is None:
                        t_eeg = transform(eeg=clip_sample)['eeg']

                    clip_id = f'{file_name}_{write_pointer}'
                    write_pointer += 1

                    # record meta info for each signal
                    record_info = {
                        'start_at': i * 800,
                        'end_at': (i + 1) *
                        800, # The size of the sliding time windows for feature extraction is 4 seconds.
                        'clip_id': clip_id
                    }
                    record_info.update(trial_meta_info)
                    if after_trial:
                        trial_queue.append({
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info})
                    elif after_subject:
                        subject_queue.append({
                            'eeg': t_eeg,
                            'key': clip_id,
                            'info': record_info})
                    else:
                        queue.put({'eeg': t_eeg, 'key': clip_id, 'info': record_info})

                if len(trial_queue) and after_trial:
                    trial_queue = after_trial(
                        trial_queue)  # after_trial is a function for LDS, moving averages or other methods which transfer based on a trial
                    for obj in trial_queue:
                        assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_trial must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg` and `key` as keys.'
                        queue.put(obj)

        if len(subject_queue) and after_subject:
            subject_queue = after_subject(subject_queue)  # this transform is for a subject
            for obj in subject_queue:
                assert 'eeg' in obj and 'key' in obj and 'info' in obj, 'after_subject must return a list of dictionaries, where each dictionary corresponds to an EEG sample, containing `eeg` and `key` as keys.'
                queue.put(obj)


def io_consumer(write_eeg_fn: Callable, write_info_fn: Callable, queue: Queue):
    while True:
        item = queue.get()
        if not item is None:
            eeg = item['eeg']
            key = item['key']
            write_eeg_fn(eeg, key)
            if 'info' in item:
                info = item['info']
                write_info_fn(info)
        else:
            break


class SingleProcessingQueue:
    def __init__(self, write_eeg_fn: Callable, write_info_fn: Callable):
        self.write_eeg_fn = write_eeg_fn
        self.write_info_fn = write_info_fn

    def put(self, item):
        eeg = item['eeg']
        key = item['key']
        self.write_eeg_fn(eeg, key)
        if 'info' in item:
            info = item['info']
            self.write_info_fn(info)


def seed_v_feature_constructor(
        root_path: str = './EEG_DE_features',
        num_channel: int = 62,
        before_trial: Union[None, Callable] = None,
        transform: Union[None, Callable] = None,
        after_trial: Union[Callable, None] = None,
        after_subject: Union[Callable, None] = None,
        io_path: str = './io/seed_iv_feature',
        io_size: int = 10485760,
        io_mode: str = 'lmdb',
        num_worker: int = 0,
        verbose: bool = True) -> None:
    # init IO
    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    if os.path.exists(io_path) and not os.path.getsize(meta_info_io_path) == 0:
        print(
            f'The target folder already exists, if you need to regenerate the database IO, please delete the path {io_path}.'
        )
        return

    os.makedirs(io_path, exist_ok=True)

    meta_info_io_path = os.path.join(io_path, 'info.csv')
    eeg_signal_io_path = os.path.join(io_path, 'eeg')

    info_io = MetaInfoIO(meta_info_io_path)
    eeg_io = EEGSignalIO(eeg_signal_io_path, io_size=io_size)

    # loop to access the dataset files
    file_list = os.listdir(root_path)
    skip_set = ['load_DE_features.ipynb']
    file_list = [f for f in file_list if f not in skip_set]

    if verbose:
        # show process bar
        pbar = tqdm(total=len(file_list))
        pbar.set_description("[SEED-V FEATURE]")

    if num_worker > 1:
        manager = Manager()
        queue = manager.Queue(maxsize=MAX_QUEUE_SIZE)
        io_consumer_process = Process(target=io_consumer,
                                      args=(eeg_io.write_eeg,
                                            info_io.write_info, queue),
                                      daemon=True)
        io_consumer_process.start()

        partial_mp_fn = partial(transform_producer,
                                root_path=root_path,
                                num_channel=num_channel,
                                before_trial=before_trial,
                                transform=transform,
                                after_trial=after_trial,
                                after_subject=after_subject,
                                queue=queue)

        for _ in Pool(num_worker).imap(partial_mp_fn, file_list):
            if verbose:
                pbar.update(1)

        queue.put(None)

        io_consumer_process.join()
        io_consumer_process.close()

    else:
        for file_name in file_list:
            transform_producer(file_name=file_name,
                               root_path=root_path,
                               num_channel=num_channel,
                               before_trial=before_trial,
                               transform=transform,
                               after_trial=after_trial,
                               after_subject=after_subject,
                               queue=SingleProcessingQueue(
                                   eeg_io.write_eeg, info_io.write_info))
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
        print('Please wait for the writing process to complete...')