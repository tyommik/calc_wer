import pathlib
import json
import time
import argparse
from abc import ABC, abstractmethod
import sys
import csv
from collections import OrderedDict

import jiwer
import pysrt
from intervaltree import Interval, IntervalTree
from pretty_wer import wer


"""
ВНИМАНИЕ!

Необходимо модифицировать метод compute_measures, заменив возвращаемый объект на

-------------
    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "H": H,
        "S": S,
        "D": D,
        "I": I
    }
_____________
"""


class SubIterator:
    def __init__(self, srt):
        self.len = len(srt)
        self.cur = 0
        self.it = iter(srt)
        self.buff = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.buff is not None:
            t = self.buff
            self.buff = None
            return t
        self.cur += 1
        if self.cur < self.len:
            return next(self.it)
        return None

    def push(self, t):
        self.buff = t


class BaseJson(ABC):
    """ Base json representation """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data = []
        self.tree = IntervalTree()

    def __str__(self):
        return f"<{self.path}, {len(self.data)} objects>"

    def __repr__(self):
        return f"{len(self.data)}"

    def __load(self):
        raise NotImplementedError

    def query(self, start, stop):
        return self.tree.overlap(start, stop)

    def mark_words(self, start, stop):
        results = sorted(self.tree.overlap(start, stop))
        for r in results:
            r.data['hit'] = True

    def query_text(self, start, stop):
        results = sorted(self.tree.overlap(start, stop))
        s = " ".join([r.data["word"] for r in results if not r.data['hit']]).strip()
        return s


class KaldiJson(BaseJson):
    """ Kaldi json representation """
    _type = 'kaldi'

    def __init__(self, path: pathlib.Path):
        super().__init__(path)
        self.__load()

    def __load(self):
        with pathlib.Path.open(self.path, 'r') as json_data:
            data = json.load(json_data)
            if data:
                self.data = data

            try:
                for obj in self.data:
                    if obj.get("result"):
                        for r in obj["result"]:
                            i = Interval(r["start"], r["end"], r)
                            r['hit'] = False
                            self.tree.add(i)
            except TypeError as err:
                raise TypeError('Incorrect engine type!')


class YandexJson(BaseJson):

    """ Yandex json representation """
    _type = 'yandex'

    def __init__(self, path: pathlib.Path):
        super().__init__(path)
        self.__load()

    def __load(self):
        with pathlib.Path.open(self.path, 'r') as json_data:
            data = json.load(json_data)
            if data:
                self.data = data

            try:
                for chunk in self.data['response']['chunks']:
                    if chunk['channelTag'] == '1':
                        for r in chunk['alternatives'][0]['words']:
                            i = Interval(float(r["startTime"][:-1]), float(r["endTime"][:-1]), r)
                            r['hit'] = False
                            self.tree.add(i)
            except TypeError as err:
                raise TypeError('Incorrect engine type!')


def load_srt(path: pathlib.Path):
    return pysrt.open(path)


def replace_pairs(ground_truth, hp):
    gt = ground_truth.copy()
    gt_paired = [(x, y) for x,y in zip(gt[:-1], gt[1:])]
    for (x, y) in gt_paired:
        if x + y in hp:
            for idx, chr in enumerate(gt):
                if chr == x and idx + 1 < len(gt) and gt[idx + 1] == y:
                    gt[idx] = x + y
                    del gt[idx + 1]
    return gt, hp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=pathlib.Path, help='path to dir')

    parser.add_argument('-e', '--engine_type', type=str, default="kaldi",  choices=['kaldi', 'yandex'], help='Type of ASR engine: kaldi/yandex')
    parser.add_argument('-r', '--save-report', action='store_true', help='save details of each file processing')
    parser.add_argument('--csv', action='store_true', help='show log or not')
    args = parser.parse_args()

    GT_DIR = pathlib.Path(args.path)
    HYP_DIR = pathlib.Path(args.path)

    hyp_files = {file.stem.rsplit('.', maxsplit=1)[0]: file for file in HYP_DIR.glob('*.json')}
    factory = {child._type: child for child in BaseJson.__subclasses__()}
    JsonParser = factory[args.engine_type]

    all_results = OrderedDict()

    for gt_file in sorted(GT_DIR.glob('*.gt')):
        if args.save_report:
            original_stdout = sys.stdout
            f = pathlib.Path.open(gt_file.with_suffix(f'.rep'), 'a+')
            sys.stdout = f

        print(f'Processing: {gt_file}')
        if gt_file.stem not in hyp_files:
            print(f'Skip file {gt_file}. No hyp file!')
            if args.save_report: sys.stdout = original_stdout
            continue
        srt = load_srt(gt_file)
        kd = JsonParser(hyp_files[gt_file.stem])

        S, D, I, H = 0, 0, 0, 0

        start_time = time.time()
        srt = SubIterator(srt)
        CONTINUE_FLAG = True
        for idx, sub in enumerate(srt):
            if not CONTINUE_FLAG or sub is None: break

            start = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
            end = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
            ground_truth = sub.text_without_tags
            hypothesis = kd.query_text(start, end)

            while True:
                t = kd.query(end + 0.000001, end + 0.0000015)
                if t:
                    next_sub = next(srt)

                    if next_sub is None:
                        CONTINUE_FLAG = False
                        break
                    tstart = next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds + next_sub.start.milliseconds / 1000
                    if (tstart - end) > 0.5:
                        srt.push(next_sub)
                        break
                    end = next_sub.end.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds + next_sub.end.milliseconds / 1000

                    ground_truth = ground_truth + " " + next_sub.text_without_tags
                    hypothesis = kd.query_text(start, end)
                else:
                    break
            kd.mark_words(start, end)

            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.SubstituteRegexes({r"…|–|«|»": r""}),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemoveWhiteSpace(replace_by_space=True),
                jiwer.SentencesToListOfWords(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveEmptyStrings(),
                jiwer.SubstituteRegexes({r"ё": r"е"}),
            ])
            gt = transformation([ground_truth])
            hp = transformation([hypothesis])

            gt, hp = replace_pairs(gt, hp)
            hp, gt = replace_pairs(hp, gt)

            wer(gt, hp)

            r = jiwer.compute_measures(
                gt,
                hp
            )
            print(f"\nWER:{r['wer'] * 100:.3f}\t\tS:{r['S']} D:{r['D']} H:{r['H']} I:{r['I']}\n")

            S += r["S"]
            D += r["D"]
            I += r["I"]
            H += r["H"]

        insertions = kd.query_text(0, 100000000)

        print(f"Найденные лишние слова: {kd.query_text(0, 100000000).split(' ')}, I:{len(insertions.split(' '))}")
        I += len(insertions.split(' '))

        stop_time = time.time()

        wer_metric = float(S + D + I) / float(H + S + D)
        print("\n===========================================\n")
        print(f"TOTAL WER: {wer_metric:.3f}, time: {stop_time - start_time:.3f}s | Processed: {gt_file}")

        # Reset standart output to its original value
        if args.save_report: sys.stdout = original_stdout

        all_results[gt_file.stem] = {
            'file': gt_file.stem,
            'WER': round(wer_metric, 4),
            'S': S,
            'D': D,
            'I': I,
            'H': H,
            'report': str(gt_file.with_suffix(f'.rep'))
        }

    if args.csv:
        with pathlib.Path.open(args.path / 'output.csv', 'a+', newline='') as csvfile:
            fieldnames = ['file', 'WER', 'S', 'D', 'I', 'H', 'report']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for v in all_results.values():
                writer.writerow(v)