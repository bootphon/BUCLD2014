#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: bucld2014.py
# date: Wed October 22 21:19 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""bucld2014: supporting script for BUCLD2014

"""

from __future__ import division

import numpy as np
import pandas as pd
import os.path as path
import os
import glob
from collections import namedtuple
from scipy.stats import scoreatpercentile
import scikits.audiolab

import textgrid
import spectral
import cdtw

# load filenames into databases, one for tl_allo and one for no_allo
DATADIR = path.join(os.environ['HOME'], 'data', 'multi_allo')
FILEFIELDS = ['filename', 'participant', 'age', 'language_background',
              'language_spoken', 'register', 'vowel', 'contrast', 'word',
              'token']
FileInfo = namedtuple('FileInfo', FILEFIELDS)

def parse_fname(f):
    """Parse the filename encoding. Return FileInfo object."""
    bname = path.splitext(path.basename(f))[0]
    split = bname.split('_')
    if bname.startswith('corpusbiling'):
        participant = split[0].split('-')[1]
        language_background = 'bilingual'
        age = '11mo'
    else:
        participant = split[0]
        language_background = 'monolingual'
        age = split[1]

    language_spoken = split[2]
    register = {'i': 'IDS', 'a': 'ADS', 'NA': 'NA', 'n': 'NA'}[split[3]]
    vowel = split[4]
    contrast = split[5]
    word = split[6]
    token = '_'.join(split[7:])
    return FileInfo(bname, participant, age, language_background,
                    language_spoken, register, vowel,
                    contrast, word, token)

def load_file_dfs():
    """Construct two DataFrames, one for tense/lax, one for nasal/oral. These
    outline the database; which files there are and what vowel they contain
    and where that is.

    The DataFrames columns follow the FILEFIELDS defined above.

    :return: DataFrame for nasal/oral, DataFrame for tense/lax
    """

    tl_allo_files = glob.glob(path.join(DATADIR, 'tl_allo', 'wav', '*.wav'))
    tl_finfos = [parse_fname(f) for f in tl_allo_files]
    tl_df = pd.DataFrame(tl_finfos, columns=FILEFIELDS)
    tl_starts = []
    tl_ends = []
    for fname in tl_df['filename']:
        tg_fname = path.join(DATADIR, 'tl_allo', 'textgrid',
                             fname + '.TextGrid')
        if not path.exists(tg_fname):
            raise OSError('No such file: {0}'.format(tg_fname))
        with open(tg_fname, 'r') as fid:
            tg = textgrid.TextGrid.read(fid)
        interval = [(x.start, x.end) for x in tg['vowel'] if x.mark == 'x']
        if len(interval) != 1:
            raise ValueError('wrong number of mark hits ({0}) for {1}'.format(
                len(interval), fname))
        tl_starts.append(interval[0][0])
        tl_ends.append(interval[0][1])
    tl_df['start'] = tl_starts
    tl_df['end'] = tl_ends
    tl_df['length'] = tl_df['end'] - tl_df['start']

    no_allo_files = glob.glob(path.join(DATADIR, 'no_allo', 'wav', '*.wav'))
    no_finfos = [parse_fname(f) for f in no_allo_files]
    no_df = pd.DataFrame(no_finfos, columns=FILEFIELDS)
    no_starts = []
    no_ends = []
    for fname in no_df['filename']:
        tg_fname = path.join(DATADIR, 'no_allo', 'textgrid',
                             fname + '.TextGrid')
        if not path.exists(tg_fname):
            raise OSError('No such file: {0}'.format(tg_fname))
        with open(tg_fname, 'r') as fid:
            tg = textgrid.TextGrid.read(fid)
        interval = [(x.start, x.end) for x in tg['vowel'] if x.mark == 'x']
        if len(interval) != 1:
            raise ValueError('wrong number of mark hits ({0}) for {1}'.format(
                len(interval), fname))
        no_starts.append(interval[0][0])
        no_ends.append(interval[0][1])
    no_df['start'] = no_starts
    no_df['end'] = no_ends
    no_df['length'] = no_df['end'] - no_df['start']

    return no_df, tl_df


def filter_df(df):
    """Filter rows in the DataFrame df based on these conditions:
    1. language_background == monolingual
    2. language_spoken == English
    3. age == 11mo
    4. length above 1st and below 99th percentile
    """
    df = df[df.language_background == 'monolingual']
    df = df[df.language_spoken == 'English']
    df = df[df.age == '11mo']
    df = df[(df.length > scoreatpercentile(df.length, 1)) &
            (df.length < scoreatpercentile(df.length, 99))]
    return df

def calc_div(list1, list2, mfccs,
              normalize_dtw=False,
              normalize_mfcc=True,
              normalize_stats=None):
    """Calculate the average linkage of the pairwise divergences between
    two clusters of vowels.

    The vowel clusters are represented as lists of filenames. These filenames
    are keys into the mfccs dict. If either list is empty, the function
    return NaN.

    Arguments:
    :param normalize_dtw: normalize the dtw by the length of the diagonal.
      This is not generally a good idea, as it discards duration as a feature.
    :param normalize_mfcc: normalize the mfc coefficients
    :param normalize_stats: if normalize_mfcc is True, this needs to be a tuple
      of the means and standard deviations of each of the mfc coefficients.
    """
    if normalize_mfcc:
        if normalize_stats is None:
            raise ValueError('for normalizing mfccs, pass normalize_stats')
        else:
            means, stds = normalize_stats

    if len(list1) == 0 or len(list2) == 0:
        divergence = np.nan
    else:
        mfcc1 = []
        for fname in list1:
            mfcc = mfccs[fname]
            if normalize_mfcc:
                mfcc = (mfcc - means) / stds
            mfcc1.append(mfcc)
        mfcc2 = []
        for fname in list2:
            mfcc = mfccs[fname]
            if normalize_mfcc:
                mfcc = (mfcc - means) / stds
            mfcc2.append(mfcc)
        divergence = 0
        for it1 in mfcc1:
            for it2 in mfcc2:
                d = cdtw.dtw(it1, it2)
                if normalize_dtw:
                    d /= np.sqrt(it1.shape[0] ** 2 + it2.shape[0] ** 2)
                divergence += d
        if divergence == 0:
            divergence = np.nan
        else:
            divergence /= len(list1) * len(list2)
    return divergence


def preload_mfcc(no_df, tl_df):
    """Load all the wavfiles and calculate the mfccs.

    This function serves two purposes.
    1. Memoizing the spectral representation and so minimizing disk calls.
    2. Collecting all mfcc frames for the calculation of sufficient statistics
       to normalize them later on.
    """
    FRATE = 100
    encoder = spectral.Spectral(nfilt=40,
                                ncep=13,
                                do_dct=True,
                                lowerf=66.6666,
                                upperf=6855.4976,
                                alpha=0.97,
                                fs=44100,
                                frate=FRATE,
                                wlen=0.025,
                                nfft=512,
                                compression='log',
                                do_deltas=True,
                                do_deltasdeltas=True)
    no_grouped = no_df.groupby(['participant', 'register', 'vowel']).groups
    tl_grouped = tl_df.groupby(['participant', 'register', 'vowel']).groups

    all_mfccs = {}
    for prog_idx, ((participant, register, vowel), ix) in \
        enumerate(no_grouped.iteritems()):
        if prog_idx % 25 == 0:
            print '  ', prog_idx, participant, register, vowel
        sub_df = no_df.loc[no_grouped[(participant, register, vowel)]]
        for _, r in sub_df.iterrows():
            fname = path.join(DATADIR, 'no_allo', 'wav', r.filename + '.wav')
            sig, fs, _ = scikits.audiolab.wavread(fname)
            if fname in all_mfccs:
                raise ValueError('{0} already in all_mfccs!'.format(fname))
            all_mfccs[fname] = encoder.transform(sig)[r.start*FRATE:r.end*FRATE]
    for prog_idx, ((participant, register, vowel), ix) in \
        enumerate(tl_grouped.iteritems()):
        if prog_idx % 25 == 0:
            print '  ', prog_idx, participant, register, vowel
        sub_df = tl_df.loc[tl_grouped[(participant, register, vowel)]]
        for _, r in sub_df.iterrows():
            fname = path.join(DATADIR, 'tl_allo', 'wav', r.filename + '.wav')
            sig, fs, _ = scikits.audiolab.wavread(fname)
            if fname in all_mfccs:
                raise ValueError('{0} already in all_mfccs!'.format(fname))
            all_mfccs[fname] = encoder.transform(sig)[r.start*FRATE:r.end*FRATE]
    return all_mfccs



if __name__ == '__main__':
    print 'loading dataset information...',
    no_df, tl_df = load_file_dfs()
    print 'done.'

    print 'excluding files...',
    no_df = filter_df(no_df)
    tl_df = filter_df(tl_df)
    print 'done.'

    print 'loading wavs & mfccs...'
    all_mfccs = preload_mfcc(no_df, tl_df)
    print 'done.'

    print 'calculating statistics...',
    mfccs = np.vstack(all_mfccs.values())
    means = mfccs.mean(axis=0)
    stds = mfccs.std(axis=0)
    del mfccs
    print 'done.'

    print 'calculating divergences...',
    no_grouped = no_df.groupby(['participant', 'register', 'vowel']).groups
    tl_grouped = tl_df.groupby(['participant', 'register', 'vowel']).groups
    NORMALIZE_DTW = False
    NORMALIZE_MFCC = True
    results = []
    for prog_idx, ((participant, register, vowel), ix) in \
        enumerate(no_grouped.iteritems()):
        sub_df = no_df.loc[no_grouped[(participant, register, vowel)]]

        oral = [path.join(DATADIR, 'no_allo', 'wav', r.filename + '.wav')
                for _, r in sub_df[sub_df['contrast'] == 'oral'].iterrows()]
        nasal = [path.join(DATADIR, 'no_allo', 'wav', r.filename + '.wav')
                 for _, r in sub_df[sub_df['contrast'] == 'nasal'].iterrows()]
        divergence = calc_div(oral, nasal, all_mfccs,
                             normalize_dtw=NORMALIZE_DTW,
                             normalize_mfcc=NORMALIZE_MFCC,
                             normalize_stats=(means, stds))

        contrast = 'nasal/oral'
        results.append((participant, register, vowel, contrast, divergence))

    for prog_idx, ((participant, register, vowel), ix) in \
        enumerate(tl_grouped.iteritems()):
        sub_df = tl_df.loc[tl_grouped[(participant, register, vowel)]]

        tense = [path.join(DATADIR, 'tl_allo', 'wav', r.filename + '.wav')
                 for _, r in sub_df[sub_df['contrast'] == 'tense'].iterrows()]
        lax = [path.join(DATADIR, 'tl_allo', 'wav', r.filename + '.wav')
                for _, r in sub_df[sub_df['contrast'] == 'lax'].iterrows()]
        divergence = calc_div(tense, lax, all_mfccs,
                             normalize_dtw=NORMALIZE_DTW,
                             normalize_mfcc=NORMALIZE_MFCC,
                             normalize_stats=(means, stds))

        contrast = 'tense/lax'
        results.append((participant, register, vowel, contrast, divergence))
    results_df = pd.DataFrame(results,
                              columns=['speaker', 'register', 'vowel',
                                       'contrast', 'divergence'])
    results_df.to_csv('bucld.csv', na_rep='NaN', index=False, float_format='%.3f')
    print 'done.'

    print 'making little picture...',
    # make a little picture
    import seaborn as sns
    import matplotlib.pyplot as plt

    results_df['vowel_contrast'] = results_df.vowel + ' ' + results_df.contrast

    sns.set(style='ticks')
    contrasts = ['ivowel tense/lax', 'E tense/lax',
                 'E nasal/oral',  'a nasal/oral']
    g = sns.factorplot('vowel_contrast', 'divergence', 'register', results_df,
                       kind='box', palette='PRGn', aspect=1.5,
                       x_order=contrasts)
    g.despine(offset=10, trim=True)
    g.set_axis_labels('Vowel Contrast', 'Cluster Divergence')
    plt.savefig('bucld_example.png')
    print 'done.'
