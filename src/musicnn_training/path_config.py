#!/usr/bin/env python


MTGDB_DIR = '/mnt/mtgdb-audio'

MUSICNN_DIR = '/home/pablo/reps/musicnn-training'

DATASETS_DATA = {
    # 'ismir04_rhythm':     ['ismir04_rhythm/'],
    'genre_dortmund':     ['genre_dortmund/'],
    'genre_tzanetakis':   ['genre_tzanetakis/'],
    # 'moods_mirex':        ['moods_mirex/'],
    # 'voice_instrumental': ['voice_instrumental/'],
    # 'tonal_atonal':       ['tonal_atonal/'],
    # 'timbre':             ['timbre_bright_dark/'],
    # 'gender':             ['gender/'],
    # 'danceability':       ['danceability/'],
    # 'genre_electronic':   ['genre_electronic/'],
    'genre_rosamerica':   ['genre_rosamerica/audio/mp3/'],
    # 'mood_acoustic':      ['moods_claurier/audio/mp3/acoustic', 'moods_claurier/audio/mp3/not_acoustic'],
    # 'mood_aggressive':    ['moods_claurier/audio/mp3/aggressive', 'moods_claurier/audio/mp3/not_aggressive'],
    # 'mood_electronic':    ['moods_claurier/audio/mp3/electronic', 'moods_claurier/audio/mp3/not_electronic'],
    # 'mood_happy':         ['moods_claurier/audio/mp3/happy', 'moods_claurier/audio/mp3/not_happy'],
    # 'mood_party':         ['moods_claurier/audio/mp3/party', 'moods_claurier/audio/mp3/not_party'],
    # 'mood_relaxed':       ['moods_claurier/audio/mp3/relaxed', 'moods_claurier/audio/mp3/not_relaxed'],
    # 'mood_sad':           ['moods_claurier/audio/mp3/sad', 'moods_claurier/audio/mp3/not_sad']
    }


DATASETS_GROUNDTRUTH = {
    # 'ismir04_rhythm':     'ismir04_rhythm/metadata/groundtruth.yaml',
    'genre_dortmund':     'genre_dortmund/metadata/groundtruth.yaml',
    'genre_tzanetakis':   'genre_tzanetakis/metadata/groundtruth.yaml',
    # 'moods_mirex':        'moods_mirex/metadata/groundtruth.yaml',
    # 'voice_instrumental': 'voice_instrumental/metadata/groundtruth.yaml',
    # 'tonal_atonal':       'tonal_atonal/metadata/groundtruth.yaml',
    # 'timbre':             'timbre_bright_dark/metadata/groundtruth.yaml',
    # 'gender':             'gender/metadata/groundtruth.yaml',
    # 'danceability':       'danceability/metadata/groundtruth.yaml',
    # 'genre_electronic':   'genre_electronic/metadata/groundtruth.yaml',
    'genre_rosamerica':   'genre_rosamerica/metadata/groundtruth.yaml',
    # 'mood_acoustic':      'moods_claurier/metadata/groundtruth_acoustic.yaml',
    # 'mood_aggressive':    'moods_claurier/metadata/groundtruth_aggressive.yaml',
    # 'mood_electronic':    'moods_claurier/metadata/groundtruth_electronic.yaml',
    # 'mood_happy':         'moods_claurier/metadata/groundtruth_happy.yaml',
    # 'mood_party':         'moods_claurier/metadata/groundtruth_party.yaml',
    # 'mood_relaxed':       'moods_claurier/metadata/groundtruth_relaxed.yaml',
    # 'mood_sad':           'moods_claurier/metadata/groundtruth_sad.yaml'
    }
