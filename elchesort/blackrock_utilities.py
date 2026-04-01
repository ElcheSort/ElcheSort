# -*- coding: utf-8 -*-
"""
Collection of classes and utilities for reading headers and data from Blackrock files.
Incorporates brMiscFxns functionality directly.

current version: 2.0.0 --- 04/07/2021

@author: Mitch Frankel - Blackrock Microsystems
         Stephen Hou - v1.4.0 edits
         David Kluger - v2.0.0 overhaul
"""

from __future__ import division  # for Python 2.6+
import numpy as np
from collections import namedtuple
from datetime import datetime
from math import ceil
from os import getcwd, path as ospath
from struct import calcsize, pack, unpack, unpack_from

from qtpy.QtWidgets import QFileDialog, QApplication

# Patch for Python 2.6+
try:
    input = raw_input
except NameError:
    pass


# =============================================================================
# Version Control
# =============================================================================

brpylib_ver    = "2.0.0"
brmiscfxns_ver = "1.2.0"
brmiscfxns_ver_req = "1.2.0"

if brmiscfxns_ver.split('.') < brmiscfxns_ver_req.split('.'):
    raise Exception("brpylib requires brMiscFxns " + brmiscfxns_ver_req + " or higher, please use latest version")


# =============================================================================
# Constants
# =============================================================================

WARNING_SLEEP_TIME      = 5
DATA_PAGING_SIZE        = 1024 ** 3
DATA_FILE_SIZE_MIN      = 1024 ** 2 * 10
STRING_TERMINUS         = '\x00'
UNDEFINED               = 0
ELEC_ID_DEF             = 'all'
START_TIME_DEF          = 0
DATA_TIME_DEF           = 'all'
DOWNSAMPLE_DEF          = 1
START_OFFSET_MIN        = 0
STOP_OFFSET_MIN         = 0

UV_PER_BIT_21             = 0.25
WAVEFORM_SAMPLES_21       = 48
NSX_BASIC_HEADER_BYTES_22 = 314
NSX_EXT_HEADER_BYTES_22   = 66
DATA_BYTE_SIZE            = 2
TIMESTAMP_NULL_21         = 0

NO_FILTER               = 0
BUTTER_FILTER           = 1
SERIAL_MODE             = 0

RB2D_MARKER             = 1
RB2D_BLOB               = 2
RB3D_MARKER             = 3
BOUNDARY_2D             = 4
MARKER_SIZE             = 5

DIGITAL_PACKET_ID       = 0
NEURAL_PACKET_ID_MIN    = 1
NEURAL_PACKET_ID_MAX    = 16384
COMMENT_PACKET_ID       = 65535
VIDEO_SYNC_PACKET_ID    = 65534
TRACKING_PACKET_ID      = 65533
BUTTON_PACKET_ID        = 65532
CONFIGURATION_PACKET_ID = 65531

PARALLEL_REASON         = 1
PERIODIC_REASON         = 64
SERIAL_REASON           = 129
LOWER_BYTE_MASK         = 255
FIRST_BIT_MASK          = 1
SECOND_BIT_MASK         = 2

CLASSIFIER_MIN          = 1
CLASSIFIER_MAX          = 16
CLASSIFIER_NOISE        = 255

CHARSET_ANSI            = 0
CHARSET_UTF             = 1
CHARSET_ROI             = 255

COMM_RGBA               = 0
COMM_TIME               = 1

BUTTON_PRESS            = 1
BUTTON_RESET            = 2

CHG_NORMAL              = 0
CHG_CRITICAL            = 1

ENTER_EVENT             = 1
EXIT_EVENT              = 2


# =============================================================================
# Utility Functions (formerly brMiscFxns)
# =============================================================================

def openfilecheck(open_mode, file_name='', file_ext='', file_type=''):
    """
    :param open_mode: {str} method to open the file (e.g., 'rb' for binary read only)
    :param file_name: [optional] {str} full path of file to open
    :param file_ext:  [optional] {str} file extension (e.g., '.nev')
    :param file_type: [optional] {str} file type for use when browsing for file (e.g., 'Blackrock NEV Files')
    :return: {file} opened file
    """
    while True:
        if not file_name:
            file_name = input("Enter complete " + file_ext + " file path or hit enter to browse: ")

            if not file_name:
                if 'app' not in locals():
                    app = QApplication([])
                if not file_ext:
                    file_type = 'All Files'
                file_name = QFileDialog.getOpenFileName(
                    QFileDialog(), "Select File", getcwd(), file_type + " (*" + file_ext + ")")

        if ospath.isfile(file_name):
            if file_ext:
                _, fext = ospath.splitext(file_name)
                test_extension = file_ext[:-1] if file_ext[-1] == '*' else file_ext
                if fext[:len(test_extension)] != test_extension:
                    file_name = ''
                    print("\n*** File given is not a " + file_ext + " file, try again ***\n")
                    continue
            break
        else:
            file_name = ''
            print("\n*** File given does exist, try again ***\n")

    return open(file_name, open_mode)


def checkequal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True


# =============================================================================
# Header Processing Functions
# =============================================================================

FieldDef = namedtuple('FieldDef', ['name', 'formatStr', 'formatFnc'])


def processheaders(curr_file, packet_fields):
    """
    Read a packet from a binary data file and return a dict of formatted fields.
    """
    packet_format_str = '<' + ''.join([fmt for name, fmt, fun in packet_fields])
    bytes_in_packet = calcsize(packet_format_str)
    packet_binary = curr_file.read(bytes_in_packet)
    packet_unpacked = unpack(packet_format_str, packet_binary)
    data_iter = iter(packet_unpacked)

    return {name: fun(data_iter) for name, fmt, fun in packet_fields}


def format_filespec(header_list):
    return str(next(header_list)) + '.' + str(next(header_list))


def format_timeorigin(header_list):
    year        = next(header_list)
    month       = next(header_list)
    _           = next(header_list)
    day         = next(header_list)
    hour        = next(header_list)
    minute      = next(header_list)
    second      = next(header_list)
    millisecond = next(header_list)
    day = 30 if day >= 30 else day
    return datetime(year, 1, 1, 1, 1, 1, 1 * 1000)


def format_stripstring(header_list):
    string = bytes.decode(next(header_list), 'latin-1')
    return string.split(STRING_TERMINUS, 1)[0]


def format_none(header_list):
    return next(header_list)


def format_freq(header_list):
    return str(float(next(header_list)) / 1000) + ' Hz'


def format_filter(header_list):
    filter_type = next(header_list)
    if filter_type == NO_FILTER:       return "none"
    elif filter_type == BUTTER_FILTER: return "butterworth"


def format_charstring(header_list):
    return int(next(header_list))


def format_digconfig(header_list):
    config = next(header_list) & FIRST_BIT_MASK
    return 'active' if config else 'ignored'


def format_anaconfig(header_list):
    config = next(header_list)
    if config & FIRST_BIT_MASK:  return 'low_to_high'
    if config & SECOND_BIT_MASK: return 'high_to_low'
    return 'none'


def format_digmode(header_list):
    dig_mode = next(header_list)
    return 'serial' if dig_mode == SERIAL_MODE else 'parallel'


def format_trackobjtype(header_list):
    trackobj_type = next(header_list)
    type_map = {
        UNDEFINED:   'undefined',
        RB2D_MARKER: '2D RB markers',
        RB2D_BLOB:   '2D RB blob',
        RB3D_MARKER: '3D RB markers',
        BOUNDARY_2D: '2D boundary',
        MARKER_SIZE: 'marker size',
    }
    return type_map.get(trackobj_type, 'error')


def getdigfactor(ext_headers, idx):
    max_analog  = ext_headers[idx]['MaxAnalogValue']
    min_analog  = ext_headers[idx]['MinAnalogValue']
    max_digital = ext_headers[idx]['MaxDigitalValue']
    min_digital = ext_headers[idx]['MinDigitalValue']
    return float(max_analog - min_analog) / float(max_digital - min_digital)


# =============================================================================
# Header Dictionaries
# =============================================================================

nev_header_dict = {
    'basic': [FieldDef('FileTypeID',            '8s',   format_stripstring),
              FieldDef('FileSpec',              '2B',   format_filespec),
              FieldDef('AddFlags',              'H',    format_none),
              FieldDef('BytesInHeader',         'I',    format_none),
              FieldDef('BytesInDataPackets',    'I',    format_none),
              FieldDef('TimeStampResolution',   'I',    format_none),
              FieldDef('SampleTimeResolution',  'I',    format_none),
              FieldDef('TimeOrigin',            '8H',   format_timeorigin),
              FieldDef('CreatingApplication',   '32s',  format_stripstring),
              FieldDef('Comment',               '256s', format_stripstring),
              FieldDef('NumExtendedHeaders',    'I',    format_none)],

    'ARRAYNME': FieldDef('ArrayName',           '24s',  format_stripstring),
    'ECOMMENT': FieldDef('ExtraComment',        '24s',  format_stripstring),
    'CCOMMENT': FieldDef('ContComment',         '24s',  format_stripstring),
    'MAPFILE':  FieldDef('MapFile',             '24s',  format_stripstring),

    'NEUEVWAV': [FieldDef('ElectrodeID',        'H',    format_none),
                 FieldDef('PhysicalConnector',  'B',    format_charstring),
                 FieldDef('ConnectorPin',       'B',    format_charstring),
                 FieldDef('DigitizationFactor', 'H',    format_none),
                 FieldDef('EnergyThreshold',    'H',    format_none),
                 FieldDef('HighThreshold',      'h',    format_none),
                 FieldDef('LowThreshold',       'h',    format_none),
                 FieldDef('NumSortedUnits',     'B',    format_charstring),
                 FieldDef('BytesPerWaveform',   'B',    format_charstring),
                 FieldDef('SpikeWidthSamples',  'H',    format_none),
                 FieldDef('EmptyBytes',         '8s',   format_none)],

    'NEUEVLBL': [FieldDef('ElectrodeID',        'H',    format_none),
                 FieldDef('Label',              '16s',  format_stripstring),
                 FieldDef('EmptyBytes',         '6s',   format_none)],

    'NEUEVFLT': [FieldDef('ElectrodeID',        'H',    format_none),
                 FieldDef('HighFreqCorner',     'I',    format_freq),
                 FieldDef('HighFreqOrder',      'I',    format_none),
                 FieldDef('HighFreqType',       'H',    format_filter),
                 FieldDef('LowFreqCorner',      'I',    format_freq),
                 FieldDef('LowFreqOrder',       'I',    format_none),
                 FieldDef('LowFreqType',        'H',    format_filter),
                 FieldDef('EmptyBytes',         '2s',   format_none)],

    'DIGLABEL': [FieldDef('Label',              '16s',  format_stripstring),
                 FieldDef('Mode',               '?',    format_digmode),
                 FieldDef('EmptyBytes',         '7s',   format_none)],

    'NSASEXEV': [FieldDef('Frequency',          'H',    format_none),
                 FieldDef('DigitalInputConfig', 'B',    format_digconfig),
                 FieldDef('AnalogCh1Config',    'B',    format_anaconfig),
                 FieldDef('AnalogCh1DetectVal', 'h',    format_none),
                 FieldDef('AnalogCh2Config',    'B',    format_anaconfig),
                 FieldDef('AnalogCh2DetectVal', 'h',    format_none),
                 FieldDef('AnalogCh3Config',    'B',    format_anaconfig),
                 FieldDef('AnalogCh3DetectVal', 'h',    format_none),
                 FieldDef('AnalogCh4Config',    'B',    format_anaconfig),
                 FieldDef('AnalogCh4DetectVal', 'h',    format_none),
                 FieldDef('AnalogCh5Config',    'B',    format_anaconfig),
                 FieldDef('AnalogCh5DetectVal', 'h',    format_none),
                 FieldDef('EmptyBytes',         '6s',   format_none)],

    'VIDEOSYN': [FieldDef('VideoSourceID',      'H',    format_none),
                 FieldDef('VideoSource',        '16s',  format_stripstring),
                 FieldDef('FrameRate',          'f',    format_none),
                 FieldDef('EmptyBytes',         '2s',   format_none)],

    'TRACKOBJ': [FieldDef('TrackableType',      'H',    format_trackobjtype),
                 FieldDef('TrackableID',        'I',    format_none),
                 FieldDef('VideoSource',        '16s',  format_stripstring),
                 FieldDef('EmptyBytes',         '2s',   format_none)]
}

nsx_header_dict = {
    'basic_21': [FieldDef('Label',              '16s', format_stripstring),
                 FieldDef('Period',             'I',   format_none),
                 FieldDef('ChannelCount',       'I',   format_none)],

    'basic': [FieldDef('FileSpec',              '2B',   format_filespec),
              FieldDef('BytesInHeader',         'I',    format_none),
              FieldDef('Label',                 '16s',  format_stripstring),
              FieldDef('Comment',               '256s', format_stripstring),
              FieldDef('Period',                'I',    format_none),
              FieldDef('TimeStampResolution',   'I',    format_none),
              FieldDef('TimeOrigin',            '8H',   format_timeorigin),
              FieldDef('ChannelCount',          'I',    format_none)],

    'extended': [FieldDef('Type',               '2s',   format_stripstring),
                 FieldDef('ElectrodeID',        'H',    format_none),
                 FieldDef('ElectrodeLabel',     '16s',  format_stripstring),
                 FieldDef('PhysicalConnector',  'B',    format_none),
                 FieldDef('ConnectorPin',       'B',    format_none),
                 FieldDef('MinDigitalValue',    'h',    format_none),
                 FieldDef('MaxDigitalValue',    'h',    format_none),
                 FieldDef('MinAnalogValue',     'h',    format_none),
                 FieldDef('MaxAnalogValue',     'h',    format_none),
                 FieldDef('Units',              '16s',  format_stripstring),
                 FieldDef('HighFreqCorner',     'I',    format_freq),
                 FieldDef('HighFreqOrder',      'I',    format_none),
                 FieldDef('HighFreqType',       'H',    format_filter),
                 FieldDef('LowFreqCorner',      'I',    format_freq),
                 FieldDef('LowFreqOrder',       'I',    format_none),
                 FieldDef('LowFreqType',        'H',    format_filter)],

    'data': [FieldDef('Header',                 'B',    format_none),
             FieldDef('Timestamp',              'I',    format_none),
             FieldDef('NumDataPoints',          'I',    format_none)]
}


# =============================================================================
# Safety Check Functions
# =============================================================================

def check_elecid(elec_ids):
    if type(elec_ids) is str and elec_ids != ELEC_ID_DEF:
        print("\n*** WARNING: Electrode IDs must be 'all', a single integer, or a list of integers.")
        print("      Setting elec_ids to 'all'")
        return ELEC_ID_DEF
    if elec_ids != ELEC_ID_DEF and type(elec_ids) is not list:
        if type(elec_ids) == range:  return list(elec_ids)
        elif type(elec_ids) == int:  return [elec_ids]
    return elec_ids


def check_starttime(start_time_s):
    if not isinstance(start_time_s, (int, float)) or \
            (isinstance(start_time_s, (int, float)) and start_time_s < START_TIME_DEF):
        print("\n*** WARNING: Start time is not valid, setting start_time_s to 0")
        return START_TIME_DEF
    return start_time_s


def check_datatime(data_time_s):
    if (type(data_time_s) is str and data_time_s != DATA_TIME_DEF) or \
            (isinstance(data_time_s, (int, float)) and data_time_s < 0):
        print("\n*** WARNING: Data time is not valid, setting data_time_s to 'all'")
        return DATA_TIME_DEF
    return data_time_s


def check_downsample(downsample):
    if not isinstance(downsample, int) or downsample < DOWNSAMPLE_DEF:
        print("\n*** WARNING: Downsample must be an integer value greater than 0. "
              "      Setting downsample to 1 (no downsampling)")
        return DOWNSAMPLE_DEF
    return downsample


def check_dataelecid(elec_ids, all_elec_ids):
    unique_elec_ids = set(elec_ids)
    all_elec_ids    = set(all_elec_ids)

    if not unique_elec_ids.issubset(all_elec_ids):
        if not unique_elec_ids & all_elec_ids:
            print('\nNone of the elec_ids passed exist in the data, returning None')
            return None
        else:
            print("\n*** WARNING: Channels " + str(sorted(list(unique_elec_ids - all_elec_ids))) +
                  " do not exist in the data")
            unique_elec_ids = unique_elec_ids & all_elec_ids

    return sorted(list(unique_elec_ids))


def check_filesize(file_size):
    if file_size < DATA_FILE_SIZE_MIN:
        print('\n file_size must be larger than 10 Mb, setting file_size=10 Mb')
        return DATA_FILE_SIZE_MIN
    return int(file_size)


# =============================================================================
# NevFile Class
# =============================================================================

class NevFile:
    """
    Attributes and methods for all BR event data files.  Initialization opens the file and extracts the
    basic header information.
    """

    def __init__(self, datafile=''):
        self.datafile         = datafile
        self.basic_header     = {}
        self.extended_headers = []

        self.datafile = openfilecheck('rb', file_name=self.datafile, file_ext='.nev', file_type='Blackrock NEV Files')
        self.basic_header = processheaders(self.datafile, nev_header_dict['basic'])

        for i in range(self.basic_header['NumExtendedHeaders']):
            self.extended_headers.append({})
            header_string = bytes.decode(unpack('<8s', self.datafile.read(8))[0], 'latin-1')
            self.extended_headers[i]['PacketID'] = header_string.split(STRING_TERMINUS, 1)[0]
            self.extended_headers[i].update(
                processheaders(self.datafile, nev_header_dict[self.extended_headers[i]['PacketID']]))

            if header_string == 'NEUEVWAV' and float(self.basic_header['FileSpec']) < 2.3:
                self.extended_headers[i]['SpikeWidthSamples'] = WAVEFORM_SAMPLES_21

    def getdata(self, elec_ids='all', wave_read='read'):
        """
        This function is used to return a set of data from the NEV datafile.

        :param elec_ids:   [optional] {list} User selection of elec_ids to extract specific spike waveforms
        :param wave_read:  [optional] {str}  'read' or 'no_read' - whether to read waveforms or not
        :return: output:   {dict} with one or more of the following:
                    spike_events, digital_events, comments, tracking_events,
                    video_sync_events, tracking, PatientTrigger, reconfig
        """
        output   = dict()
        elec_ids = check_elecid(elec_ids)

        # Extract raw data
        self.datafile.seek(0, 2)
        lData    = self.datafile.tell()
        nPackets = int((lData - self.basic_header['BytesInHeader']) / self.basic_header['BytesInDataPackets'])
        self.datafile.seek(self.basic_header['BytesInHeader'], 0)
        rawdata = self.datafile.read()

        rawdataArray = np.reshape(
            np.frombuffer(rawdata, 'B'),
            (nPackets, self.basic_header['BytesInDataPackets']))

        # Find all timestamps and PacketIDs
        tsBytes  = 4 if self.basic_header['FileTypeID'] == 'BREVENTS' else 2
        stride   = (self.basic_header['BytesInDataPackets'],)
        ts       = np.ndarray((nPackets,), '<I', rawdata, 0, stride)
        PacketID = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 2, stride)

        # --- Neural / spike data ---
        neuralPackets = [idx for idx, el in enumerate(PacketID)
                         if NEURAL_PACKET_ID_MIN <= el <= NEURAL_PACKET_ID_MAX]
        if neuralPackets:
            ChannelID = PacketID
            if type(elec_ids) is list:
                elecindices = [idx for idx, el in enumerate(ChannelID[neuralPackets]) if el in elec_ids]
                neuralPackets = [neuralPackets[index] for index in elecindices]

            spikeUnit = np.ndarray((nPackets,), '<B', rawdata, tsBytes + 4, stride)
            output['spike_events'] = {
                'TimeStamps': list(ts[neuralPackets]),
                'Unit':       list(spikeUnit[neuralPackets]),
                'Channel':    list(ChannelID[neuralPackets])}

            if wave_read == 'read':
                wf_samples = int((self.basic_header['BytesInDataPackets'] - (tsBytes + 10)) / 2)
                wfs = np.ndarray(
                    (nPackets, wf_samples), '<h', rawdata, tsBytes + 10,
                    (self.basic_header['BytesInDataPackets'], 2))
                output['spike_events']['Waveforms'] = wfs[neuralPackets, :]

        # --- Digital events ---
        digiPackets = [idx for idx, el in enumerate(PacketID) if el == DIGITAL_PACKET_ID]
        if digiPackets:
            insertionReason = np.ndarray((nPackets,), '<B', rawdata, tsBytes + 4, stride)
            digiVals        = np.ndarray((nPackets,), '<I', rawdata, tsBytes + 6, stride)
            output['digital_events'] = {
                'TimeStamps':      list(ts[digiPackets]),
                'InsertionReason': list(insertionReason[digiPackets]),
                'UnparsedData':    list(digiVals[digiPackets])}

        # --- Comments + NeuroMotive ROI events ---
        commentPackets = [idx for idx, el in enumerate(PacketID) if el == COMMENT_PACKET_ID]
        if commentPackets:
            charSet   = np.ndarray((nPackets,), '<B', rawdata, tsBytes + 4, stride)
            tsStarted = np.ndarray((nPackets,), '<I', rawdata, tsBytes + 6, stride)
            charSet   = charSet[commentPackets]

            charSetList = np.array([None] * len(charSet))
            ANSIPackets = [idx for idx, el in enumerate(charSet) if el == CHARSET_ANSI]
            if ANSIPackets:
                charSetList[ANSIPackets] = 'ANSI'
            UTFPackets = [idx for idx, el in enumerate(charSet) if el == CHARSET_UTF]
            if UTFPackets:
                charSetList[UTFPackets] = 'UTF '

            ROIPackets = [idx for idx, el in enumerate(charSet) if el == CHARSET_ROI]

            lcomment = self.basic_header['BytesInDataPackets'] - tsBytes - 10
            comments = np.chararray(
                (nPackets, lcomment), 1, False, rawdata, tsBytes + 10,
                (self.basic_header['BytesInDataPackets'], 1))

            # Extract "true" comments (not ROI packets)
            trueComments    = np.setdiff1d(list(range(0, len(commentPackets) - 1)), ROIPackets)
            trueCommentsidx = np.asarray(commentPackets)[trueComments]
            textComments    = comments[trueCommentsidx]
            textComments[:, -1] = '$'
            stringarray  = textComments.tostring()
            stringvector = stringarray.decode('latin-1')[:-1]
            validstrings = stringvector.replace('\x00', '')
            commentsFinal = validstrings.split('$')

            subsetInds = list(set(list(range(0, len(charSetList) - 1))) - set(ROIPackets))

            output['comments'] = {
                'TimeStamps':        list(ts[trueCommentsidx]),
                'TimeStampsStarted': list(tsStarted[trueCommentsidx]),
                'Data':              commentsFinal,
                'CharSet':           list(charSetList[subsetInds])}

            # Parse ROI events
            if ROIPackets:
                nmCommentsidx = np.asarray(commentPackets)[ROIPackets]
                nmcomments = comments[nmCommentsidx]
                nmcomments[:, -1] = ':'
                nmstringarray  = nmcomments.tostring()
                nmstringvector = nmstringarray.decode('latin-1')[:-1]
                nmvalidstrings = nmstringvector.replace('\x00', '')
                nmcommentsFinal = nmvalidstrings.split(':')
                ROIfields   = [l.split(':') for l in ':'.join(nmcommentsFinal).split(':')]
                ROIfieldsRS = np.reshape(ROIfields, (len(ROIPackets), 5))
                output['tracking_events'] = {
                    'TimeStamps': list(ts[nmCommentsidx]),
                    'ROIName':    list(ROIfieldsRS[:, 0]),
                    'ROINumber':  list(ROIfieldsRS[:, 1]),
                    'Event':      list(ROIfieldsRS[:, 2]),
                    'Frame':      list(ROIfieldsRS[:, 3])}

        # --- Video sync packets ---
        vidsyncPackets = [idx for idx, el in enumerate(PacketID) if el == VIDEO_SYNC_PACKET_ID]
        if vidsyncPackets:
            fileNumber  = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 4,  stride)
            frameNumber = np.ndarray((nPackets,), '<I', rawdata, tsBytes + 6,  stride)
            elapsedTime = np.ndarray((nPackets,), '<I', rawdata, tsBytes + 10, stride)
            sourceID    = np.ndarray((nPackets,), '<I', rawdata, tsBytes + 14, stride)
            output['video_sync_events'] = {
                'TimeStamps':  list(ts[vidsyncPackets]),
                'FileNumber':  list(fileNumber[vidsyncPackets]),
                'FrameNumber': list(frameNumber[vidsyncPackets]),
                'ElapsedTime': list(elapsedTime[vidsyncPackets]),
                'SourceID':    list(sourceID[vidsyncPackets])}

        # --- Object tracking packets ---
        trackingPackets = [idx for idx, el in enumerate(PacketID) if el == TRACKING_PACKET_ID]
        if trackingPackets:
            trackerObjs = [sub['VideoSource'] for sub in self.extended_headers if sub['PacketID'] == 'TRACKOBJ']
            trackerIDs  = [sub['TrackableID'] for sub in self.extended_headers if sub['PacketID'] == 'TRACKOBJ']
            output['tracking'] = {
                'TrackerIDs':   trackerIDs,
                'TrackerTypes': [sub['TrackableType'] for sub in self.extended_headers if sub['PacketID'] == 'TRACKOBJ']}

            parentID    = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 4,  stride)
            nodeID      = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 6,  stride)
            nodeCount   = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 8,  stride)
            markerCount = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 10, stride)
            bodyPointsX = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 12, stride)
            bodyPointsY = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 14, stride)

            R, E = 0, 0
            for i in range(len(trackerObjs)):
                indices = [idx for idx, el in enumerate(nodeID[trackingPackets]) if el == i]

                if len(indices) == 1:
                    if trackerObjs[i] == 'TrackingROI':
                        trackerObjs[i] = trackerObjs[i] + str(R)
                        R += 1
                    elif trackerObjs[i] == 'EventROI':
                        trackerObjs[i] = trackerObjs[i] + str(E)
                        E += 1
                    bodyPointsX = np.ndarray(
                        (nPackets, 4), '<H', rawdata, tsBytes + 12,
                        (self.basic_header['BytesInDataPackets'], 2))
                    bodyPointsY = np.ndarray(
                        (nPackets, 4), '<H', rawdata, tsBytes + 14,
                        (self.basic_header['BytesInDataPackets'], 2))

                selectedIndices = [trackingPackets[index] for index in indices]
                output['tracking'][trackerObjs[i]] = {
                    'TimeStamps':  list(ts[selectedIndices]),
                    'ParentID':    list(parentID[selectedIndices]),
                    'NodeCount':   list(nodeCount[selectedIndices]),
                    'MarkerCount': list(markerCount[selectedIndices]),
                    'X':           list(bodyPointsX[selectedIndices]),
                    'Y':           list(bodyPointsY[selectedIndices])}

            output['tracking']['TrackerObjs'] = trackerObjs

        # --- Patient trigger events ---
        buttonPackets = [idx for idx, el in enumerate(PacketID) if el == BUTTON_PACKET_ID]
        if buttonPackets:
            trigType = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 4, stride)
            output['PatientTrigger'] = {
                'TimeStamps':  list(ts[buttonPackets]),
                'TriggerType': list(trigType[buttonPackets])}

        # --- Configuration packets ---
        configPackets = [idx for idx, el in enumerate(PacketID) if el == CONFIGURATION_PACKET_ID]
        if configPackets:
            changeType = np.ndarray((nPackets,), '<H', rawdata, tsBytes + 4, stride)
            output['reconfig'] = {
                'TimeStamps': list(ts[configPackets]),
                'ChangeType': list(ts[configPackets])}

        return output

    def processroicomments(self, comments):
        """
        Obsolete in v2.0.0+, ROI comments come out parsed from NevFile.getdata().
        Process comment data packets associated with NeuroMotive ROI enter/exit events.
        """
        roi_events = {'Regions': [], 'EnterTimeStamps': [], 'ExitTimeStamps': []}

        for i in range(len(comments['TimeStamps'])):
            if comments['CharSet'][i] == 'NeuroMotive ROI':
                temp_data = pack('<I', comments['Data'][i])
                roi   = unpack_from('<B', temp_data)[0]
                event = unpack_from('<B', temp_data, 1)[0]

                source_label = next(d['VideoSource'] for d in self.extended_headers if d["TrackableID"] == roi)

                if source_label in roi_events['Regions']:
                    idx = roi_events['Regions'].index(source_label)
                else:
                    idx = -1
                    roi_events['Regions'].append(source_label)
                    roi_events['EnterTimeStamps'].append([])
                    roi_events['ExitTimeStamps'].append([])

                if   event == ENTER_EVENT: roi_events['EnterTimeStamps'][idx].append(comments['TimeStamp'][i])
                elif event == EXIT_EVENT:  roi_events['ExitTimeStamps'][idx].append(comments['TimeStamp'][i])

        return roi_events

    def close(self):
        name = self.datafile.name
        self.datafile.close()
        print('\n' + name.split('/')[-1] + ' closed')


# =============================================================================
# NsxFile Class
# =============================================================================

class NsxFile:
    """
    Attributes and methods for all BR continuous data files.  Initialization opens the file and extracts the
    basic header information.
    """

    def __init__(self, datafile=''):
        self.datafile         = datafile
        self.basic_header     = {}
        self.extended_headers = []

        self.datafile = openfilecheck('rb', file_name=self.datafile, file_ext='.ns*', file_type='Blackrock NSx Files')

        # Determine File ID to determine if File Spec 2.1
        self.basic_header['FileTypeID'] = bytes.decode(self.datafile.read(8), 'latin-1')

        if self.basic_header['FileTypeID'] == 'NEURALSG':
            self.basic_header.update(processheaders(self.datafile, nsx_header_dict['basic_21']))
            self.basic_header['FileSpec']            = '2.1'
            self.basic_header['TimeStampResolution'] = 30000
            self.basic_header['BytesInHeader']       = 32 + 4 * self.basic_header['ChannelCount']
            shape = (1, self.basic_header['ChannelCount'])
            self.basic_header['ChannelID'] = list(
                np.fromfile(file=self.datafile, dtype=np.uint32,
                            count=self.basic_header['ChannelCount']).reshape(shape)[0])
        else:
            self.basic_header.update(processheaders(self.datafile, nsx_header_dict['basic']))
            for i in range(self.basic_header['ChannelCount']):
                self.extended_headers.append(processheaders(self.datafile, nsx_header_dict['extended']))

    def getdata(self, elec_ids='all', start_time_s=0, data_time_s='all', downsample=1):
        """
        This function is used to return a set of data from the NSx datafile.

        :param elec_ids:      [optional] {list}  List of elec_ids to extract (e.g., [13])
        :param start_time_s:  [optional] {float} Starting time for data extraction (e.g., 1.0)
        :param data_time_s:   [optional] {float} Length of time of data to return (e.g., 30.0)
        :param downsample:    [optional] {int}   Downsampling factor (e.g., 2)
        :return: output:      {dict} of: data_headers, elec_ids, start_time_s, data_time_s,
                               downsample, samp_per_s, data (numpy array)
        """

        # Safety checks
        start_time_s = check_starttime(start_time_s)
        data_time_s  = check_datatime(data_time_s)
        downsample   = check_downsample(downsample)
        elec_ids     = check_elecid(elec_ids)

        # Initialize output
        output = {
            'elec_ids':              elec_ids,
            'start_time_s':          float(start_time_s),
            'data_time_s':           data_time_s,
            'downsample':            downsample,
            'data':                  [],
            'data_headers':          [],
            'ExtendedHeaderIndices': [],
        }

        datafile_samp_per_sec = self.basic_header['TimeStampResolution'] / self.basic_header['Period']
        data_pt_size          = self.basic_header['ChannelCount'] * DATA_BYTE_SIZE
        elec_id_indices       = []
        front_end_idxs        = []
        analog_input_idxs     = []
        front_end_idx_cont    = True
        analog_input_idx_cont = True
        hit_start             = False
        hit_stop              = False
        d_ptr                 = 0

        self.datafile.seek(self.basic_header['BytesInHeader'], 0)

        # Set elec_ids and data headers based on FileSpec
        if self.basic_header['FileSpec'] == '2.1':
            output['elec_ids'] = self.basic_header['ChannelID']
            output['data_headers'].append({})
            output['data_headers'][0]['Timestamp']     = TIMESTAMP_NULL_21
            output['data_headers'][0]['NumDataPoints'] = (ospath.getsize(self.datafile.name) - self.datafile.tell()) \
                                                         // (DATA_BYTE_SIZE * self.basic_header['ChannelCount'])
        else:
            output['elec_ids'] = [d['ElectrodeID'] for d in self.extended_headers]

        # Determine start and stop indices
        start_idx = START_OFFSET_MIN if start_time_s == START_TIME_DEF else int(round(start_time_s * datafile_samp_per_sec))
        stop_idx  = STOP_OFFSET_MIN  if data_time_s == DATA_TIME_DEF  else int(round((start_time_s + data_time_s) * datafile_samp_per_sec))

        # If a subset of electrodes is requested, validate and determine indices
        if elec_ids != ELEC_ID_DEF:
            elec_ids = check_dataelecid(elec_ids, output['elec_ids'])
            if not elec_ids:
                return output
            elec_id_indices    = [output['elec_ids'].index(e) for e in elec_ids]
            output['elec_ids'] = elec_ids
        num_elecs = len(output['elec_ids'])

        # Determine extended header indices and Front End vs. Analog Input channels
        if self.basic_header['FileSpec'] != '2.1':
            for i in range(num_elecs):
                idx = next(item for (item, d) in enumerate(self.extended_headers)
                           if d["ElectrodeID"] == output['elec_ids'][i])
                output['ExtendedHeaderIndices'].append(idx)

                if self.extended_headers[idx]['PhysicalConnector'] < 5:
                    front_end_idxs.append(i)
                else:
                    analog_input_idxs.append(i)

            if any(np.diff(np.array(front_end_idxs)) != 1):    front_end_idx_cont    = False
            if any(np.diff(np.array(analog_input_idxs)) != 1): analog_input_idx_cont = False

        # Pre-allocate output data
        if self.basic_header['FileSpec'] == '2.1':
            timestamp    = TIMESTAMP_NULL_21
            num_data_pts = output['data_headers'][0]['NumDataPoints']
        else:
            while self.datafile.tell() != ospath.getsize(self.datafile.name):
                self.datafile.seek(1, 1)
                timestamp    = unpack('<I', self.datafile.read(4))[0]
                num_data_pts = unpack('<I', self.datafile.read(4))[0]
                self.datafile.seek(num_data_pts * self.basic_header['ChannelCount'] * DATA_BYTE_SIZE, 1)

        stop_idx_output = ceil(timestamp / self.basic_header['Period']) + num_data_pts
        if data_time_s != DATA_TIME_DEF and stop_idx < stop_idx_output:
            stop_idx_output = stop_idx
        total_samps = int(ceil((stop_idx_output - start_idx) / downsample))

        if (total_samps * self.basic_header['ChannelCount'] * DATA_BYTE_SIZE) > DATA_PAGING_SIZE:
            print("\nOutput data requested is larger than 1 GB, attempting to preallocate output now")

        try:
            output['data'] = np.zeros((total_samps, num_elecs), dtype=np.float32)
        except MemoryError as err:
            err.args += (" Output data size requested is larger than available memory. Use the parameters\n"
                         "              for getdata(), e.g., 'elec_ids', to request a subset of the data or use\n"
                         "              NsxFile.savesubsetnsx() to create subsets of the main nsx file\n", )
            raise

        # Loop through all data packets
        self.datafile.seek(self.basic_header['BytesInHeader'], 0)
        while not hit_stop:

            if self.basic_header['FileSpec'] != '2.1':
                output['data_headers'].append(processheaders(self.datafile, nsx_header_dict['data']))
                if output['data_headers'][-1]['Header'] == 0:
                    print('Invalid Header.  File may be corrupt')
                if output['data_headers'][-1]['NumDataPoints'] < downsample:
                    self.datafile.seek(self.basic_header['ChannelCount'] * output['data_headers'][-1]['NumDataPoints']
                                       * DATA_BYTE_SIZE, 1)
                    continue

            timestamp_sample = int(round(output['data_headers'][-1]['Timestamp'] / self.basic_header['Period']))

            # Patch for file sync (2 NSP clocks)
            if timestamp_sample < d_ptr:
                d_ptr = 0
                hit_start = False
                output['data_headers'] = []
                self.datafile.seek(-9, 1)
                continue

            if len(output['data_headers']) == 1 and (STOP_OFFSET_MIN < stop_idx < timestamp_sample):
                print("\nData requested is before any data was saved, which starts at t = {0:.6f} s".format(
                    output['data_headers'][0]['Timestamp'] / self.basic_header['TimeStampResolution']))
                return

            # First data packet
            if not hit_start:
                start_offset = start_idx - timestamp_sample

                if start_offset > output['data_headers'][-1]['NumDataPoints']:
                    self.datafile.seek(output['data_headers'][-1]['NumDataPoints'] * data_pt_size, 1)
                    if self.datafile.tell() == ospath.getsize(self.datafile.name):
                        break
                    continue
                else:
                    if start_offset < 0:
                        if STOP_OFFSET_MIN < stop_idx < timestamp_sample:
                            print("\nBecause of pausing, data section requested is during pause period")
                            return
                        else:
                            print("\nFirst data packet requested begins at t = {0:.6f} s, "
                                  "initial section padded with zeros".format(
                                   output['data_headers'][-1]['Timestamp'] / self.basic_header['TimeStampResolution']))
                            start_offset = START_OFFSET_MIN
                            d_ptr        = (timestamp_sample - start_idx) // downsample
                    hit_start = True

            # Subsequent data packets
            else:
                if STOP_OFFSET_MIN < stop_idx < timestamp_sample:
                    print("\nSection padded with zeros due to file pausing")
                    hit_stop = True
                    break
                elif (timestamp_sample - start_idx) > d_ptr:
                    print("\nSection padded with zeros due to file pausing")
                    start_offset = START_OFFSET_MIN
                    d_ptr        = (timestamp_sample - start_idx) // downsample

            # Set number of samples to read
            if STOP_OFFSET_MIN < stop_idx <= (timestamp_sample + output['data_headers'][-1]['NumDataPoints']):
                total_pts = stop_idx - timestamp_sample - start_offset
                hit_stop = True
            else:
                total_pts = output['data_headers'][-1]['NumDataPoints'] - start_offset

            curr_file_pos = self.datafile.tell()
            file_offset   = int(curr_file_pos + start_offset * data_pt_size)

            # Extract data with paging (max 1 GB at a time)
            downsample_data_size = data_pt_size * downsample
            max_length           = (DATA_PAGING_SIZE // downsample_data_size) * downsample_data_size
            num_loops            = int(ceil(total_pts * data_pt_size / max_length))

            for loop in range(num_loops):
                if loop == 0:
                    num_pts = total_pts if num_loops == 1 else max_length // data_pt_size
                else:
                    file_offset += max_length
                    if loop == (num_loops - 1):
                        num_pts = ((total_pts * data_pt_size) % max_length) // data_pt_size
                    else:
                        num_pts = max_length // data_pt_size

                if num_loops != 1:
                    print('Data extraction requires paging: {0} of {1}'.format(loop + 1, num_loops))

                num_pts = int(num_pts)
                shape   = (num_pts, self.basic_header['ChannelCount'])
                mm      = np.memmap(self.datafile, dtype=np.int16, mode='r', offset=file_offset, shape=shape)

                if downsample != 1:
                    mm = mm[::downsample]
                if elec_id_indices:
                    output['data'][d_ptr:d_ptr + mm.shape[0]] = np.array(mm[:, elec_id_indices]).astype(np.float32)
                else:
                    output['data'][d_ptr:d_ptr + mm.shape[0]] = np.array(mm).astype(np.float32)
                d_ptr += mm.shape[0]
                del mm

            # Reset file position for next header
            curr_file_pos += self.basic_header['ChannelCount'] * output['data_headers'][-1]['NumDataPoints'] \
                             * DATA_BYTE_SIZE
            self.datafile.seek(curr_file_pos, 0)
            if curr_file_pos == ospath.getsize(self.datafile.name):
                hit_stop = True

        # Final validation
        if not hit_stop and start_idx > START_OFFSET_MIN:
            raise Exception('Error: End of file found before start_time_s')
        elif not hit_stop and stop_idx:
            print("\n*** WARNING: End of file found before stop_time_s, returning all data in file")

        # Transpose: rows = electrodes, columns = samples
        output['data'] = output['data'].transpose()

        # Scale data based on extended header factors
        if self.basic_header['FileSpec'] == '2.1':
            output['data'] *= UV_PER_BIT_21
        else:
            for idx_list, is_cont in [(front_end_idxs, front_end_idx_cont),
                                      (analog_input_idxs, analog_input_idx_cont)]:
                if idx_list:
                    if is_cont:
                        output['data'][idx_list[0]:idx_list[-1] + 1] *= \
                            getdigfactor(self.extended_headers, output['ExtendedHeaderIndices'][idx_list[0]])
                    else:
                        for i in idx_list:
                            output['data'][i] *= getdigfactor(self.extended_headers, output['ExtendedHeaderIndices'][i])

        output['samp_per_s']  = float(datafile_samp_per_sec / downsample)
        output['data_time_s'] = len(output['data'][0]) / output['samp_per_s']

        return output

    def savesubsetnsx(self, elec_ids='all', file_size=None, file_time_s=None, file_suffix=''):
        """
        Save a subset of data based on electrode IDs, file sizing, or file data time.
        If both file_time_s and file_size are passed, defaults to file_time_s.

        :param elec_ids:    [optional] {list}  List of elec_ids to extract
        :param file_size:   [optional] {int}   Byte size of each subset file
        :param file_time_s: [optional] {float} Time length of data for each subset file, in seconds
        :param file_suffix: [optional] {str}   Suffix to append to filename for subset files
        :return: None or "SUCCESS"
        """

        # Initializations
        elec_id_indices      = []
        file_num             = 1
        pausing              = False
        datafile_datapt_size = self.basic_header['ChannelCount'] * DATA_BYTE_SIZE
        self.datafile.seek(0, 0)

        # Electrode ID checks
        elec_ids = check_elecid(elec_ids)
        if self.basic_header['FileSpec'] == '2.1':
            all_elec_ids = self.basic_header['ChannelID']
        else:
            all_elec_ids = [x['ElectrodeID'] for x in self.extended_headers]

        if elec_ids == ELEC_ID_DEF:
            elec_ids = all_elec_ids
        else:
            elec_ids = check_dataelecid(elec_ids, all_elec_ids)
            if not elec_ids:
                return None
            elec_id_indices = [all_elec_ids.index(x) for x in elec_ids]

        num_elecs = len(elec_ids)

        # Determine file sizing
        if file_time_s:
            if file_size:
                print("\nWARNING: Only one of file_size or file_time_s can be passed, defaulting to file_time_s.")
            file_size = int(num_elecs * DATA_BYTE_SIZE * file_time_s *
                            self.basic_header['TimeStampResolution'] / self.basic_header['Period'])
            if self.basic_header['FileSpec'] == '2.1':
                file_size += 32 + 4 * num_elecs
            else:
                file_size += NSX_BASIC_HEADER_BYTES_22 + NSX_EXT_HEADER_BYTES_22 * num_elecs + 5
            print("\nBased on timing request, file size will be {0:d} Mb".format(int(file_size / 1024 ** 2)))
        elif file_size:
            file_size = check_filesize(file_size)

        # Create subset file
        file_name, file_ext = ospath.splitext(self.datafile.name)
        file_name += ('_' + file_suffix) if file_suffix else '_subset'

        if ospath.isfile(file_name + "_000" + file_ext):
            if 'y' != input("\nFile '" + file_name.split('/')[-1] + "_xxx" + file_ext +
                            "' already exists, overwrite [y/n]: "):
                print("\nExiting, no overwrite, returning None")
                return None
            else:
                print("\n*** Overwriting existing subset files ***")

        subset_file = open(file_name + "_000" + file_ext, 'wb')
        print("\nWriting subset file: " + ospath.split(subset_file.name)[1])

        # Write headers based on file spec
        if self.basic_header['FileSpec'] == '2.1':
            subset_file.write(self.datafile.read(28))
            subset_file.write(np.array(num_elecs).astype(np.uint32).tobytes())
            subset_file.write(np.array(elec_ids).astype(np.uint32).tobytes())
            self.datafile.seek(4 + 4 * self.basic_header['ChannelCount'], 1)
        else:
            subset_file.write(self.datafile.read(10))
            bytes_in_headers = NSX_BASIC_HEADER_BYTES_22 + NSX_EXT_HEADER_BYTES_22 * num_elecs
            num_pts_header_pos = bytes_in_headers + 5
            subset_file.write(np.array(bytes_in_headers).astype(np.uint32).tobytes())
            self.datafile.seek(4, 1)
            subset_file.write(self.datafile.read(296))
            subset_file.write(np.array(num_elecs).astype(np.uint32).tobytes())
            self.datafile.seek(4, 1)

            for i in range(len(self.extended_headers)):
                h_type  = self.datafile.read(2)
                chan_id = self.datafile.read(2)
                if unpack('<H', chan_id)[0] in elec_ids:
                    subset_file.write(h_type)
                    subset_file.write(chan_id)
                    subset_file.write(self.datafile.read(62))
                else:
                    self.datafile.seek(62, 1)

        # Loop through all data packets
        while self.datafile.tell() != ospath.getsize(self.datafile.name):

            if self.basic_header['FileSpec'] == '2.1':
                packet_pts = (ospath.getsize(self.datafile.name) - self.datafile.tell()) \
                             / (DATA_BYTE_SIZE * self.basic_header['ChannelCount'])
            else:
                header_binary     = self.datafile.read(1)
                timestamp_binary  = self.datafile.read(4)
                packet_pts_binary = self.datafile.read(4)
                packet_pts        = unpack('<I', packet_pts_binary)[0]
                if packet_pts == 0:
                    continue
                subset_file.write(header_binary)
                subset_file.write(timestamp_binary)
                subset_file.write(packet_pts_binary)

            datafile_pos        = self.datafile.tell()
            file_offset         = datafile_pos
            mm_length           = (DATA_PAGING_SIZE // datafile_datapt_size) * datafile_datapt_size
            num_loops           = int(ceil(packet_pts * datafile_datapt_size / mm_length))
            packet_read_pts     = 0
            subset_file_pkt_pts = 0

            for loop in range(num_loops):
                if loop == 0:
                    num_pts = packet_pts if num_loops == 1 else mm_length // datafile_datapt_size
                else:
                    file_offset += mm_length
                    if loop == (num_loops - 1):
                        num_pts = ((packet_pts * datafile_datapt_size) % mm_length) // datafile_datapt_size
                    else:
                        num_pts = mm_length // datafile_datapt_size

                shape = (int(num_pts), self.basic_header['ChannelCount'])
                mm = np.memmap(self.datafile, dtype=np.int16, mode='r', offset=file_offset, shape=shape)
                if elec_id_indices:
                    mm = mm[:, elec_id_indices]
                start_idx = 0

                # Check if we need to start an additional file
                if file_size and (file_size - subset_file.tell()) < DATA_PAGING_SIZE:
                    pts_can_add = int((file_size - subset_file.tell()) // (num_elecs * DATA_BYTE_SIZE)) + 1
                    stop_idx    = start_idx + pts_can_add

                    while pts_can_add < num_pts:
                        if elec_id_indices:
                            subset_file.write(np.array(mm[start_idx:stop_idx]).tobytes())
                        else:
                            subset_file.write(mm[start_idx:stop_idx])

                        prior_file_name    = subset_file.name
                        prior_file_pkt_pts = subset_file_pkt_pts + pts_can_add
                        subset_file.close()

                        prior_file = open(prior_file_name, 'rb+')
                        if file_num < 10:          numstr = "_00" + str(file_num)
                        elif 10 <= file_num < 100: numstr = "_0" + str(file_num)
                        else:                      numstr = "_" + str(file_num)
                        subset_file = open(file_name + numstr + file_ext, 'wb')
                        print("Writing subset file: " + ospath.split(subset_file.name)[1])

                        if self.basic_header['FileSpec'] == '2.1':
                            subset_file.write(prior_file.read(32 + 4 * num_elecs))
                        else:
                            subset_file.write(prior_file.read(bytes_in_headers))
                            subset_file.write(header_binary)
                            timestamp_new = unpack('<I', timestamp_binary)[0] \
                                            + (packet_read_pts + pts_can_add) * self.basic_header['Period']
                            subset_file.write(np.array(timestamp_new).astype(np.uint32).tobytes())
                            subset_file.write(np.array(num_pts - pts_can_add).astype(np.uint32).tobytes())
                            prior_file.seek(num_pts_header_pos, 0)
                            prior_file.write(np.array(prior_file_pkt_pts).astype(np.uint32).tobytes())
                            num_pts_header_pos = bytes_in_headers + 5

                        prior_file.close()
                        packet_read_pts     += pts_can_add
                        start_idx           += pts_can_add
                        num_pts             -= pts_can_add
                        file_num            += 1
                        subset_file_pkt_pts  = 0
                        pausing              = False

                        pts_can_add = int((file_size - subset_file.tell()) // (num_elecs * DATA_BYTE_SIZE)) + 1
                        stop_idx    = start_idx + pts_can_add

                # Write remaining data
                if elec_id_indices:
                    subset_file.write(np.array(mm[start_idx:]).tobytes())
                else:
                    subset_file.write(mm[start_idx:])
                packet_read_pts     += num_pts
                subset_file_pkt_pts += num_pts
                del mm

            # Update num_pts header position
            if self.basic_header['FileSpec'] != '2.1':
                curr_hdr_num_pts_pos = num_pts_header_pos
                num_pts_header_pos  += 4 + subset_file_pkt_pts * num_elecs * DATA_BYTE_SIZE + 5

            datafile_pos += self.basic_header['ChannelCount'] * packet_pts * DATA_BYTE_SIZE
            self.datafile.seek(datafile_pos, 0)

            if file_time_s and not pausing and (self.datafile.tell() != ospath.getsize(self.datafile.name)):
                pausing = True
                print("\n*** Because of pausing in original datafile, this file may be slightly time shorter\n"
                      "       than others, and will contain multiple data packets offset in time\n")

            if self.basic_header['FileSpec'] != '2.1':
                subset_file_pos = subset_file.tell()
                subset_file.seek(curr_hdr_num_pts_pos, 0)
                subset_file.write(np.array(subset_file_pkt_pts).astype(np.uint32).tobytes())
                subset_file.seek(subset_file_pos, 0)

        subset_file.close()
        print("\n *** All subset files written to disk and closed ***")
        return "SUCCESS"

    def close(self):
        name = self.datafile.name
        self.datafile.close()
        print('\n' + name.split('/')[-1] + ' closed')