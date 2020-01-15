import re
import hashlib
import multiprocessing as mp

import pefile
import numpy as np
from sklearn.feature_extraction import FeatureHasher

# code from https://github.com/endgameinc/ember/blob/master/ember/features.py
def byte_entropy_histogram(bytez, window=1024, step=256):
    output = np.zeros((16, 16), dtype=np.int)
    a = np.frombuffer(bytez, dtype=np.uint8)
    def entropy_bin_counts(block):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        Hbin = int(H * 2)
        if Hbin == 16:
            Hbin = 15
        return Hbin, c

    if a.shape[0] <window:
        Hbin, c = entropy_bin_counts(a)
        output[Hbin, :] += c
    else:
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
        for block in blocks:
            Hbin, c = entropy_bin_counts(block)
            output[Hbin, :] += c
    output = output.flatten()
    sum = output.sum()
    return output / sum

def pe_import(pe):
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            import_info = []
            if isinstance(entry.dll, bytes):
                libname = entry.dll.decode().lower()
            else:
                libname = entry.dll.lower()
            import_info.append(libname)
        libraries_hashed = FeatureHasher(256, input_type="string").transform([import_info]).toarray()[0]
        sum = libraries_hashed.sum()
        return libraries_hashed / sum
    else:
        return np.zeros(256, dtype=np.float64)

def string_2d_histogram(bytez):
    output = np.zeros((16, 16), dtype=np.int)
    ascii_strings = re.compile(b'[\x20-\x7f]{6,}')
    ascii_strings = ascii_strings.findall(bytez)
    for string in ascii_strings:
        y = int(np.log(len(string)) / np.log(1.25))
        x = int(hashlib.md5(string).hexdigest(), 16) & 15
        output[y][x] += 1
    output = output.flatten()
    sum = output.sum()
    return output / sum

def pe_metadata(pe):
    ret = []
    def split_to_byte_list(value, size):
        ret = []
        for i in range(size):
            ret.append( (value & 255) / 255.0 )
            value >>= 8
        return ret

    if hasattr(pe, 'FILE_HEADER'):
        FILE_HEADER = [
            ('Machine', 2),  # The architecture type of the computer.
            ('NumberOfSections', 2),  # The number of sections.
            ('TimeDateStamp', 4), # The low 32 bits of the time stamp of the image.
            ('PointerToSymbolTable', 4),  # The offset of the symbol table, in bytes, or zero if no COFF symbol table exists.
            ('NumberOfSymbols', 4),  # The number of symbols in the symbol table.
            ('SizeOfOptionalHeader', 2),  # The size of the optional header, in bytes.
            ('Characteristics', 2)  # The characteristics of the image.
        ]
        for field, size in FILE_HEADER:
           ret.extend(split_to_byte_list(getattr(pe.FILE_HEADER, field, 0), size))
    else:
        ret.extend([0.0] * 20)

    if hasattr(pe, 'OPTIONAL_HEADER'):
        OPTIONAL_HEADER = [
            # The state of the image file.
            ('Magic', 2),
            # The major version number of the linker.
            ('MajorLinkerVersion', 1),
            # The minor version number of the linker.
            ('MinorLinkerVersion', 1),
            # The size of the code section, in bytes, or the sum of all such sections if there are multiple code sections.
            ('SizeOfCode', 4),
            # The size of the initialized data section, in bytes, or the sum of all such sections if there are multiple initialized data sections.
            ('SizeOfInitializedData', 4),
            # The size of the uninitialized data section, in bytes, or the sum of all such sections if there are multiple uninitialized data sections.
            ('SizeOfUninitializedData', 4),
            # A pointer to the entry point function, relative to the image base address.
            ('AddressOfEntryPoint', 4),
            # A pointer to the beginning of the code section, relative to the image base.
            ('BaseOfCode', 4),
            # A pointer to the beginning of the data section, relative to the image base.
            ('BaseOfData', 4),
            # The preferred address of the first byte of the image when it is loaded in memory.
            ('ImageBase', 4),  # ('ImageBase', 8), PE32+
            # The alignment of sections loaded in memory, in bytes.
            ('SectionAlignment', 4),
            # The alignment of the raw data of sections in the image file, in bytes.
            ('FileAlignment', 4),
            # The major version number of the required operating system.
            ('MajorOperatingSystemVersion', 2),
            # The minor version number of the required operating system.
            ('MinorOperatingSystemVersion', 2),
            # The major version number of the image.
            ('MajorImageVersion', 2),
            # The minor version number of the image.
            ('MinorImageVersion', 2),
            # The major version number of the subsystem.
            ('MajorSubsystemVersion', 2),
            # The minor version number of the subsystem.
            ('MinorSubsystemVersion', 2),
            # (Win32VersionValue) This member is reserved and must be 0.
            ('Reserved1', 4),  # Win32VersionValue
            # The size of the image, in bytes, including all headers.
            ('SizeOfImage', 4),
            # The combined size of the following items, rounded to a multiple of the value specified in the FileAlignment member.
            ('SizeOfHeaders', 4),
            # The image file checksum.
            ('CheckSum', 4),
            # The subsystem required to run this image.
            ('Subsystem', 2),
            # The DLL characteristics of the image.
            #('DllCharacteristics', 2),
            # The number of bytes to reserve for the stack.
            ('SizeOfStackReserve', 4),  # ('SizeOfStackReserve', 8) PE32+
            # The number of bytes to commit for the stack.
            ('SizeOfStackCommit', 4),  # ('SizeOfStackCommit', 8) PE32+
            # The number of bytes to commit for the local heap.
            ('SizeOfHeapReserve', 4),  # ('SizeOfHeapReserve', 8) PE32+
            # This member is obsolete.
            ('SizeOfHeapCommit', 4),  # ('SizeOfHeapCommit', 8) PE32+
            # The number of directory entries in the remainder of the optional header.
            ('LoaderFlags', 4),
            # A pointer to the first IMAGE_DATA_DIRECTORY structure in the data directory.
            ('NumberOfRvaAndSizes', 4),
        ]
        for field, size in OPTIONAL_HEADER:
           ret.extend(split_to_byte_list(getattr(pe.OPTIONAL_HEADER, field, 0), size))
        dll_characteristics = [0.0] * len(pefile.dll_characteristics)
        dll_characteristics_value = getattr(pe.OPTIONAL_HEADER, 'DllCharacteristics', 0)
        for i, (constant, value) in enumerate(pefile.dll_characteristics):
            if dll_characteristics_value & value == value:
                dll_characteristics[i] = 1.0
        ret.extend(dll_characteristics)
        if hasattr(pe.OPTIONAL_HEADER, 'DATA_DIRECTORY'):
            directory_entry_types_dict = {
                'IMAGE_DIRECTORY_ENTRY_EXPORT': 0,
                'IMAGE_DIRECTORY_ENTRY_IMPORT': 1,
                'IMAGE_DIRECTORY_ENTRY_RESOURCE': 2,
                'IMAGE_DIRECTORY_ENTRY_EXCEPTION': 3,
                'IMAGE_DIRECTORY_ENTRY_SECURITY': 4,
                'IMAGE_DIRECTORY_ENTRY_BASERELOC': 5,
                'IMAGE_DIRECTORY_ENTRY_DEBUG': 6,
                'IMAGE_DIRECTORY_ENTRY_COPYRIGHT': 7,
                'IMAGE_DIRECTORY_ENTRY_GLOBALPTR': 8,
                'IMAGE_DIRECTORY_ENTRY_TLS': 9,
                'IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG': 10,
                'IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT': 11,
                'IMAGE_DIRECTORY_ENTRY_IAT': 12,
                'IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT': 13,
                'IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR': 14,
                'IMAGE_DIRECTORY_ENTRY_RESERVED': 15
            }
            data_directory = [0.0] * 128
            for member in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
                idx = directory_entry_types_dict[member.name]
                idx = idx * 8
                data_directory[idx:idx+4] = split_to_byte_list(member.Size, 4)
                data_directory[idx+4:idx + 8] = split_to_byte_list(member.VirtualAddress, 4)
            ret.extend(data_directory)
        else:
            ret.extend([0.0] * 128)
    else:
        ret.extend([0.0] * 109)
        ret.extend([0.0] * 128)
    return np.array(ret, dtype=np.float64)

def preprocess(fn):
    ret = []
    with open(fn, 'rb') as f:
        data = f.read()
    a = byte_entropy_histogram(data)
    c = string_2d_histogram(data)
    try:
        pe = pefile.PE(data = data)
        b = pe_import(pe)
        d = pe_metadata(pe)
    except:
        b = np.zeros(256, dtype=np.float64)
        d = np.zeros(257, dtype=np.float64)
    return np.concatenate([a, b, c, d], axis=0)