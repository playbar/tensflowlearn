import numpy as np
import pickle
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def GetPhoto(pixel):
    assert len(pixel) == 3072
    r = pixel[0:1024]; r = np.reshape(r, [32, 32, 1])
    g = pixel[1024:2048]; g = np.reshape(g, [32, 32, 1])
    b = pixel[2048:3072]; b = np.reshape(b, [32, 32, 1])

    photo = np.concatenate([r, g, b], -1)

    return photo


def getTrainDataByKeyword(keyword, size=(32, 32), normalized=False, filelist=[]):
    '''
    :param keyword:'data', 'labels', 'batch_label', 'filenames', 表示需要返回的项目
    :param size:当keyword 是data时，表示需要返回的图片的尺寸
    :param normalized:当keyword是data时，表示是否需要归一化
    :param filelist:一个list， 表示需要使用的文件对象，仅1， 2， 3， 4， 5是有效的，其他数字无效
    :return:需要返回的数据对象。'data'表示需要返回像素数据。'labels'表示需要返回标签数据。'batch_label'表示需要返回文件标签数据。'filenames'表示需要返回文件的文件名信息。
    '''

    keyword = str.encode(keyword)

    assert keyword in [b'data', b'labels', b'batch_label', b'filenames']
    assert type(filelist) is list and len(filelist) != 0
    assert type(normalized) is bool
    assert type(size) is tuple

    files = []
    for i in filelist:
        if 1 <= i <= 5 and i not in files:
            files.append(i)

    if len(files) == 0:
        raise ValueError("No valid input files!")

    if keyword == b'data':
        data = []
        for i in files:
            data.append(unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d" % i)[b'data'])
        data = np.concatenate(data, 0)
        if normalized == False:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                array[i] = cv2.resize(GetPhoto(data[i]), size)
            return array
        else:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                array[i] = cv2.resize(GetPhoto(data[i]), size)/255
            return array
        pass
    if keyword == b'labels':
        labels = []
        for i in files:
            labels += unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d" % i)[b'labels']
        return labels
        pass
    elif keyword == b'batch_label':
        batch_label = []
        for i in files:
            batch_label.append(unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d" % i)[b'batch_label'])
        return batch_label
        pass
    elif keyword == b'filenames':
        filenames = []
        for i in files:
            filenames += unpickle("cifar-10-python/cifar-10-batches-py/data_batch_%d" % i)[b'filenames']
        return filenames
        pass
    pass


def getTestDataByKeyword(keyword, size=(32, 32), normalized=False):
    '''
    :param keyword:'data', 'labels', 'batch_label', 'filenames', 表示需要返回的项目
    :param size:当keyword 是data时，表示需要返回的图片的尺寸
    :param normalized:当keyword是data时，表示是否需要归一化
    :return:需要返回的数据对象。'data'表示需要返回像素数据。'labels'表示需要返回标签数据。'batch_label'表示需要返回文件标签数据。'filenames'表示需要返回文件的文件名信息。
    '''
    keyword = str.encode(keyword)

    assert keyword in [b'data', b'labels', b'batch_label', b'filenames']
    assert type(size) is tuple
    assert type(normalized) is bool

    batch_label = []
    filenames = []

    batch_label.append(unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'batch_label'])
    labels = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'labels']
    data = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'data']
    filenames += unpickle("cifar-10-python/cifar-10-batches-py/test_batch")[b'filenames']

    label = str.encode(keyword)
    if label == b'data':
        if normalized == False:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                array[i] = cv2.resize(GetPhoto(data[i]), size)
            return array
        else:
            array = np.ndarray([len(data), size[0], size[1], 3], dtype=np.float32)
            for i in range(len(data)):
                array[i] = cv2.resize(GetPhoto(data[i]), size) / 255
            return array
        pass
    elif label == b'labels':
        return labels
        pass
    elif label == b'batch_label':
        return batch_label
        pass
    elif label == b'filenames':
        return filenames
        pass
    else:
        raise NameError
    pass

