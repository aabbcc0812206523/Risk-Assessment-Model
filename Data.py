import numpy as np
import heapq


class Data():
    def __init__(self, filename, batch_size=50, divide=True, smote = False, **kw):
        self.batch_size = 50
        self.data = np.load(filename)
        self.shape = np.shape(self.data)
        self.batch_size = batch_size
        self.batch_start_num = 0
        self.batch_end_num = batch_size
        self.smote_batch_start_num = 0
        self.smote_batch_end_num = batch_size
        self.valid_batch_start_num = 0
        self.valid_batch_end_num = batch_size
        self.train_smote_data = []
        self.train_smote_data_num = 372*32
        if divide:
            self.train_data = self.data[0:int(self.shape[0] * 0.8)]
            self.valid_data = self.data[int(self.shape[0] * 0.8):]
            self.train_data_num = np.shape(self.train_data)[0]
            self.valid_data_num = np.shape(self.valid_data)[0]

        if smote:
            if not kw:
                self.smote(16,32)
            else:
                self.smote(kw['k'], kw['n'])
            self.train_smote_data_num = np.shape(self.train_smote_data)[0]

    def get_train_data(self):
        '''
        0 1 2 3 4 5
        3-8 5
        3-6 2
        0-3 3
        3 = 5 - (8-6)
        '''
        if self.batch_end_num > self.train_data_num:
            result = np.concatenate((self.train_data[self.batch_start_num:],
                                     self.train_data[0:self.batch_end_num - self.train_data_num]))

            self.batch_start_num = self.batch_end_num - self.train_data_num
        else:
            result = self.train_data[self.batch_start_num:self.batch_end_num]
            self.batch_start_num += self.batch_size

        self.batch_end_num = self.batch_start_num + self.batch_size
        return result

    def get_train_smote_data(self):
        if self.smote_batch_end_num >= self.train_smote_data_num:
            result = np.concatenate((self.train_smote_data[self.smote_batch_start_num:],
                                     self.train_smote_data[0:self.smote_batch_end_num - self.train_smote_data_num]))

            self.smote_batch_start_num = self.smote_batch_end_num - self.train_smote_data_num
        else:
            result = self.train_smote_data[self.smote_batch_start_num:self.smote_batch_end_num]
            self.smote_batch_start_num += self.batch_size

        self.smote_batch_end_num = self.smote_batch_start_num + self.batch_size
        return np.array(result)

    def get_valid_data(self):
        if self.valid_batch_end_num > self.valid_data_num:
            result = np.concatenate((self.valid_data[self.valid_batch_start_num:],
                                     self.valid_data[0:self.valid_batch_end_num - self.valid_data_num]))

            self.valid_batch_start_num = self.valid_batch_end_num - self.valid_data_num
        else:
            result = self.valid_data[self.valid_batch_start_num:self.valid_batch_end_num]
            self.valid_batch_start_num += self.batch_size

        self.valid_batch_end_num = self.valid_batch_start_num + self.batch_size
        return result

    def get_all_valid_data(self):
        return self.valid_data

    def get_all_train_data(self):
        return self.train_data

    def get_valid_data_itr(self, max):
        n = 0
        while n < max:
            yield self.valid_data[n]
            n = n + 1

    def get_train_data_itr(self, max):
        n = 0
        while n < max:
            yield self.train_data[n]
            n = n + 1

    def _top_k(self, alist, k):
        max_heap = []
        length = len(alist)
        if not alist or k <= 0 or k > length:
            return
        k = k - 1
        for ele in alist:
            ele = -ele
            if len(max_heap) <= k:
                heapq.heappush(max_heap, ele)
            else:
                heapq.heappushpop(max_heap, ele)

        return map(lambda x: -x, max_heap)

    def _big_heap(self, k, index):
        '''通过大顶堆排序，k大顶堆尺寸，index是待扩充数据的下标'''
        big_heap = [99999999.0] * (k + 2)  # 存储距离值
        big_heap_data = [[]] * (k + 2)  # 存储向量数据
        for i in range(k):
            dis_arr = np.array(
                [np.sum(np.square(self.train_data[i][:-2] - self.train_data[index][-2])) for i in range(k)])
            # print(dis_arr)
            index_sort = np.argsort(-dis_arr)
            for i in range(k):
                big_heap[i + 1] = np.sum(np.square(self.train_data[index_sort[i]][:-2] - self.train_data[index][-2]))
                big_heap_data[i + 1] = self.train_data[index_sort[i]]

        for i in range(k, len(self.train_data)-1):
            dis = np.sum(np.square(self.train_data[i][:-2] - self.train_data[index][-2]))
            if dis > big_heap[1] or dis == 0:
                pass
            else:
                pos = 1
                big_heap[1] = dis
                big_heap_data[1] = self.train_data[i]
                while 2 * pos <= k:
                    if dis > big_heap[2 * pos] and dis > big_heap[2 * pos + 1]:
                        break
                    elif dis < big_heap[2 * pos] and big_heap[2 * pos] > big_heap[2 * pos + 1]:
                        big_heap[pos] = big_heap[2 * pos]
                        big_heap[2 * pos] = dis
                        temp = big_heap_data[2 * pos]
                        big_heap_data[2 * pos] = big_heap_data[pos]
                        big_heap_data[pos] = temp
                        pos = 2 * pos
                    else:
                        big_heap[pos] = big_heap[2 * pos + 1]
                        big_heap[2 * pos + 1] = dis
                        temp = big_heap_data[2 * pos + 1]
                        big_heap_data[2 * pos + 1] = big_heap_data[pos]
                        big_heap_data[pos] = temp
                        pos = 2 * pos + 1
            return big_heap, big_heap_data

    def smote(self, k, n):
        '''n是采样倍率 k是临近个数'''
        for i in range(len(self.train_data)):
            big_heap, big_heap_data = self._big_heap(k, i)
            for i in range(n):
                new_data = self.train_data[i] + np.random.rand() * (
                        big_heap_data[int(np.random.rand() * 16) + 1] - self.train_data[i])
                self.train_smote_data.append(list(new_data))
            # print(big_heap)


if __name__ == '__main__':
    # np.save('test.npy', np.array([x for x in range(400)]))
    # d = Data('test.npy')
    d = Data('data_2.npy')
    d.smote(16, 32)
    print(len(d.train_smote_data))
    while 1:
        try:
            print(len(d.get_train_smote_data()))

        except:
            pass