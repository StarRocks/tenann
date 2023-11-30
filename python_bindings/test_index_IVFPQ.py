import unittest
import tenann_py
import math
import os
import numpy as np

class TestIndexHNSW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_path = '/tmp/tenann_index_ivfpq_py'
        # init data
        dim = 4                       # dimension
        nb = 10000                      # database size
        nq = 10                       # nb of queries
        M = 4
        nbits = 4
        nlist = math.floor(math.sqrt(nb))
        nprobe = math.floor(math.sqrt(nb))
        cls.meta_json = '''
            {
                "meta_version": 0,
                "family": 0,
                "type": 2,
                "common": {
                    "dim": %d,
                    "is_vector_normed": false,
                    "metric_type": 0
                },
                "index": {
                    "nlist" : %d,
                    "nbits": %d,
                    "M": %d
                },
                "search": {
                    "nprobe" : %d,
                    "max_codes" : 0,
                    "scan_table_threshold" : 0,
                    "polysemous_ht" : 0
                },
                "extra": {
                    "comments": "Testing IndexIVFPQ.",
                    "use_custom_row_id": false
                }
            }
        ''' % (dim, nlist, nbits, M, nprobe)
        print("meta_json: ", cls.meta_json)

        np.random.seed(1234)          # make reproducible
        cls.xb = np.random.random((nb, dim)).astype('float32')
        cls.xb[:, 0] += np.arange(nb) / 1000.
        cls.xq = np.random.random((nq, dim)).astype('float32')
        cls.xq[:, 0] += np.arange(nq) / 1000.
        cls.ground_truth = cls.calculate_ground_truth(cls.xb, cls.xq, np, 10)
        cls.recall_rates = []

    @classmethod
    def setUp(cls):
        cls.ann = tenann_py.TenANN()
        # 创建 builder
        try:
            cls.ann = cls.ann.create_builder(cls.meta_json)
            print("Builder created successfully.")
        except Exception as e:
            cls.fail("An error occurred while creating the builder:", e)
        # create searcher
        try:
            cls.ann = cls.ann.create_searcher(cls.meta_json)
            print("AnnSearcher created successfully.")
        except Exception as e:
            cls.fail("An error occurred while creating the searcher:", e)

    def test_1_create_function(self):
        # 新创建临时 ANN
        test_ann = tenann_py.TenANN()
        # builder 为空
        with self.assertRaises(RuntimeError) as e:
            test_ann.open()
        print(e.exception)
        # searcher 为空
        with self.assertRaises(RuntimeError) as e:
            test_ann.read_index(self.index_path)
        print(e.exception)

    def test_2_open_close(self):
        # Index builder has not been opened.
        with self.assertRaises(RuntimeError):
            self.ann.close()
        # Index builder opened successfully.
        try:
            self.ann.open()
        except Exception as e:
            self.fail("Unexpected exception occurred while opening:", e)
        # Index builder has already been opened
        with self.assertRaises(RuntimeError):
            self.ann.open()
        # Index builder closed successfully.
        try:
            self.ann.close()
        except Exception as e:
            self.fail("Unexpected exception occurred while closing:", e)

    def test_3_add(self):
        # Add before index open
        with self.assertRaises(RuntimeError):
            self.ann.add(self.xb)
        # Add after index close
        with self.assertRaises(RuntimeError):
            self.ann.open()
            self.ann.close()
            self.ann.add(self.xb)
        # Add successfully
        try:
            self.ann.open()
            self.ann.add(self.xb)
            self.ann.close()
        except Exception as e:
            self.fail("Unexpected exception occurred while opening:", e)

    def test_4_flush(self):
        # Flush before index open
        with self.assertRaises(RuntimeError):
            self.ann.flush()
        # Flush after index close
        with self.assertRaises(RuntimeError):
            self.ann.open()
            self.ann.close()
            self.ann.flush()
        # Flush successfully
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            self.ann.open(self.index_path)
            self.ann.add(self.xb)
            self.ann.flush()
            self.ann.close()
            if not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0:
                self.fail("Flush failed, index_path: ", self.index_path)
        except Exception as e:
            self.fail("Unexpected exception occurred while opening:", e)

    def test_5_read_index(self):
        # ann_search before read_index
        with self.assertRaises(RuntimeError):
            self.ann.ann_search(self.xq[0], 10)

        # read_index
        try:
            if not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0:
                self.fail("Index_file is not exist, read_index failed, index_path: ", self.index_path)
            self.ann.read_index(self.index_path)
            print("Read Index successfully.")
            print("ann_search successed after read_index, result:", self.ann.ann_search(self.xq[0], 10))
        except Exception as e:
            self.fail("An error occurred while reading index:", e)

    def test_6_ann_search(self):
        # Testing recall rate of ann search
        try:
            if not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0:
                self.fail("Index_file is not exist, read_index failed, index_path: ", self.index_path)
            self.ann.read_index(self.index_path)

            for i, query_vector in enumerate(self.xq):
                result_ids = self.ann.ann_search(query_vector, 10)
                print("Query {}: {}".format(i, result_ids))
                # Step 2: Calculate the recall.
                # Compare the search results with the ground truth to calculate the recall rate.
                true_positive_count = len(set(result_ids).intersection(set(self.ground_truth[i])))
                self.recall_rate = true_positive_count / len(self.ground_truth[i])  # Assuming ground_truth[i] has 'k' neighbors.
                self.recall_rates.append(self.recall_rate)

                print(f"Query {i}: Recall rate is {self.recall_rate:.2f}")
            average_recall_rate = sum(self.recall_rates) / len(self.recall_rates)
            print(f"Average recall rate: {average_recall_rate:.2f}")    
        except Exception as e:
            self.fail("An error occurred while ann_search:", e)

    # Perform a brute-force search to find the actual nearest neighbors for each query.
    # Note: This can be computationally expensive.
    def calculate_ground_truth(xb, xq, np, k):
        # Calculate L2 distance between each pair of points (query and database) and get the closest ones.
        # This assumes that the distance metric is L2. If not, you should replace this with the appropriate distance calculation.
        squared_diff = np.sum(np.square(xb[:, np.newaxis] - xq[np.newaxis, :]), axis=-1)
        nearest_neighbors = np.argsort(squared_diff, axis=0)[:k]
        return nearest_neighbors.T  # Transpose to have it in a more intuitive shape

if __name__ == '__main__':
    unittest.main()