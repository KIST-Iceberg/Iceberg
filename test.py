# Copy Right Kairos03 2017. All Right Reserved.
"""
test
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from data import data_input
from data import process
import train

# hyper parameter
# BATCH_SIZE = train.BATCH_SIZE
BATCH_SIZE = train.BATCH_SIZE
RANDOM_SEED = train.RANDOM_SEED


def test(model_path, is_test=False):
    with tf.Graph().as_default() as graph:
        # data set load
        x, y, angle = process.load_from_pickle(is_test=is_test)
        inputs = data_input.get_dataset(BATCH_SIZE, x, y, angle, is_shuffle=False, is_valid=(not is_test))
        total_batch = inputs.total_batch if is_test else inputs.valid_total_batch

        print('Test Start')

        init_op = tf.global_variables_initializer()
        saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta', clear_devices=True)

        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            graph = tf.get_default_graph()

            X = graph.get_tensor_by_name('input/X:0')
            Y = graph.get_tensor_by_name('input/Y:0')
            A = graph.get_tensor_by_name('input/A:0')
            keep_prob = graph.get_tensor_by_name('input/keep_prob:0')

            # load Variables and ops
            probability = sess.graph.get_tensor_by_name('matrices/proba:0')
            accuracy = sess.graph.get_tensor_by_name('matrices/accuracy:0')
            xent = sess.graph.get_tensor_by_name('matrices/xent:0')

            total_predict = None
            total_acc = total_loss = 0

            for batch in range(total_batch):
                # xs, _ = inputs.next_batch()

                if not is_test:
                    xs, ys, ans = inputs.next_batch(RANDOM_SEED, valid_set=True)
                    acc, loss, predict = sess.run([accuracy, xent, probability],

                                         feed_dict={X: xs,
                                                    Y: ys,
                                                    A: ans,
                                                    keep_prob: 1})

                    total_acc += acc
                    total_loss += loss
                    if total_predict is None:
                        total_predict = predict
                    else:
                        total_predict = np.concatenate((total_predict, predict))
                else:
                    xs, ys, ans = inputs.next_batch(RANDOM_SEED, valid_set=False)
                    predict = sess.run(probability,
                                        feed_dict={X: xs,
                                                    Y: ys,
                                                    A: ans,
                                                    keep_prob: 1})
                    if total_predict is None:
                        total_predict = predict
                    else:
                        total_predict = np.concatenate((total_predict, predict))

            if not is_test:
                print('Accuracy: {:.6f}, Loss: {:.6f}'.format(total_acc/total_batch, total_loss/total_batch))
                is_iceberg = total_predict[:, 1]
                print('Log Loss', log_loss(inputs.valid_label[:,1], is_iceberg))
            else:
                idx = pd.read_json('data/origin/test.json')
                idx = idx['id']
                is_iceberg = total_predict[:, 1]
                data = pd.DataFrame({'id': idx, 'is_iceberg': is_iceberg}, index=None)
                data.to_csv('test.csv', index=False, float_format='%.6f')
                # predict = [1 if p >= 0.98 else p for p in predict]
                    # predict = [0 if p <= 0.02 else p for p in predict]
                    # round_logloss += log_loss(ys, predict)

        print('Test Finish')


if __name__ == '__main__':
    test('./log/BEST_0.0615 2018_lr0.001_ep200/train/', is_test=True)
