# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf
import numpy as np
import pandas as pd
import data_input
from sklearn.metrics import log_loss

# hyper parameter
batch_size = 30


def test(model_path, is_valid=False):
    # data set load
    if is_valid:
        x, y, idx = data_input.load_data()
        inputs = data_input.get_dataset(batch_size, x, y, is_shuffle=False, is_valid=True)
    else:
        x, idx = data_input.load_test()
        inputs = data_input.get_dataset(batch_size, x, np.zeros(x.shape[0]), is_shuffle=False, is_valid=False)

    print('Test Start')

    init_op = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(model_path + 'model.ckpt.meta')

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name('input/X:0')
        Y = graph.get_tensor_by_name('input/Y:0')
        keep_prob = graph.get_tensor_by_name('input/keep_prob:0')

        # load Variables and ops
        probability = sess.graph.get_tensor_by_name('matrices/proba/proba:0')
        accuracy = sess.graph.get_tensor_by_name('matrices/accuracy/accuracy:0')
        xent = sess.graph.get_tensor_by_name('matrices/xent/xent:0')

        total_predict = None
        total_acc = total_loss = 0

        for batch in range(inputs.total_batch):
            # xs, _ = inputs.next_batch()

            xs, ys = inputs.next_batch()

            if is_valid:
                acc, loss = sess.run([accuracy, xent],
                                     feed_dict={X: xs,
                                                Y: ys,
                                                keep_prob: 1})

                total_acc += acc
                total_loss += loss

            else:
                predict = sess.run(probability,
                                   feed_dict={X: xs,
                                              Y: ys,
                                              keep_prob: 1})
                if total_predict is None:
                    total_predict = predict
                else:
                    total_predict = np.concatenate((total_predict, predict))

        if is_valid:
            print('Accuracy: {}, Loss: {}'.format(total_acc/inputs.valid_size, total_loss/inputs.valid_size))
        else:
            is_iceberg = total_predict[:, 1]
            data = pd.DataFrame({'id': idx, 'is_iceberg': is_iceberg}, index=None)
            data.to_csv('test.csv', index=False, float_format='%.6f')
            print('Log Loss', log_loss(inputs.label, is_iceberg))

    print('Test Finish')


if __name__ == '__main__':
    test('./log/train/Mon Dec 18 11:16:08 2017_lr0.0001_ep30_b30/')
