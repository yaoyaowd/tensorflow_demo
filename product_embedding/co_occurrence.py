import tensorflow as tf

pid_pairs = {}

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input filename")
flags.DEFINE_string("output", "", "Output filename")


def load_data(filename):
    num_lines_read = 0
    prev_uid = ''
    pids = []
    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 3:
                continue

            uid, action_time, pid = items[0], items[1], items[2]
            if uid != prev_uid and len(pids) > 0:
                for i in range(len(pids) - 1):
                    id1, id2 = pids[i], pids[i + 1]
                    if id1 == id2:
                        continue
                    if id1 > id2:
                        id1, id2 = id2, id1
                    key = id1 + '_' + id2
                    if key in pid_pairs:
                        pid_pairs[key] = pid_pairs[key] + 1
                    else:
                        pid_pairs[key] = 1
                prev_uid = uid
                pids = []
            pids.append(pid)

            if num_lines_read % 10000000 == 0:
                print "read %d lines" % num_lines_read
            if num_lines_read % 50000000 == 0:
                break


def main():
    load_data(FLAGS.input)

    with open(FLAGS.output, 'w') as out:
        items = sorted(pid_pairs, key=pid_pairs.get, reverse=True)
        for key in items:
            vs = key.split('_')
            out.write(vs[0] + '\t' + vs[1] + '\t' + str(pid_pairs[key]) + '\n')


if __name__ == '__main__':
    main()