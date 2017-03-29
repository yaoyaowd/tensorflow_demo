import sys


pid_pairs = {}


def process(actions, pids):
    vpids = []
    for i in range(1, len(pids)):
        p1 = min(pids[i - 1], pids[i])
        p2 = max(pids[i - 1], pids[i])
        if p1 != p2:
            if p1 not in pid_pairs:
                pid_pairs[p1] = dict()
            if p2 not in pid_pairs[p1]:
                pid_pairs[p1][p2] = 1
            else:
                pid_pairs[p1][p2] = 1 + pid_pairs[p1][p2]


def load_data(filename):
    num_lines_read = 0
    prev_uid = ''
    actions = []
    pids = []

    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 3:
                continue

            uid = items[0]
            pid = items[2]
            if uid != prev_uid:
                process(actions, pids)
                prev_uid = uid
                pids = []
            pids.append(pid)

            if num_lines_read % 10000000 == 0:
                print "read %d lines" % num_lines_read
                print "saw %d valid products" % len(pid_pairs)
                break


def main():
    global product_dict
    global paragraphs

    load_data(sys.argv[1])
    with open('/home/dwang/co_occurrence.tsv', 'w') as out:
        for i in pid_pairs:
            items = sorted(pid_pairs[i], key=pid_pairs[i].get, reverse=True)
            vitems = []
            for j in items:
                if pid_pairs[i][j] > 1:
                    vitems.append(j)
            if len(vitems) > 0:
                out.write(i + "," + ",".join(vitems) + "\n")


if __name__ == '__main__':
    main()