#! /usr/bin/awk -f
# Author(s): Ming Tong <tongming@qiyi.com>
# split data to train validation test
# usage: input training file path

BEGIN {
    FS=","
}
{
    if ($3>14101000) {
        print $0 > "test.txt"
    } else if ($3>14100900) {
        print $0 > "validation.txt"
    } else {
        print $0 > "train.txt"
    }
}

