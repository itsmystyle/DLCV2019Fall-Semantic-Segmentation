# TODO: create shell script for running the testing code of the baseline model

wget 'https://www.dropbox.com/s/z6paaybrpsh7odt/baseline_model_best.pth.tar?dl=1' -O models/baseline_model/model_best.pth.tar
python3 test.py --resume models/baseline_model/model_best.pth.tar --test_batch 8 --data_dir $1 --output_dir $2 --baseline
