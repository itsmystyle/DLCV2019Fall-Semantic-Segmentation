# TODO: create shell script for running the testing code of your improved model

wget 'https://www.dropbox.com/s/wd7xcrumcnzv2d6/improved_model_best.pth.tar?dl=1' -O models/improved_model/model_best.pth.tar
python3 test.py --resume models/improved_model/model_best.pth.tar --test_batch 8 --data_dir $1 --output_dir $2
