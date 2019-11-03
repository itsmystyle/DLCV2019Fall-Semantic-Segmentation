# TODO: create shell script for running the testing code of your improved model

# wget -O models/improved_model/model_best.pth.tar
python3 test.py --resume models/inet_v2/model_best.pth.tar --test_batch 24 --data_dir $1 --output_dir $2
