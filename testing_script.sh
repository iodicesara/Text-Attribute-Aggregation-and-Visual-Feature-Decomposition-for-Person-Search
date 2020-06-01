
: ' 
Market-1501 TAVD w/o TA
'
python Main.py --dataset market1501 --is_deterministic --seed 8 --save-dir data/glove_aug_5_l1_01_loc --log_to_file --exp_dir data/glove_aug_5_l1_01_loc --is_warmup --lr 0.000035 --print-freq 100 --arch resnetAttW2VText --attribute_path_bin data/attributes/market.npy --test_attribute_path data/attributes/glove-market.npy --attribute_path data/attributes/glove-market-cam-5.npy --self_attribute_path data/attributes/glove-market-cam-5.npy --random_label 0 --max-epoch 120 --coeff_loss_attributes_reid 0.1 --attr_loss_type L1  --attraug_reid --num_classes_attributes 5 --is_frame --load_weights data/glove_aug_5_l1_01_loc/checkpoint_ep120.pth.tar --evaluate

: ' 
Market-1501 TAVD w/o aug
'
python Main.py --dataset market1501 --is_deterministic --seed 8 --save-dir data/glove_aug_no_aug --log_to_file --exp_dir data/glove_aug_no_aug --is_warmup --lr 0.000035 --print-freq 100 --arch resnetAttW2VText --attribute_path_bin data/attributes/market.npy --test_attribute_path data/attributes/glove-market.npy --attribute_path data/attributes/glove-market.npy --self_attribute_path data/attributes/glove-market.npy --random_label 0 --max-epoch 120 --attr_loss_type L1 --global_learning --num_classes_attributes 0 --coeff_loss_attributes_reid 0.1 --attraug_reid --load_weights data/glove_aug_no_aug/checkpoint_ep120.pth.tar --evaluate 


: ' 
Market-1501 TAVD
'
python Main.py --dataset market1501 --is_deterministic --seed 8 --save-dir data/glove_aug_5_l1_01 --log_to_file --exp_dir data/glove_aug_5_l1_01 --is_warmup --lr 0.000035 --print-freq 100 --arch resnetAttW2VText --attribute_path_bin data/attributes/market.npy --test_attribute_path data/attributes/glove-market.npy --attribute_path data/attributes/glove-market-cam-5.npy --self_attribute_path data/attributes/glove-market-cam-5.npy --random_label 0 --max-epoch 120 --coeff_loss_attributes_reid 0.1 --attr_loss_type L1 --global_learning --attraug_reid --num_classes_attributes 5 --is_frame --load_weights data/glove_aug_5_l1_01/checkpoint_ep120.pth.tar --evaluate

: ' 
Market-1501 TAVD w/o VD
'
python Main.py --dataset market1501 --is_deterministic --seed 8 --save-dir data/glove_aug_5_glob --log_to_file --exp_dir data/glove_aug_5_glob --is_warmup --lr 0.000035 --print-freq 100 --arch resnetAttW2VText --attribute_path_bin data/attributes/market.npy --test_attribute_path data/attributes/glove-market.npy --attribute_path data/attributes/glove-market-cam-5.npy --self_attribute_path data/attributes/glove-market-cam-5.npy --random_label 0 --max-epoch 120 --attr_loss_type L1 --global_learning --num_classes_attributes 5 --is_frame --load_weights data/glove_aug_5_glob/checkpoint_ep120.pth.tar --evaluate

