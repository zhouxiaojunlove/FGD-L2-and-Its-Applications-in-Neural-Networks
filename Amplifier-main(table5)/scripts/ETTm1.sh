
model_name=Amplifier

python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model Amplifier --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --hidden_size 128 --SCI 0 --batch_size 32 --learning_rate 0.02 --des 'Exp' --itr 1  --flinear_c 1.0 --flinear_alpha  0.9

python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model Amplifier --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 192 --enc_in 7 --hidden_size 128 --SCI 0 --batch_size 32 --learning_rate 0.02 --des 'Exp' --itr 1  --flinear_c  0.2  --flinear_alpha  0.9

python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model Amplifier  --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 336 --enc_in 7 --hidden_size 128 --SCI 0 --batch_size 32 --learning_rate 0.02 --des 'Exp' --itr 1  --flinear_c  0.2  --flinear_alpha  0.9

python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model_id ETTm1 --model Amplifier --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 720 --enc_in 7 --hidden_size 128 --SCI 0 --batch_size 256 --learning_rate 0.005 --des 'Exp' --itr 1   --flinear_c  5.0  --flinear_alpha  0.9
