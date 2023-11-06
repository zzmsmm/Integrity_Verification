for ((i=0;i<=9;i++));do
python AdvFT_attack.py --dataset caltech-101 --num_classes 101 --arch resnet18 --lr 0.001 --runname caltech101_resnet18_clean_adversarial_center_2000_500_$i --initial_attack_method pruning --pruning_rate 0.2 --fpmethod adversarial_center --epochs_advft_attack 30 --loss_lambda 0.1 --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel caltech101_resnet18_clean --cuda cuda:3 --test_type test_A
python AdvFT_attack.py --dataset caltech-101 --num_classes 101 --arch resnet18 --lr 0.001 --runname caltech101_resnet18_clean_adversarial_center_500_500_$i --initial_attack_method pruning --pruning_rate 0.2 --fpmethod adversarial_center --epochs_advft_attack 30 --loss_lambda 0.1 --D1_number 500 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel caltech101_resnet18_clean --cuda cuda:3 --test_type test_A
python AdvFT_attack.py --dataset caltech-101 --num_classes 101 --arch resnet18 --lr 0.001 --runname caltech101_resnet18_clean_adversarial_center_2000_200_$i --initial_attack_method pruning --pruning_rate 0.2 --fpmethod adversarial_center --epochs_advft_attack 30 --loss_lambda 0.1 --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 200 --loadmodel caltech101_resnet18_clean --cuda cuda:3 --test_type test_A
python AdvFT_attack.py --dataset caltech-101 --num_classes 101 --arch resnet18 --lr 0.001 --runname caltech101_resnet18_clean_adversarial_center_500_200_$i --initial_attack_method pruning --pruning_rate 0.2 --fpmethod adversarial_center --epochs_advft_attack 30 --loss_lambda 0.1 --D1_number 500 --D1_batch_size 16 --D2_batch_size 16 --D2_number 200 --loadmodel caltech101_resnet18_clean --cuda cuda:3 --test_type test_A
done
