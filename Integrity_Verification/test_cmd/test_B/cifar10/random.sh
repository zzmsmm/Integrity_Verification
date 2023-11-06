for ((i=0;i<=9;i++));do
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_pruning_2000_500_$i --attack_method pruning --fpmethod random --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_random-smoothing_2000_500_$i --attack_method random-smoothing --fpmethod random --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_backdoor_pattern1_2000_500_$i --attack_method backdoor --backdoor_type pattern1 --fpmethod random --epochs 20 --loss_lambda 0.05 --trg_set_size 500 --wm_batch_size 16 --lr 0.001 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_backdoor_pattern2_2000_500_$i --attack_method backdoor --backdoor_type pattern2 --fpmethod random --epochs 20 --loss_lambda 0.05 --trg_set_size 500 --wm_batch_size 16 --lr 0.001 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_backdoor_pattern3_2000_500_$i --attack_method backdoor --backdoor_type pattern3 --fpmethod random --epochs 20 --loss_lambda 0.05 --trg_set_size 500 --wm_batch_size 16 --lr 0.001 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
done

python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_0 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.001 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_1 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.002 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_2 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.003 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_3 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.004 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_4 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.005 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_5 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.006 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_6 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.007 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_7 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.008 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_8 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.009 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
python attack.py --dataset cifar10 --num_classes 10 --arch vgg16 --runname cifar10_vgg16_clean_random_fine-tune_2000_500_9 --attack_method fine-tune --fpmethod random --epochs 20 --lr 0.01 --sched CosineAnnealingLR --D1_number 2000 --D1_batch_size 16 --D2_batch_size 16 --D2_number 500 --loadmodel cifar10_vgg16_clean --cuda cuda:2 --test_type test_B
