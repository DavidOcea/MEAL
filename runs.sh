



20200728: 学生： purn_20200717_5T_t_20e   老师：mult_prun8_gpu,multnas5_gpu_2_18
nohup python -u main2muilt.py --gpu_id 7 --lr 0.01 --batch_size 96 --teachers [\'mul_mult_prun8_gpu\',\'mul_multnas5_gpu_2_18\'] --student mul_mult_prun8_gpu_prun \
--d_lr 0.01 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [0,1,2,3,4] \
--epochs 60 -r >./logs/distill_0728.log 2>&1 &
20200725: 学生： purn_20200717_5T_t_20e   老师：mult_prun8_gpu  模型不一样 学生：purn_20200717_5T_t_20e  老师：0713ckpt_epoch_47  采用cosine 存在0724里了，前面的递推
nohup python -u main2muilt.py --gpu_id 7 --lr 0.01 --batch_size 96 --teachers [\'mul_mult_prun8_gpu\'] --student mul_mult_prun8_gpu_prun \
--d_lr 0.01 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [0,1,2,3,4] \
--epochs 60 -r >./logs/distill_0725.log 2>&1 &
20200724: 学生： purn_20200717_5T_t_20e   老师：mult_prun8_gpu  模型不一样 学生：purn_20200717_5T_t_20e  老师：0713ckpt_epoch_47  
nohup python -u main2muilt.py --gpu_id 7 --lr 0.01 --batch_size 96 --teachers [\'mul_mult_prun8_gpu\'] --student mul_mult_prun8_gpu_prun \
--d_lr 0.01 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [0,1,2,3,4] \
--epochs 60 -r >./logs/distill_0723.log 2>&1 &
20200721: 学生： purn_20200717_5T_t_20e   老师：mult_prun8_gpu  模型不一样 学生：purn_20200717_5T_t_20e  老师：0713ckpt_epoch_47  停了，发现平均结果和学生的差不多，但是细分到每一类却没什么优势
nohup python -u main2muilt.py --gpu_id 0 --lr 0.1 --batch_size 96 --teachers [\'mul_mult_prun8_gpu\'] --student mul_mult_prun8_gpu_prun \
--d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [0,1,2,3,4] \
--epochs 100 -r >./logs/distill_0721.log 2>&1 &







实验
python main.py --gpu_id 0 --lr 0.1 --batch_size 256 --teachers [\'vgg19_BN\',\'dpn92\',\'resnet18\'] --student densenet_cifar --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [-1] 

python main.py --gpu_id 0 --lr 0.1 --batch_size 64 --teachers [\'resnet18\'] --student densenet_cifar --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [-1] --epochs 1 -r True

python main2muilt.py --gpu_id 0 --lr 0.1 --batch_size 8 --teachers [\'mul_mult_prun8_gpu\'] --student mul_mult_prun8_gpu --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --gamma [1,1,1,1,1] --eta [1,1,1,1,1] --name 5_ensemble_for_densenet --out_layer [-1,0] --epochs 1

