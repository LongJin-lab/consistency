#!/bin/bash


# b1=('1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1')
a0=('1' '0.3')
tao=('1' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1')
env='pt37_backup'
lr=0.1
bs=128
epoch=160 #160

# model=('r20' 'r32' 'r44' 'r56' 'r110')
# model=('zr20' 'zr32' 'zr44' 'zr56' 'zr110')





dir_loc='noise-ab2r56-h-0.1-c10'
device='7'
train_dir='train_ori_noise.py' #train_ori1.py train_ori_gradient.py train_ori_gradient_3step.py

for ((h=9;h<10;h++));do #h
    for ((l=0;l<1;l++));do #a0系数
        for ((k=4;k<8;k++));do #b1,b2,b3系数
            for ((i=3;i<4;i++));do #模型
                for ((j=0;j<3;j++));do #重复实验次数
                    #⬇1step⬇
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[device]} --a0 ${a0[l]} --b1 ${b1[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[device]}_a0_${a0[l]}_b1_${b1[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇2step⬇
                    CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[h]} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇3step⬇ !!!!!!!!换环境导致的r110性能变好！！！！！！！！！
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[h]} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --b3 ${b3[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_b3_${b3[k]}_${model[i]}_${j}.txt 2>&1
                done;
            done;
        done;
    done;
done;





