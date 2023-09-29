#!/bin/bash
#b1=('1.5' '0.37' '-0.38' '0.71' '0.5' '-0.33' '3' '-2.5' '-2' '2.5')
#b2=('-0.5' '0.63' '1.38' '0.29' '0.5' '1.33' '4' '-3' '5' '-4')
# b1=('-3' '4' '3' '-3' '-2.5' '3.5' '2.5' '-2.5' '-2' '3' '2' '-2' '-1.5' '2.5' '1.5' '-1.5' '-1' '2' '1' '-1' '-3' '3')
# b2=('4' '-3' '4' '-4' '3.5' '-2.5' '3.5' '-3.5' '3' '-2' '3' '-3' '2.5' '-1.5' '2.5' '-2.5' '2' '-1' '2' '-2' '-2' '2')
# b1=('-2' '2' '-2' '3' '-3' '3' '2'    '2'  '0' '0'  '4'  '4'   '0' '0' '0' '0' '4'  '4')
# b2=('5' '5' '-5' '-6' '-6' '6' '-1.5' '1.5' '0' '0' '-2' '2'   '0' '0' '0' '0' '-2' '2')
# b3=('-2' '2' '-2' '4' '-4' '4' '0.5' '0.5'  '0'  '0' '-1'  '1' '0' '0' '0' '0' '-1' '1')
# b1=('1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1')

b1=('-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3')
b2=('4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4')

a0=('1' '0.3')
tao=('1' '0.8' '0.6' '0.4' '0.2')
env='pt37_backup'
lr=0.1
bs=128
epoch=300 #300


model=('100ab2r20_a' '100ab2r32_a' '100ab2r44_a' '100ab2r56_a' '100ab2r110_a')
# model=('100ab2r20_b' '100ab2r32_b' '100ab2r44_b' '100ab2r56_b' '100ab2r110_b')
# model=('100ab2r68_a' '100ab2r80_a' '100ab2r92_a' '100ab2r104_a')
# model=('100ab2r68_b' '100ab2r80_b' '100ab2r92_b' '100ab2r104_b')
# model=('100ab2r68_c' '100ab2r80_c' '100ab2r92_c' '100ab2r104_c')
# model=('100ab2r20_c' '100ab2r32_c' '100ab2r44_c' '100ab2r56_c' '100ab2r110_c')
# model=('100ab2r20' '100ab2r32' '100ab2r44' '100ab2r56' '100ab2r110')
# model=('100ab2r68' '100ab2r80' '100ab2r92' '100ab2r104')

# model=('100ab3r20' '100ab3r32' '100ab3r44' '100ab3r56' '100ab3r110')
# model=('100ab1r20' '100ab1r32' '100ab1r44' '100ab1r56' '100ab1r110')
# model=('100ab1r98' '100ab1r104' '100ab1r116' '100ab1r122' '100ab1r128')
# model=('100ab1r134' '100ab1r140' '100ab1r146' '100ab1r152' '100ab1r158')
# model=('100ab1r164' '100ab1r170' '100ab1r176' '100ab1r182' '100ab1r188')
# model=('100ab1r218' '100ab1r248' '100ab1r278' '100ab1r308' '100ab1r338')
# model=('100ab1r368' '100ab1r398' '100ab1r428' '100ab1r458' '100ab1r488')
# model=('100ab3r68' '100ab3r80' '100ab3r92' '100ab3r104')
#model=( '100ab3r20' '100ab3r26' '100ab3r32' '100ab3r38' '100ab3r44' '100ab3r50a' '100ab3r56' '100ab3r62' '100ab3r68' '100ab3r74' '100ab3r80' '100ab3r86' '100ab3r92' '100ab3r98' '100ab3r104' '100ab3r110' '100ab3r116' '100ab3r122' '100ab3r128' '100ab3r134' '100ab3r140' '100ab3r146' '100ab3r152a' '100ab3r158')
dir_loc='10.29'
device='2'
train_dir='train_100ori_gradient.py' #train_ori1.py

for ((h=0;h<1;h++));do #tao 步长
    for ((l=0;l<1;l++));do #a0系数
        for ((k=1;k<2;k++));do #b1,b2,b3系数
            for ((i=2;i<3;i++));do #模型
                for ((j=0;j<1;j++));do #重复实验次数
                    #⬇1step⬇
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --a0 ${a0[l]} --b1 ${b1[k]} --arch ${model[i]} > ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇2step⬇
                    CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --arch ${model[i]} > ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇3step⬇
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --b3 ${b3[k]} --arch ${model[i]} > ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_b3_${b3[k]}_${model[i]}_${j}.txt 2>&1
                done;
            done;
        done;
    done;
done;





