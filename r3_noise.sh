#!/bin/bash
#b1=('0.33' '1.92' '0.7' '-1' '0.5' '2' '-1' '0.5' '-2' '9')
#b2=('0.33' '-1.34' '0.7' '1' '0.25' '2' '-2' '7' '-5' '9')
#b3=('0.33' '0.42' '-0.4' '1' '0.25' '2' '-3' '-1' '3' '6')

# b1=('-3' '4' '3' '-3' '-2.5' '3.5' '2.5' '-2.5' '-2' '3' '2' '-2' '-1.5' '2.5' '1.5' '-1.5' '-1' '2' '1' '-1' '-3' '3')
# b2=('4' '-3' '4' '-4' '3.5' '-2.5' '3.5' '-3.5' '3' '-2' '3' '-3' '2.5' '-1.5' '2.5' '-2.5' '2' '-1' '2' '-2' '-2' '2')

b1=('-3' '-3' '3'  '-2.5' '2.5' '-2.5'  '3' '-3' '3'  '1.5' '1.5' '-1.5')
b2=('4'  '-4' '4'  '3.5'  '3.5' '-3.5' '-2' '-2' '2' '-0.5' '0.5' '-0.5')

# b1=( '-2.5' '2.5' '-2.5'  '3' '-3' '3'  '1.5' '1.5' '-1.5')
# b2=( '3.5'  '3.5' '-3.5' '-2' '-2' '2' '-0.5' '0.5' '-0.5')

# b1=('-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3' '-3' '3')
# b2=('4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4' '4')
# b1=('0' '1' '2' '3')
# b2=('0' '1' '2' '3')
# b3=('0' '1' '2' '3')
# b1=('4' '4')
# b2=('-2' '2')
# b3=('-1' '1')
# b1=('-3' '3')
# b2=('4' '4')
# b1=('-2' '2' '-2' '3' '-3' '3' '2'    '2'  '0' '0'  '4'  '4' )
# b2=('5' '5' '-5' '-6' '-6' '6' '-1.5' '1.5' '0' '0' '-2' '2')
# b3=('-2' '2' '-2' '4' '-4' '4' '0.5' '0.5'  '0'  '0' '-1'  '1')

# b1=('-2' '2'  '3' '-3'  '2'    '2'    '4'  '-4' )
# b2=('5' '5'  '-6' '-6'  '-1.5' '1.5'  '-2' '-2')
# b3=('-2' '2'  '4' '-4'  '0.5' '0.5'  '-1'  '-1')

# b1=('1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1' '1' '-1')
a0=('1' '0.3')
tao=('1' '0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3' '0.2' '0.1')
env='pt37_backup'
lr=0.1
bs=128
epoch=160 #160

# model=('r20' 'r32' 'r44' 'r56' 'r110')
# model=('abl2r20' 'abl2r32' 'abl2r44' 'abl2r56' 'abl2r110')
# model=('abll2r20' 'abll2r32' 'abll2r44' 'abll2r56' 'abll2r110')
# model=('ab2r20' 'ab2r32' 'ab2r44' 'ab2r56' 'ab2r110')
model=('ab2pr20' 'ab2pr32' 'ab2pr44' 'ab2pr56' 'ab2pr110')
# model=('ab2r68' 'ab2r80' 'ab2r92' 'ab2r104')
# model=('ab2r20_a' 'ab2r32_a' 'ab2r44_a' 'ab2r56_a' 'ab2r110_a')
# model=('ab2r20_f' 'ab2r110_f')
# model=('ab2r20_f1' 'ab2r110_f1')
# model=('ab3r20_f' 'ab3r110_f')
# model=('ab3r20_f1' 'ab3r110_f1')
# model=('ab2r68_a' 'ab2r80_a' 'ab2r92_a' 'ab2r104_a')
# model=('ab2r68_b' 'ab2r80_b' 'ab2r92_b' 'ab2r104_b')
# model=('ab2r68_c' 'ab2r80_c' 'ab2r92_c' 'ab2r104_c')
# model=('ab2r20_b' 'ab2r32_b' 'ab2r44_b' 'ab2r56_b' 'ab2r110_b')
# model=('ab2r20_c' 'ab2r32_c' 'ab2r44_c' 'ab2r56_c' 'ab2r110_c')
# model=('ab3r20' 'ab3r32' 'ab3r44' 'ab3r56' 'ab3r110')
# model=('ab3r68' 'ab3r80' 'ab3r92' 'ab3r104')
# model=('ab1r20' 'ab1r32' 'ab1r44' 'ab1r56' 'ab1r110')
# model=('ab1r98' 'ab1r104' 'ab1r116' 'ab1r122' 'ab1r128')
# model=('ab1r134' 'ab1r140' 'ab1r146' 'ab1r152' 'ab1r158')
# model=('ab1r164' 'ab1r170' 'ab1r176' 'ab1r182' 'ab1r188')
# model=('ab1r218' 'ab1r248' 'ab1r278' 'ab1r308' 'ab1r338')
# model=('ab1r368' 'ab1r398' 'ab1r428' 'ab1r458' 'ab1r488')
# model=('ab1r236' 'ab1r242' 'ab1r254' 'ab1r260' 'ab1r266')
#model=( 'ab3r20' 'ab3r26' 'ab3r32' 'ab3r38' 'ab3r44' 'ab3r50a' 'ab3r56' 'ab3r62' 'ab3r68' 'ab3r74' 'ab3r80' 'ab3r86' 'ab3r92' 'ab3r98' 'ab3r104' 'ab3r110' 'ab3r116' 'ab3r122' 'ab3r128' 'ab3r134' 'ab3r140' 'ab3r146' 'ab3r152a' 'ab3r158')
#model=( '100ab3r20' '100ab3r26' '100ab3r32' '100ab3r38' '100ab3r44' '100ab3r50a' '100ab3r56' '100ab3r62' '100ab3r68' '100ab3r74' '100ab3r80' '100ab3r86' '100ab3r92' '100ab3r98' '100ab3r104' '100ab3r110' '100ab3r116' '100ab3r122' '100ab3r128' '100ab3r134' '100ab3r140' '100ab3r146' '100ab3r152a' '100ab3r158')
dir_loc='noise-ab2pr110-h-1-c10' #noise-ab2pr44-h-1-c10
device='9'
train_dir='train_ori_noise.py' #train_ori1.py train_ori_gradient.py train_ori_gradient_3step.py

for ((h=0;h<1;h++));do #h
    for ((l=0;l<1;l++));do #a0系数
        for ((k=0;k<12;k++));do #b1,b2,b3系数
            for ((i=4;i<5;i++));do #模型
                for ((j=0;j<3;j++));do #重复实验次数
                    #⬇1step⬇
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[device]} --a0 ${a0[l]} --b1 ${b1[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[device]}_a0_${a0[l]}_b1_${b1[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇2step⬇
                    CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[h]} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_${model[i]}_${j}.txt 2>&1
                    #⬇3step⬇ 
                    # CUDA_VISIBLE_DEVICES=${device} nohup /home/lab416/anaconda3/envs/pt37_backup/bin/python ${train_dir} --repnum ${j} --nnlayer ${i} --tao ${k} --epoch ${epoch} -bs ${bs} --lr ${lr} --h ${tao[h]} --a0 ${a0[l]} --b1 ${b1[k]} --b2 ${b2[k]} --b3 ${b3[k]} --arch ${model[i]} --table_loc ${dir_loc}> ./log/${dir_loc}/tao_${tao[h]}_a0_${a0[l]}_b1_${b1[k]}_b2_${b2[k]}_b3_${b3[k]}_${model[i]}_${j}.txt 2>&1
                done;
            done;
        done;
    done;
done;





