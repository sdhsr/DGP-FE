
#!/bin/bash
commands=(
##实验室--patchsize的影响？

#
##----todo----

##消融实验
#"python main.py --machine machine --patch_size=33 --dataset GraphShenZhen_288 --wide=627 --num_nodes=627   --used_data 'GridGraphall' --task_id UniFlow_A_F --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=3 --dataset GraphMetrla_288  --wide=207 --num_nodes=207    --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id UniFlow_A_F --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170    --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
######云30014
#"python main.py --machine machine --patch_size=33 --dataset GraphShenZhen_288 --wide=627 --num_nodes=627   --used_data 'GridGraphall' --early_stop=12 --task_id UniFlow_A_D_M_F --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=3 --dataset GraphMetrla_288  --wide=207 --num_nodes=207  --early_stop=12 --task_id UniFlow_A_D_M_F   --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228   --early_stop=12 --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"


#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --early_stop=12 --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307  --early_stop=12 --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#"python main.py --machine machine --patch_size=3 --dataset GraphPems03_288 --wide=358 --num_nodes=358    --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"

#"python main.py --machine machine --patch_size=3 --dataset GraphPems03_288 --wide=358 --num_nodes=358    --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307   --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphChengdu_288 --wide=524 --num_nodes=524  --used_data 'GridGraphall'  --task_id UniFlow_A_F --batch_size_graph_large=60  --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228   --task_id UniFlow_A_F  --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsBay_288 --wide=325 --num_nodes=325   --task_id UniFlow_A_D_M --used_data 'GridGraphall' --batch_size_graph_large=60 --model UniFlow_A_D_M"
##
#
####修改1-- x——>out
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
##修改2--重要性频率
##"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
###修改3--序列分解
##"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow_our_D --used_data 'GridGraphall' --model UniFlow_our_D"
####修改4-用特征
##"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow_our_F --used_data 'GridGraphall' --model UniFlow_our_F"
##
##########---实验
##已完成
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170    --task_id P1_UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow --used_data 'GridGraphall' --model UniFlow"
#
#
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P2_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P2_UniFlow --used_data 'GridGraphall' --model UniFlow"
#
#"python main.py --machine machine --patch_size=4 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P4_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=4 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P4_UniFlow --used_data 'GridGraphall' --model UniFlow"
#
#"python main.py --machine machine --patch_size=8 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P8_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=8 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P8_UniFlow --used_data 'GridGraphall' --model UniFlow"

#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id test --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

###

#"python main.py --machine machine --patch_size=12 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P12_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=12 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P12_UniFlow --used_data 'GridGraphall' --model UniFlow"
#
#"python main.py --machine machine --patch_size=16 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P16_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=12 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id test --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
##
#
#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P2_UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=4 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P4_UniFlow_A_D_M_F_I--used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#





###修改2--重要性频率
#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
##修改3--序列分解
#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_our_D"

#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F --batch_size_graph_large=60 --used_data 'GridGraphall' --model UniFlow_A_D_M"

#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M"

#
###修改4-用特征
#"python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_our_F"
#




#----todo 成员推理攻击---MAE优化
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id UniFlow_A_D_M_F --batch_size_graph_large=80 --used_data 'GridGraphall' --mode testing --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170   --task_id UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --cut=0.1 --balance=0.2 --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id Ours_c_0.1_b_  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"



#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170    --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170  --mode testing  --task_id clean_UniFlow  --used_data 'GridGraphall' --model UniFlow"

#--todo-- 服务器34787
#"python main.py --machine machine --cut=0.6 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c6_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c6_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c6_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c6_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c6_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#"python main.py --machine machine --cut=0.4 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c4_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c4_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c4_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c4_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c4_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#
#"python main.py --machine machine --cut=0.2 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c2_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.2 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c2_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c2_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c2_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c2_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#
#"python main.py --machine machine --cut=0.8 --balance=0.6 -  -patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c8_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.8 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c8_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.8 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c8_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.8 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c8_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.8 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c8_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#"python main.py --machine machine --cut=0.1 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c1_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.1 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c1_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.1 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c1_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.1 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c1_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.1 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=160  --task_id AutoFlow_c1_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"


###todo ---服务器14080
"python main.py --machine machine --cut=0.1 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c1_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.1 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c1_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.1 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c1_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.1 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c1_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.1 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c1_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"


"python main.py --machine machine --cut=0.8 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c8_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.8 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c8_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.8 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c8_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.8 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c8_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.8 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c8_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"



"python main.py --machine machine --cut=0.2 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c2_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
"python main.py --machine machine --cut=0.2 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c2_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c2_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c2_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.2 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c2_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#
#"python main.py --machine machine --cut=0.4 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c4_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c4_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c4_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c4_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.4 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c4_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
#
#"python main.py --machine machine --cut=0.6 --balance=0.1 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c6_b1 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.2 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c6_b2 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.4 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c6_b4 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.6 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c6_b6 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --cut=0.6 --balance=0.8 --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170  --batch_size_graph_large=80  --task_id AutoFlow_c6_b8 --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#



#"python main.py --machine machine --patch_size=33 --dataset GraphShenZhen_288 --wide=627 --num_nodes=627   --mode testing --used_data 'GridGraphall' --task_id clean_UniFlow --batch_size_graph_large=80 --model UniFlow"
#"python main.py --machine machine --patch_size=2 --dataset GraphMetrla_288  --wide=207 --num_nodes=207    --mode testing   --task_id relaxloss  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphChengdu_288 --wide=524 --num_nodes=524  --used_data 'GridGraphall'  --mode testing  --task_id P1_UniFlow_A_D_M_F  --batch_size_graph_large=40  --model UniFlow_A_D_M_F"



#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307  --task_id UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307  --task_id UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170    --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1--dataset GraphShenZhen_288 --wide=627 --num_nodes=627   --used_data 'GridGraphall' --task_id UniFlow_A_D_M_F --batch_size_graph_large=80 --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_ALL"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_ALL"
#"python main.py --machine machine --patch_size=1 --dataset GraphShenZhen_288 --wide=627 --num_nodes=627   --used_data 'GridGraphall' --task_id P1_UniFlow_A_D_M_F --batch_size_graph_large=40 --model UniFlow_ALL"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems03_288 --wide=358 --num_nodes=358    --task_id P1_UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_ALL"
#"python main.py --machine machine --patch_size=1 --dataset GraphChengdu_288 --wide=524 --num_nodes=524  --used_data 'GridGraphall'  --task_id P1_UniFlow_A_D_M_F  --batch_size_graph_large=60  --model UniFlow_ALL"

#31672

#"python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_20  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --seq_len=72 --his_len=36 --pred_len=36 --patch_size=3 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow  --used_data 'GridGraphall' --model UniFlow"
#"python main.py --machine machine --seq_len=72 --his_len=36 --pred_len=36 --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228    --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#----------------------------------------------------------------------------------


#"python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_20  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F_I --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_20  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
#
#"python main.py --machine machine  --patch_size=1 --dataset GraphPemsd7_288_noise_15 --wide=228 --num_nodes=228  --early_stop=12  --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine  --patch_size=2 --dataset GraphPemsd7_288_noise_15 --wide=228 --num_nodes=228  --early_stop=12  --task_id UniFlow  --used_data 'GridGraphall' --model UniFlow"
#
#"python main.py --machine machine  --patch_size=2 --dataset GraphPems08_speed_288  --wide=170 --num_nodes=170 --task_id clean_UniFlow --early_stop=10  --used_data 'GridGraphall' --mode testing --model UniFlow"

#----------------------------------------------------------------------------------
#"python main.py --machine machine --seq_len=128 --his_len=64 --pred_len=64  --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow  --used_data 'GridGraphall' --model UniFlow"
#"python main.py --machine machine --seq_len=24 --his_len=12 --pred_len=12 --patch_size=1 --dataset GraphPems08_flow_288  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --mode testing --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --seq_len=128 --his_len=64 --pred_len=64  --patch_size=1 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"


#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_speed_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=1 --dataset GraphMetrla_288  --wide=207 --num_nodes=207    --task_id P1_UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems03_288 --wide=358 --num_nodes=358    --task_id P1_UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --patch_size=1 --dataset GraphChengdu_288 --wide=524 --num_nodes=524  --used_data 'GridGraphall'  --task_id P1_UniFlow_A_D_M_F  --batch_size_graph_large=60  --model UniFlow_A_D_M_F"
#

#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --batch_size_graph_large=60 --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --batch_size_graph_large=60 --model UniFlow_our_D"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307  --task_id P1_UniFlow_A_D_M_F --used_data 'GridGraphall' --batch_size_graph_large=60 --model UniFlow_our_F"
#
#"python main.py --machine machine --patch_size=12 --dataset GraphMetrla_288  --wide=207 --num_nodes=207    --task_id test  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"




#   "python main.py --machine machine --patch_size=2 --dataset TaxiNYCIn_48 --wide=10 --height=20 --num_nodes=200 --isGraph=False --log_interval=5  --early_stop=10 --used_data 'GridGraphall'  --task_id UniFlow_A_D_M_F_I  --model UniFlow_A_D_M_F_I"
#  "python main.py --machine machine --patch_size=2 --dataset TaxiBJ13_48 --wide=32 --height=32 --num_nodes=1024 --isGraph=False  --used_data 'GridGraphall'   --task_id UniFlow_A_D_M_F_I  --model UniFlow_A_D_M_F_I"

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170   --task_id UniFlow_A_D_M_F_I --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=3 --dataset GraphMetrla_288  --wide=207 --num_nodes=207   --task_id UniFlow_A_D_M_F_I   --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=3 --dataset GraphPems03_288 --wide=358 --num_nodes=358    --task_id UniFlow_A_D_M_F_I  --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#
#

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id offline   --train_mode AT_policy_atten_dist_offline --used_data 'GridGraphall' --model UniFlow_A_D_M_F  --policynet_path experiments_all/Exp/GraphPems08_speed_288_his_12_pred_12_baseline/model_save/policy_epoch12.pt"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id offline   --train_mode AT_policy_atten_dist_offline --mode testing --used_data 'GridGraphall' --model UniFlow_A_D_M_F  --policynet_path experiments_all/Exp/GraphPems08_speed_288_his_12_pred_12_baseline/model_save/policy_epoch12.pt"
#

##AT_policy_atten--todo
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170 --task_id UniFlow_baseline --train_mode AT_policy_atten    --used_data 'GridGraphall' --model UniFlow"

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170 --task_id UniFlow_offline --train_mode AT_policy_atten_dist_offline  --mode testing   --used_data 'GridGraphall' --model UniFlow  --policynet_path experiments_all/Exp/GraphPems08_flow_288_his_12_pred_12_UniFlow_baseline/model_save/policy_epoch22.pt"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow_offline   --train_mode AT_policy_atten_dist_offline   --mode testing --used_data 'GridGraphall' --model UniFlow  --policynet_path experiments_all/Exp/GraphPems08_speed_288_his_12_pred_12_UniFlow_baseline/model_save/policy_epoch24.pt"

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170 --task_id baseline --train_mode AT_policy_atten    --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --patch_size=2 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228  --task_id UniFlow_offline    --train_mode AT_policy_atten_dist_offline   --mode testing  --batch_size_graph_large=40 --used_data 'GridGraphall'  --model UniFlow --policynet_path experiments_all/Exp/GraphPemsd7_288_his_12_pred_12_UniFlow_baseline/model_save/policy_epoch30.pt"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems08_flow_288 --wide=170 --num_nodes=170 --task_id offline --train_mode AT_policy_atten_dist_offline    --mode testing  --used_data 'GridGraphall' --model UniFlow_A_D_M_F   --policynet_path experiments_all/Exp/GraphPems08_flow_288_his_12_pred_12_baseline/model_save/policy_epoch19.pt"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id baseline   --train_mode AT_policy_atten    --used_data 'GridGraphall' --model UniFlow_A_D_M_F "

#"python main.py --machine machine --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow_offline   --train_mode AT_policy_atten_dist_offline   --mode testing --used_data 'GridGraphall' --model UniFlow  --policynet_path experiments_all/Exp/GraphPems08_speed_288_his_12_pred_12_UniFlow_baseline/model_save/policy_epoch24.pt"
#

#服务器
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --task_id UniFlow_baseline --train_mode AT_policy_atten  --used_data 'GridGraphall' --model UniFlow"
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --task_id baseline --train_mode AT_policy_atten  --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I"
#"python main.py --machine machine --patch_size=2 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --task_id UniFlow_offline --train_mode AT_policy_atten_dist_offline  --used_data 'GridGraphall' --model UniFlow --policynet_path  "
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --task_id offline --train_mode AT_policy_atten_dist_offline  --used_data 'GridGraphall' --model UniFlow_A_D_M_F_I --policynet_path  "
#



###UniFlow和我们模型的鲁棒性vs加了保护的鲁棒性
#   "python main.py --machine machine --patch_size=3 --dataset GraphMetrla_288 --wide=207 --num_nodes=207 --task_id clean_UniFlow   --used_data 'GridGraphall' --model UniFlow"
#   "python main.py --machine machine --patch_size=3 --dataset GraphMetrla_288 --wide=207 --num_nodes=207 --task_id clean   --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
##   "python main.py --machine machine --patch_size=4 --dataset GraphChengdu_288 --wide=524 --num_nodes=524   --task_id clean_UniFlow   --used_data 'GridGraphall' --model UniFlow"





##实验室
#"python main.py --machine machine --patch_size=1 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --used_data 'GridGraphall' --task_id UniFlow_A_D_M_F --mode testing  --model UniFlow_A_D_M_F"
##
#  "python main.py --machine machine --patch_size=1 --dataset GraphPemsd7_288 --wide=228 --num_nodes=228   --used_data 'GridGraphall' --model UniFlow"
#   "python main.py --machine machine --patch_size=3 --dataset GraphPems03_288 --wide=358 --num_nodes=358   --used_data 'GridGraphall' --early_stop=5 --model UniFlow"
#   "python main.py --machine machine --patch_size=3 --dataset GraphPems04_flow_288 --wide=307 --num_nodes=307   --used_data 'GridGraphall'  --model UniFlow"
#
#   "python main.py --machine machine --patch_size=3 --dataset GraphPemsBay_288 --wide=325 --num_nodes=325   --used_data 'GridGraphall' --early_stop=5  --model UniFlow_D_M_F"

#   #"python main.py --machine machine   --used_data 'GridGraphall'  --dataset='GraphSH_28_96' --wide=21099 --num_nodes=21099 --batch_size_graph_large=8"
#   #"python main.py --machine machine   --used_data 'GridGraphall'  --dataset='GraphPEMSL_43_288' --wide=1026 --num_nodes=1026 --batch_size_graph_large=5"

)    #--machine machine   --used_data GridGraphall   --prompt_content node_graph --flag=0 --model DCN_model --task_id uni_testtesttest

## 运行每个命令
#for cmd in "${commands[@]}"
#d
#    echo "Running: $cmd"
#    eval $cmd
#    echo "Press Enter to continue..."
#    read -s -p "" # 等待用户按下Enter键，-s表示不显示输入，-p ""显示空提示
#done

# 运行每个命令
for cmd in "${commands[@]}"
do
    echo "Running: $cmd"
    eval $cmd
done
