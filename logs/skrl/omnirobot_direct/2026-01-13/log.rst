1. 再次训练原版本 distance 和 direction整合
2. 尝试修改distance reward
3. 尝试将success bonus 的阈值修改一下
4. 将每一局训练时间从5秒增加到15秒
5. reward修改之后再次训练
6. 将4800变成10000;之后训练时间增加到20秒
7. 重新设计reward函数,训练重新变成4800,之后训练时间变成10秒 reward函数变得好看了一些
8. 继续训练,将reward函数中的distance reward的权重修改,时间变成5秒
