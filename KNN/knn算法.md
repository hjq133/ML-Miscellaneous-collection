寻找点P的k个最近邻：



创建size=k的空列表list[k]，创建kd树。



令当前节点为根节点

Loop (当前节点不是底部节点)：

​    	假设当前节点是依据轴r进行划分的），若 Curr_r > P_r，向Curr左支进行搜索，反之进行向右支搜索。



当前节点为底部节点, 设置Visit[curr] = 1



func A() {

​	if( list.size() < K ) {

​		list.add(当前底部节点curr).

​	} else {  // list.size() = k, 表示满了 

​		计算当前节点与P的距离，若小于list中最长的，则list.add(curr).

​	}

}



if (curr 节点不是根节点):

​	(a) 	curr = curr.parent（向上爬一个节点）

​				if(visit[curr] == 0) {  如果当前节点未被访问过

​						visit[curr] = 1;

​						func A

​				}

else 