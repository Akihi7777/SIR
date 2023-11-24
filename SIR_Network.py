import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA
class SIR_model:
	def __init__(self, beta,miu,t,network,method):
		self.beta = beta
		self.miu=miu #康复概率
		self.t=t
		self.network=network
		self.method=method
		#self.rounds=rounds
		# 选择不同的网络进行生产SIR模型
		if self.network=="ER_network":
			er_network=ER(N=1000,p=0.006,title="ER network")
			self.net_matrix=er_network.Create_ER_network()
		elif self.network=="BA_network":
			ba_network=BA(N=3,p=0.006,N_end=1000,m0=3,title="BA network")
			self.net_matrix=ba_network.Create_BA_network(ba_network.Create_ER_network())
		else:
			print("Please choose correct network")

	# def Generate_SIR_model(self,net_mat):
	# 	#分布记录SIR状态的列表
	# 	SIR_t_seq_s=np.zeros(self.t)
	# 	SIR_t_seq_i=np.zeros(self.t)
	# 	SIR_t_seq_r=np.zeros(self.t)
	# 	for round in range(self.rounds):
	# 		t = self.t
	# 		SIR_list = np.ones(len(net_mat))  # 记录SIR状态的列表
	# 		if self.method == "random_node":
	# 			# 随机选择一个节点作为传播者
	# 			rand_picked_I = np.random.choice(len(net_mat))
	# 			SIR_list[rand_picked_I] = 2
	# 		elif self.method == "max_node":
	# 			# 选择度最大的节点作为传播者
	# 			row_sum = net_mat.sum(axis=1)
	# 			row_sum_max = np.where(row_sum == np.max(row_sum))
	# 			SIR_list[row_sum_max] = 2
	# 		elif self.method == "random_set":
	# 			initial_infections = random.sample(list(range(len(net_mat))), k=int(len(net_mat) * 0.01))
	# 			SIR_list[initial_infections] = 2
	# 		else:
	# 			print("Please choose a method")
	#
	# 		for times in range(t):
	# 			s_index=np.where(SIR_list==1)#s_index为状态为s的序列
	# 			# 在本次循环中需要根据已存在的SIR状态(old_SIR_list)进行操作，
	# 			# 即old_SIR_list为（t-1）时刻的SIR状态
	# 			old_SIR_list=SIR_list
	# 			# I进行感染，需要对每一个s状态的节点进行遍历
	# 			for s in s_index[0]:
	# 				# s为状态为s的节点
	# 				s_row=net_mat[s]# i是状态为s的一行
	# 				link_index=np.where(s_row==1)#在i这行中存在连线的序号
	# 				link_node=old_SIR_list[link_index]#和s这个节点相连的点的序列
	# 				S_to_I_sum=np.sum(link_node==2)#s这个节点和状态为i的节点连线的个数
	# 				P_of_infection=1-(1-self.beta)**S_to_I_sum#该节点被感染的概率
	# 				if random.random()<P_of_infection:
	# 					SIR_list[s]=2
	# 			# I可能康复
	# 			old_I_index=np.where(old_SIR_list==2)
	# 			for i in old_I_index[0]:
	# 				if random.random()<self.miu:
	# 					SIR_list[i]=3
	# 			# 记录每一时刻s,i,r状态的变化情况
	# 			SIR_t_seq_s[times]+=np.sum(SIR_list==1)
	# 			SIR_t_seq_i[times]+=np.sum(SIR_list==2)
	# 			SIR_t_seq_r[times]+=np.sum(SIR_list==3)
	# 	# 返回t时间内s，i，r的数量变化序列
	# 	SIR_t_seq_s=SIR_t_seq_s/self.rounds
	# 	SIR_t_seq_i = SIR_t_seq_i / self.rounds
	# 	SIR_t_seq_r = SIR_t_seq_r / self.rounds
	# 	return SIR_t_seq_s,SIR_t_seq_i,SIR_t_seq_r
	def Generate_SIR_model(self,net_mat):
		t = self.t
		SIR_list = np.ones(len(net_mat))  # 记录SIR状态的列表
		if self.method == "random_node":
			# 随机选择一个节点作为传播者
			rand_picked_I = np.random.choice(len(net_mat))
			SIR_list[rand_picked_I] = 2
		elif self.method == "max_node":
			# 选择度最大的节点作为传播者
			row_sum = net_mat.sum(axis=1)
			row_sum_max = np.where(row_sum == np.max(row_sum))
			SIR_list[row_sum_max] = 2
		elif self.method == "random_set":
			initial_infections = random.sample(list(range(len(net_mat))), k=int(len(net_mat) * 0.01))
			SIR_list[initial_infections] = 2
		else:
			print("Please choose a method")

		# 分布记录SIR状态的列表
		SIR_t_seq_s = []
		SIR_t_seq_i = []
		SIR_t_seq_r = []
		for times in range(t):
			s_index = np.where(SIR_list == 1)  # s_index为状态为s的序列
			# 在本次循环中需要根据已存在的SIR状态(old_SIR_list)进行操作，
			# 即old_SIR_list为（t-1）时刻的SIR状态
			old_SIR_list = SIR_list
			# I进行感染，需要对每一个s状态的节点进行遍历
			for s in s_index[0]:
				# s为状态为s的节点
				s_row = net_mat[s]  # i是状态为s的一行
				link_index = np.where(s_row == 1)  # 在i这行中存在连线的序号
				link_node = old_SIR_list[link_index]  # 和s这个节点相连的点的序列
				S_to_I_sum = np.sum(link_node == 2)  # s这个节点和状态为i的节点连线的个数
				P_of_infection = 1 - (1 - self.beta) ** S_to_I_sum  # 该节点被感染的概率
				if random.random() < P_of_infection:
					SIR_list[s] = 2
			# I可能康复
			old_I_index = np.where(old_SIR_list == 2)
			for i in old_I_index[0]:
				if random.random() < self.miu:
					SIR_list[i] = 3
			# 记录每一时刻s,i,r状态的变化情况
			SIR_t_seq_s.append(np.sum(SIR_list == 1))
			SIR_t_seq_i.append(np.sum(SIR_list == 2))
			SIR_t_seq_r.append(np.sum(SIR_list == 3))
		# 返回t时间内s，i，r的数量变化序列
		return SIR_t_seq_s, SIR_t_seq_i, SIR_t_seq_r

	def Graw_SIR(self,rounds=30):
		SIR_t_seq_s_count = np.zeros(self.t)
		SIR_t_seq_i_count = np.zeros(self.t)
		SIR_t_seq_r_count = np.zeros(self.t)
		for i in range(rounds):
			SIR_t_seq_s,SIR_t_seq_i,SIR_t_seq_r=self.Generate_SIR_model(self.net_matrix)
			SIR_t_seq_s_count+=np.array(SIR_t_seq_s)
			SIR_t_seq_i_count+=np.array(SIR_t_seq_i)
			SIR_t_seq_r_count+=np.array(SIR_t_seq_r)
		SIR_t_seq_s_count_ave=SIR_t_seq_s_count/rounds
		SIR_t_seq_i_count_ave=SIR_t_seq_i_count/rounds
		SIR_t_seq_r_count_ave=SIR_t_seq_r_count/rounds
		pl.subplot(111)
		pl.plot(SIR_t_seq_s_count_ave, '-g', label='Susceptibles')
		pl.plot(SIR_t_seq_i_count_ave, '-r', label='Infectious')
		pl.plot(SIR_t_seq_r_count_ave, '-k', label='Recovereds')
		pl.legend(loc=0)
		pl.title('SIR_Model based on '+self.network)
		pl.xlabel('Time')
		pl.ylabel('Infectious Susceptibles and Recovereds')
		pl.show()


	def main(self):
		self.Graw_SIR()















