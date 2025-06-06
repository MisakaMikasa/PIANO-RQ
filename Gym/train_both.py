import random
from DDPG_V3 import DDPG_main
from DQN_V2 import DQN_main
from gensubgraph2 import bfs_sample
import time



t = 0
while t < 10 : 

      
      start_node = random.randint(1,6300)
      max_node = random.randint(300,450)

      input_txt = "./New Graph Dataset/p2p(1).txt"
      output_txt = "./New Graph Dataset/subgraph1.txt"
      print(t)
      bfs_sample(input_txt, output_txt, start_node, max_node)

      start_time = time.time()
      #DDPG_main(max_node)
      end_time1 = time.time()

      DQN_main(max_node)
      end_time2 = time.time()


      runtime1 = end_time1 - start_time
      runtime2 = end_time2 - end_time1


      if(runtime2>10):
            t+=1
      print(f"Runtime1: {runtime1:.2f} seconds")
      print(f"Runtime2: {runtime2:.2f} seconds")
    


