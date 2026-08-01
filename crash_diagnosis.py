import os,sys; os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, gymnasium, highway_env, json
from dqn_agent import DQNAgent
a,b=int(sys.argv[1]),int(sys.argv[2])
CFG={'action':{'type':'DiscreteMetaAction'},'lanes_count':3,'ego_spacing':1.5,'vehicles_count':50,'duration':40}
ag=DQNAgent(state_dim=25,action_dim=5,hidden_dim=128); ag.load('weights/dqn_highway.pt')
env=gymnasium.make('highway-v0',config=CFG)
diag=[]
for i in range(a,b):
    s,_=env.reset(seed=90000+i); s=s.reshape(-1); d=t=False
    while not(d or t):
        u=env.unwrapped; ego=u.vehicle; ot=u.observation_type
        # exactly how KinematicObservation picks the rows it shows
        close=u.road.close_vehicles_to(ego, ot.env.PERCEPTION_DISTANCE,
                                       count=ot.vehicles_count-1, see_behind=ot.see_behind)
        observed_ids=[id(v) for v in close]
        prev_x=ego.position[0]
        act=ag.select_action(s,evaluate=True)
        s,r,d,t,_=env.step(act); s=s.reshape(-1)
        if d:
            u=env.unwrapped; ego=u.vehicle
            cands=sorted([(np.linalg.norm(np.array(v.position)-np.array(ego.position)),v)
                          for v in u.road.vehicles if v is not ego])
            dist,partner=cands[0]
            diag.append({'seed':90000+i,'rel_x':round(float(partner.position[0]-prev_x),2),
                         'was_observed':bool(id(partner) in observed_ids),
                         'n_observed':len(observed_ids)})
env.close()
json.dump(diag,open(f'/tmp/d2_{a}_{b}.json','w'))
for c in diag: print('  ',c)
print(f'seeds {a}-{b}: {len(diag)} crashes')
