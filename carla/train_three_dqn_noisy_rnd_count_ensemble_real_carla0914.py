"""Train/test 3 advanced DQN experiments in REAL CARLA 0.9.14.

Experiments:
  1. noisy_count
     NoisyNet DQN + count-based intrinsic reward.
  2. rnd_count
     RND DQN + count-based intrinsic reward.
  3. ensemble_own_noisy_rnd_count
     Its own neural network chooses among three candidate actions:
       - own DQN best action
       - trained NoisyNet expert best action
       - trained RND expert best action
     The ensemble learner uses NoisyNet + RND + count-based intrinsic reward.

Graphs match the SUMO-style project format:
  average_reward_vs_learning_rate.png
  best_learning_rate_by_experiment.png
  convergence_time_vs_experiment.png
  average_reward_vs_epsilon.png

CARLA must already be running, for example on Ubuntu:
  ./CarlaUE4.sh -quality-level=Low
or headless:
  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound
"""
from __future__ import annotations

import argparse, copy, math, random, time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.carla_env import CarlaDrivingEnv
from src.dqn_agent import DQNAgent, NoisyDQNAgent, RNDDQNAgent, NoisyRNDDQNAgent

EXPERIMENTS = ["noisy_count", "rnd_count", "ensemble_own_noisy_rnd_count"]
EXPERIMENT_LABELS = {
    "noisy_count": "Noisy + Count",
    "rnd_count": "RND + Count",
    "ensemble_own_noisy_rnd_count": "Ensemble Own/Noisy/RND + Count",
}
EXPERIMENT_COLORS = {
    "noisy_count": "tab:blue",
    "rnd_count": "tab:orange",
    "ensemble_own_noisy_rnd_count": "tab:green",
}


def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed)


def make_env(args)->CarlaDrivingEnv:
    return CarlaDrivingEnv(host=args.host, port=args.port, timeout_seconds=args.timeout_seconds,
                           reward_mode=args.reward_mode, target_speed_kmh=args.target_speed,
                           max_episode_steps=args.max_episode_steps, use_mock_when_carla_missing=False)


def average(values:Iterable[float])->float:
    values=list(values); return float(sum(values)/max(len(values),1))


def reward_mode(values:Sequence[float])->float:
    if not values: return 0.0
    rounded=[round(float(v),6) for v in values]; counts={}
    for v in rounded: counts[v]=counts.get(v,0)+1
    mc=max(counts.values()); return float(max(v for v,c in counts.items() if c==mc))


def score(rewards:Sequence[float])->float:
    rewards=np.asarray(rewards,dtype=float)
    return float(0.20*np.mean(rewards)+0.50*np.median(rewards)-0.30*(np.max(rewards)-np.min(rewards)))


def convergence_episode(train_rewards:Sequence[float], target:float, frac:float, window:int)->int:
    if not train_rewards: return 0
    window=max(1,min(int(window),len(train_rewards)))
    rolling=np.convolve(np.asarray(train_rewards,dtype=float), np.ones(window)/window, mode='valid')
    for i,v in enumerate(rolling):
        if float(v)>=float(frac)*float(target): return int(i+window)
    return int(len(train_rewards))


class CountBasedBonus:
    def __init__(self,beta:float=0.05,bin_size:float=1.0):
        self.beta=float(beta); self.bin_size=max(float(bin_size),1e-6); self.counts:Dict[tuple[int,...],int]={}
    def key(self,state:np.ndarray)->tuple[int,...]:
        return tuple(np.round(np.asarray(state,dtype=float)/self.bin_size).astype(int).tolist())
    def bonus(self,state:np.ndarray)->tuple[float,int]:
        k=self.key(state); n=self.counts.get(k,0)+1; self.counts[k]=n
        return float(self.beta/math.sqrt(n)), int(n)


def build_agent(exp:str, obs:int, actions:int, args, lr:float)->DQNAgent:
    common=dict(observation_size=obs, action_size=actions, learning_rate=lr, gamma=args.gamma,
                batch_size=args.batch_size, replay_capacity=args.replay_capacity,
                target_update_interval=args.target_update_interval, inference_margin=args.inference_margin,
                device=args.device)
    if exp=="noisy_count": return NoisyDQNAgent(**common)
    if exp=="rnd_count": return RNDDQNAgent(**common, rnd_beta=args.rnd_beta)
    if exp=="ensemble_own_noisy_rnd_count": return NoisyRNDDQNAgent(**common, rnd_beta=args.rnd_beta)
    raise ValueError(exp)


def select_ensemble_action(state:np.ndarray, own:DQNAgent, experts:Sequence[Tuple[str,DQNAgent]])->int:
    candidates=[]; seen=set()
    own_action=int(own.best_action(state)); candidates.append((own_action,"own")); seen.add(own_action)
    for name,agent in experts:
        a=int(agent.best_action(state))
        if a not in seen:
            candidates.append((a,name)); seen.add(a)
    q=np.asarray(own.get_q_values(state),dtype=float)
    return int(max(candidates, key=lambda item: float(q[item[0]]))[0])


def train_one(exp:str, lr:float, lr_mult:float, args, out:Path, experts:Sequence[Tuple[str,DQNAgent]]|None=None):
    env=make_env(args); obs=env.observation_space.shape[0]; actions=env.action_space.n
    agent=build_agent(exp,obs,actions,args,lr); counts=CountBasedBonus(args.count_beta,args.count_state_bin_size)
    rows=[]; train_rewards=[]
    for ep in range(args.train_episodes):
        state,_=env.reset(); total=0.0; losses=[]; steps=0; done=False; start=time.time()
        while not done:
            if exp=="ensemble_own_noisy_rnd_count":
                action=select_ensemble_action(state, agent, experts or [])
            elif exp=="rnd_count":
                action=int(agent.select_action(state, epsilon=args.epsilon))
            else:
                action=int(agent.select_action(state, epsilon=0.0))
            next_state, env_reward, terminated, truncated, _=env.step(action); done=terminated or truncated
            count_bonus,_n=counts.bonus(next_state)
            rnd_intrinsic=0.0
            if hasattr(agent,"intrinsic_reward") and hasattr(agent,"train_rnd_predictor"):
                rnd_intrinsic=float(agent.intrinsic_reward(next_state)); agent.train_rnd_predictor(next_state)
            training_reward=float(env_reward)+count_bonus+float(args.rnd_beta)*rnd_intrinsic
            # Always update the ensemble learner
            agent.remember(state, action, training_reward, next_state, done)
            loss=agent.learn()
            if loss is not None: losses.append(float(loss))

            # Ensemble rule requested by user:
            # while the ensemble chooses among own/noisy/RND candidate actions,
            # the noisy and RND expert networks must keep learning from the same
            # selected transition instead of staying frozen.
            if exp == "ensemble_own_noisy_rnd_count" and experts:
                for expert_name, expert_agent in experts:
                    expert_rnd_intrinsic = 0.0
                    if hasattr(expert_agent, "intrinsic_reward") and hasattr(expert_agent, "train_rnd_predictor"):
                        expert_rnd_intrinsic = float(expert_agent.intrinsic_reward(next_state))
                        expert_agent.train_rnd_predictor(next_state)
                    expert_training_reward = float(env_reward) + count_bonus + float(args.rnd_beta) * expert_rnd_intrinsic
                    expert_agent.remember(state, action, expert_training_reward, next_state, done)
                    expert_loss = expert_agent.learn()
                    if expert_loss is not None:
                        losses.append(float(expert_loss))

            total+=float(env_reward); state=next_state; steps+=1
        train_rewards.append(total)
        rows.append({"phase":"train","experiment":exp,"episode":ep,"env_reward":total,"steps":steps,
                     "convergence_time_seconds":time.time()-start,"average_loss":average(losses),
                     "epsilon":args.epsilon,"gamma":args.gamma,"learning_rate":lr,"lr_multiplier":lr_mult,
                     "dqn_technology":"Noisy/RND/Count candidate ensemble" if exp.startswith('ensemble') else EXPERIMENT_LABELS[exp],
                     "rnd_beta":args.rnd_beta,"count_beta":args.count_beta,"count_state_bin_size":args.count_state_bin_size})
        print(f"TRAIN {EXPERIMENT_LABELS[exp]:34s} ep={ep:03d} reward={total:.2f} steps={steps}")
    (out/'models').mkdir(parents=True,exist_ok=True); agent.save(str(out/'models'/f'{exp}_lrmult_{lr_mult:g}.pt'))
    pd.DataFrame(rows).to_csv(out/f'{exp}_lrmult_{lr_mult:g}_train.csv',index=False); env.close()
    return agent, train_rewards


def test_one(exp:str, agent:DQNAgent, args, lr:float, lr_mult:float, experts=None):
    env=make_env(args); rows=[]
    for ep in range(args.test_episodes):
        state,_=env.reset(); total=0.0; steps=0; done=False; start=time.time()
        while not done:
            action=select_ensemble_action(state,agent,experts or []) if exp=="ensemble_own_noisy_rnd_count" else int(agent.best_action(state))
            next_state,r,terminated,truncated,_=env.step(action); done=terminated or truncated
            total+=float(r); state=next_state; steps+=1
        rows.append({"phase":"test","experiment":exp,"episode":ep,"env_reward":total,"steps":steps,
                     "convergence_time_seconds":time.time()-start,"epsilon":args.epsilon,"gamma":args.gamma,
                     "learning_rate":lr,"lr_multiplier":lr_mult})
        print(f"TEST  {EXPERIMENT_LABELS[exp]:34s} ep={ep:03d} reward={total:.2f} steps={steps}")
    env.close(); return rows


def summarize(exp,lr_mult,lr,train_rewards,test_rows,args):
    rewards=[float(r['env_reward']) for r in test_rows]
    return {"experiment":exp,"lr_multiplier":float(lr_mult),"learning_rate":float(lr),
            "average_train_reward":float(np.mean(train_rewards)),"average_test_reward":float(np.mean(rewards)),
            "median_test_reward":float(np.median(rewards)),"mode_test_reward":reward_mode(rewards),
            "min_test_reward":float(np.min(rewards)),"max_test_reward":float(np.max(rewards)),
            "range_test_reward":float(np.max(rewards)-np.min(rewards)),"std_test_reward":float(np.std(rewards)),
            "best_score":score(rewards),"convergence_episode":convergence_episode(train_rewards,float(np.mean(rewards)),args.convergence_threshold_fraction,args.convergence_window),
            "test_rewards":rewards,"epsilon":args.epsilon,"rnd_beta":args.rnd_beta,"count_beta":args.count_beta}


def text_panel(fig,x,y,text,color):
    fig.text(x,y,text,ha='left',va='top',fontsize=7,color='black',bbox=dict(boxstyle='round,pad=0.25',fc='white',ec=color,alpha=.94),zorder=20)


def save_avg_lr(results,out,args):
    fig,ax=plt.subplots(figsize=(10,5.4))
    for i,exp in enumerate(EXPERIMENTS):
        rows=sorted([r for r in results if r['experiment']==exp],key=lambda r:r['learning_rate'])
        xs=[r['learning_rate'] for r in rows]; ys=[r['average_test_reward'] for r in rows]
        ax.plot(xs,ys,marker='o',linewidth=1.8,color=EXPERIMENT_COLORS[exp],label=EXPERIMENT_LABELS[exp])
        best=max(rows,key=lambda r:(r['average_test_reward'],r['best_score']))
        ax.scatter([best['learning_rate']],[best['average_test_reward']],marker='*',s=260,color=EXPERIMENT_COLORS[exp],edgecolor='black',zorder=8)
        text_panel(fig,.70,.82-i*.14,f"Best {EXPERIMENT_LABELS[exp]}\nLR={best['learning_rate']:.3g}\nAvg={best['average_test_reward']:.1f}\nS={best['best_score']:.1f}",EXPERIMENT_COLORS[exp])
    ax.set_xscale('log'); ticks=sorted({float(r['learning_rate']) for r in results}); ax.set_xticks(ticks); ax.set_xticklabels([f'{v:.6g}' for v in ticks],rotation=35,ha='right',fontsize=7)
    ax.set_xlabel('Learning rate'); ax.set_ylabel('Average test reward'); ax.set_title('CARLA 0.9.14 Average Reward vs Learning Rate',pad=14); ax.grid(True,alpha=.35)
    ax.legend(loc='lower left',bbox_to_anchor=(1.03,.02),fontsize=8,frameon=True); fig.text(.70,.92,'Best starred configurations',ha='left',va='top',fontsize=9,weight='bold')
    fig.text(.5,.985,f"epsilon={args.epsilon:g}; S=0.20×Average + 0.50×Median - 0.30×Range",ha='center',va='top',fontsize=8)
    fig.tight_layout(rect=[0,0,.67,.92]); fig.savefig(out/'average_reward_vs_learning_rate.png',dpi=300); plt.close(fig)


def save_best_lr(results,out):
    best_rows=[max([r for r in results if r['experiment']==exp],key=lambda r:(r['average_test_reward'],r['best_score'])) for exp in EXPERIMENTS]
    fig,ax1=plt.subplots(figsize=(10,5.4)); pos=list(range(1,len(best_rows)+1)); box=ax1.boxplot([r['test_rewards'] for r in best_rows],positions=pos,patch_artist=True,widths=.55)
    for p,row in zip(box['boxes'],best_rows): p.set_facecolor(EXPERIMENT_COLORS[row['experiment']]); p.set_alpha(.45); p.set_edgecolor('black')
    for i,(x,row) in enumerate(zip(pos,best_rows)):
        exp=row['experiment']; ax1.scatter(x,row['average_test_reward'],marker='*',s=250,color=EXPERIMENT_COLORS[exp],edgecolor='black',zorder=8)
        text_panel(fig,.70,.82-i*.14,f"{EXPERIMENT_LABELS[exp]}\nAvg={row['average_test_reward']:.1f}\nLR={row['learning_rate']:.3g}\nS={row['best_score']:.1f}",EXPERIMENT_COLORS[exp])
    ax1.set_xticks(pos); ax1.set_xticklabels([f"{r['learning_rate']:.3g}" for r in best_rows],fontsize=8); ax1.set_xlabel('Learning rate'); ax1.set_ylabel('Average test reward / distribution'); ax1.set_title('CARLA 0.9.14 Best Learning Rate by Experiment',pad=16); ax1.grid(True,axis='y',alpha=.35)
    ax2=ax1.twinx(); ax2.plot(pos,[r['best_score'] for r in best_rows],marker='D',linestyle='--',linewidth=1.4,color='black',label='Score S'); ax2.set_ylabel('Score S')
    handles=[Patch(facecolor=EXPERIMENT_COLORS[e],alpha=.45,label=EXPERIMENT_LABELS[e]) for e in EXPERIMENTS]; handles.append(Patch(facecolor='white',edgecolor='black',label='Star = maximum average reward'))
    ax1.legend(handles=handles,loc='lower left',bbox_to_anchor=(1.03,.02),fontsize=8,frameon=True); ax2.legend(loc='upper left',bbox_to_anchor=(1.03,.98),fontsize=8,frameon=True)
    fig.text(.70,.92,'Best box-plot configurations',ha='left',va='top',fontsize=9,weight='bold'); fig.tight_layout(rect=[0,0,.67,.92]); fig.savefig(out/'best_learning_rate_by_experiment.png',dpi=300); plt.close(fig)


def save_conv(results,out):
    best=[max([r for r in results if r['experiment']==exp],key=lambda r:(r['average_test_reward'],r['best_score'])) for exp in EXPERIMENTS]
    xs=list(range(1,len(best)+1)); vals=[int(r['convergence_episode']) for r in best]; fig,ax=plt.subplots(figsize=(10,5.4)); bars=ax.bar(xs,vals,width=.6,edgecolor='black')
    for i,(bar,row) in enumerate(zip(bars,best)):
        exp=row['experiment']; bar.set_color(EXPERIMENT_COLORS[exp]); bar.set_alpha(.65); ax.text(i+1,bar.get_height()+.5,str(int(row['convergence_episode'])),ha='center',fontsize=8,bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.85)); text_panel(fig,.70,.82-i*.14,f"{EXPERIMENT_LABELS[exp]}\nConv={int(row['convergence_episode'])} ep\nLR={row['learning_rate']:.3g}\nAvg={row['average_test_reward']:.1f}",EXPERIMENT_COLORS[exp])
    ax.set_xticks(xs); ax.set_xticklabels([EXPERIMENT_LABELS[r['experiment']] for r in best],fontsize=8); ax.set_xlabel('Experiment'); ax.set_ylabel('Convergence episode'); ax.set_title('CARLA 0.9.14 Convergence Time vs Experiments',pad=16); ax.grid(True,axis='y',alpha=.35)
    if vals: ax.set_ylim(0,max(vals)*1.18+1)
    fig.text(.70,.92,'Convergence summaries',ha='left',va='top',fontsize=9,weight='bold'); fig.tight_layout(rect=[0,0,.67,.92]); fig.savefig(out/'convergence_time_vs_experiment.png',dpi=300); plt.close(fig)


def save_avg_eps(eps_results,out):
    fig,ax=plt.subplots(figsize=(10,5.4)); eps_values=sorted({float(r['epsilon']) for r in eps_results})
    for i,exp in enumerate(EXPERIMENTS):
        rows=sorted([r for r in eps_results if r['experiment']==exp],key=lambda r:r['epsilon']); xs=[r['epsilon'] for r in rows]; ys=[r['average_test_reward'] for r in rows]
        ax.plot(xs,ys,marker='o',linewidth=1.7,color=EXPERIMENT_COLORS[exp],label=EXPERIMENT_LABELS[exp]); best=max(rows,key=lambda r:(r['average_test_reward'],r['best_score']))
        ax.scatter([best['epsilon']],[best['average_test_reward']],marker='*',s=260,color=EXPERIMENT_COLORS[exp],edgecolor='black',zorder=8); text_panel(fig,.70,.82-i*.14,f"Best {EXPERIMENT_LABELS[exp]}\nε={best['epsilon']:.3g}\nAvg={best['average_test_reward']:.1f}\nS={best['best_score']:.1f}",EXPERIMENT_COLORS[exp])
    ax.set_xticks(eps_values); ax.set_xticklabels([f'{v:.2f}'.rstrip('0').rstrip('.') if v else '0' for v in eps_values],rotation=45,ha='right',fontsize=7); ax.set_xlabel('Epsilon'); ax.set_ylabel('Average test reward'); ax.set_title('CARLA 0.9.14 Average Reward vs Epsilon',pad=14); ax.grid(True,alpha=.35); ax.legend(loc='lower left',bbox_to_anchor=(1.03,.02),fontsize=8,frameon=True); fig.text(.70,.92,'Best epsilon configurations',ha='left',va='top',fontsize=9,weight='bold'); fig.tight_layout(rect=[0,0,.67,.92]); fig.savefig(out/'average_reward_vs_epsilon.png',dpi=300); plt.close(fig)


def parse_float_list(s):
    vals=[float(x.strip()) for x in s.split(',') if x.strip()]
    if not vals: raise ValueError('List cannot be empty')
    return vals


def write_csvs(results,test_rows,out):
    pd.DataFrame([{k:v for k,v in r.items() if k!='test_rewards'} for r in results]).to_csv(out/'all_experiments_learning_rate_summary.csv',index=False)
    pd.DataFrame(test_rows).to_csv(out/'all_experiments_test_episode_rewards.csv',index=False)


def run_one_sweep(args,out:Path, lr_mults:Sequence[float]):
    raw_lr=args.raw_learning_rate if args.raw_learning_rate>0 else 1.0/max(args.train_episodes*args.max_episode_steps,1)
    results=[]; all_test=[]
    for lr_mult in lr_mults:
        lr=raw_lr*lr_mult; print(f"\n=== LR multiplier {lr_mult:g}; final LR={lr:.8g} ===")
        noisy,tr_noisy=train_one('noisy_count',lr,lr_mult,args,out)
        rnd,tr_rnd=train_one('rnd_count',lr,lr_mult,args,out)
        experts=[('noisy_count',noisy),('rnd_count',rnd)]
        ens,tr_ens=train_one('ensemble_own_noisy_rnd_count',lr,lr_mult,args,out,experts=experts)
        trained={'noisy_count':noisy,'rnd_count':rnd,'ensemble_own_noisy_rnd_count':ens}; train={'noisy_count':tr_noisy,'rnd_count':tr_rnd,'ensemble_own_noisy_rnd_count':tr_ens}
        for exp in EXPERIMENTS:
            rows=test_one(exp,trained[exp],args,lr,lr_mult,experts=experts if exp.startswith('ensemble') else None); all_test.extend(rows); results.append(summarize(exp,lr_mult,lr,train[exp],rows,args))
    return results, all_test, raw_lr


def run(args):
    set_seed(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    results,all_test,raw_lr=run_one_sweep(args,out,parse_float_list(args.lr_multipliers))
    write_csvs(results,all_test,out); save_avg_lr(results,out,args); save_best_lr(results,out); save_conv(results,out)
    if args.run_epsilon_sweep:
        best=max(results,key=lambda r:(r['average_test_reward'],r['best_score']))
        fixed=args.epsilon_sweep_lr_multiplier if args.epsilon_sweep_lr_multiplier>0 else float(best['lr_multiplier'])
        eps_results=[]
        for eps in parse_float_list(args.epsilon_values):
            local=argparse.Namespace(**vars(args)); local.epsilon=float(eps); tmp=out/f'epsilon_{eps:g}'; tmp.mkdir(parents=True,exist_ok=True); local.output_dir=str(tmp); local.run_epsilon_sweep=False
            local_results,_,_=run_one_sweep(local,tmp,[fixed])
            for r in local_results: r['epsilon']=float(eps); eps_results.append(r)
        pd.DataFrame([{k:v for k,v in r.items() if k!='test_rewards'} for r in eps_results]).to_csv(out/'epsilon_sweep_average_reward_summary.csv',index=False); save_avg_eps(eps_results,out)
    else:
        pd.DataFrame(columns=['experiment','epsilon','average_test_reward']).to_csv(out/'epsilon_sweep_average_reward_summary.csv',index=False)
    print(f"\nSaved CARLA 0.9.14 noisy/RND/count outputs to: {out}")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=2000); p.add_argument('--timeout-seconds',type=float,default=10.0)
    p.add_argument('--reward-mode',default='ontology_combined'); p.add_argument('--target-speed',type=float,default=30.0); p.add_argument('--max-episode-steps',type=int,default=500)
    p.add_argument('--train-episodes',type=int,default=20); p.add_argument('--test-episodes',type=int,default=5); p.add_argument('--epsilon',type=float,default=0.2)
    p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--raw-learning-rate',type=float,default=0.0); p.add_argument('--lr-multipliers',default='1,1.25,0.25,0.5,0.75,1.5,1.75,2,2.5,3,4,5')
    p.add_argument('--inference-margin',type=float,default=0.01); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--replay-capacity',type=int,default=50000); p.add_argument('--target-update-interval',type=int,default=1000)
    p.add_argument('--rnd-beta',type=float,default=0.01); p.add_argument('--count-beta',type=float,default=0.05); p.add_argument('--count-state-bin-size',type=float,default=1.0)
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=42); p.add_argument('--convergence-threshold-fraction',type=float,default=0.95); p.add_argument('--convergence-window',type=int,default=10)
    p.add_argument('--run-epsilon-sweep',action='store_true'); p.add_argument('--epsilon-values',default='0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.30,0.40,0.50'); p.add_argument('--epsilon-sweep-lr-multiplier',type=float,default=0.0)
    p.add_argument('--output-dir',default='results/graph_set_1_noisy_rnd_count')
    return p.parse_args()

if __name__=='__main__': run(parse_args())
