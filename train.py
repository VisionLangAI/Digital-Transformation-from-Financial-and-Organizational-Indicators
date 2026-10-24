"""Real, trainable FORL-Net reproduction and evaluation. No fixed or synthetic results."""
import os, json, time, random, argparse, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, log_loss)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = {
 'RD':['rd_expenses','rd_intensity','tech_ratio'],
 'Organization':['manager_ln','exec_edu_level','social_network'],
 'Governance':['top10_share','duality','ind_dir_ratio'],
 'Finance':['roa','lev','growth','tobin_q','cashflow_ratio'],
 'Environment':['fin_support','m2_growth','ip_protection'],
 'Market':['infra_score','hhi_d'],
 'Firm':['past_revenue_ln','firm_age','size_ln','soe']}
BRANCHES = [('Finance','Organization'),('Finance','RD'),('Finance','Governance'),
 ('Finance','Environment'),('Finance','Firm'),('Organization','Governance'),('RD','Environment')]
BRANCH_NAMES = ['Financial–Organizational','Financial–R&D','Financial–Governance',
 'Financial–Economic Environment','Financial–Firm Characteristics',
 'Organizational–Governance','R&D–Economic Environment']
FEATURES = [f for fields in GROUPS.values() for f in fields]
LABELS = ['Low','Medium','High']

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def block(din,dout):return nn.Sequential(nn.Linear(din,dout),nn.GELU(),nn.LayerNorm(dout))

class FORLNet(nn.Module):
 def __init__(self, emb=64, hidden=128, dropout=.25, variant='full', drop_branch=-1):
  super().__init__(); self.variant=variant;self.drop_branch=drop_branch
  self.emb=emb; self.hidden=hidden
  self.projections=nn.Parameter(torch.empty(len(FEATURES),emb));self.bias=nn.Parameter(torch.zeros(len(FEATURES),emb))
  nn.init.xavier_uniform_(self.projections)
  self.cat_enc=nn.ModuleDict({k:block(len(v),emb) for k,v in GROUPS.items()})
  self.within=nn.ModuleDict({k:block(4*emb,emb) for k in GROUPS})
  self.cat_fuse=nn.ModuleDict({k:block(emb,emb) for k in GROUPS})
  self.cross=nn.ModuleList([block(4*emb,emb) for _ in BRANCHES])
  self.rel_score=nn.Sequential(nn.Linear(emb,32),nn.GELU(),nn.Linear(32,1))
  self.enterprise=block((len(GROUPS)+1)*emb,hidden)
  self.anchor=nn.Linear(emb,hidden)
  self.gate=nn.Linear(2*hidden,hidden)
  self.refine=nn.Sequential(nn.Linear(hidden,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,hidden))
  self.refine_norm=nn.LayerNorm(hidden)
  self.head=nn.Linear(hidden,3)
  self.mlp=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,3))
  self.dropout=nn.Dropout(dropout)
 def forward(self,x,return_aux=False):
  if self.variant=='mlp':
   logits=self.mlp(x)
   if return_aux:return logits,{'attention':torch.full((len(x),7),1/7,device=x.device),'gate':torch.zeros(len(x),self.hidden,device=x.device),'relations':None}
   return logits
  e=x.unsqueeze(-1)*self.projections.unsqueeze(0)+self.bias.unsqueeze(0)
  cats={}; cursor=0
  for name,cols in GROUPS.items():
   n=len(cols); vec=e[:,cursor:cursor+n,:];raw=x[:,cursor:cursor+n];cursor+=n
   base=self.cat_enc[name](raw)
   if self.variant=='no_within' or n<2: cats[name]=base;continue
   pairs=[]
   for i in range(n):
    for j in range(i+1,n):
     a,b=vec[:,i,:],vec[:,j,:]
     pairs.append(self.within[name](torch.cat([a,b,a*b,torch.abs(a-b)],-1)))
   cats[name]=self.cat_fuse[name](base+torch.stack(pairs,0).mean(0))
  rel=[]
  for j,(a,b) in enumerate(BRANCHES):
   va,vb=cats[a],cats[b]
   r=self.cross[j](torch.cat([va,vb,va*vb,torch.abs(va-vb)],-1))
   if j==self.drop_branch or self.variant=='no_cross':r=torch.zeros_like(r)
   rel.append(r)
  rel=torch.stack(rel,1)
  active=torch.ones(len(BRANCHES),dtype=torch.bool,device=x.device)
  if self.drop_branch>=0:active[self.drop_branch]=False
  if self.variant=='no_cross':active[:]=False
  if active.any():
   if self.variant in ('equal_weights','no_reliability'):
    w=active.float()/active.float().sum();attention=w[None,:].expand(len(x),-1)
   else:
    scores=self.rel_score(rel).squeeze(-1).masked_fill(~active[None,:],-1e9)
    attention=torch.softmax(scores,dim=1)
   weighted=(rel*attention.unsqueeze(-1)).sum(1)
  else:
   attention=torch.zeros(len(x),len(BRANCHES),device=x.device);weighted=torch.zeros_like(cats['Finance'])
  combined=torch.cat([*(cats[k] for k in GROUPS),weighted],dim=-1)
  h=self.enterprise(combined)
  if self.variant=='no_anchor':
   gate=torch.ones_like(h);guided=h
  else:
   anchor=self.anchor(cats['Finance']);gate=torch.sigmoid(self.gate(torch.cat([h,anchor],-1)))
   guided=gate*h+(1-gate)*anchor
  if self.variant=='no_refine':z=guided
  else:z=self.refine_norm(guided+self.refine(guided))
  logits=self.head(self.dropout(z))
  if return_aux:return logits,{'attention':attention,'gate':gate,'relations':rel}
  return logits

def diversity(rel):
 if rel is None:return torch.tensor(0.,device='cpu')
 norm=nn.functional.normalize(rel,dim=-1)
 gram=norm@norm.transpose(1,2)
 tri=torch.triu_indices(7,7,offset=1,device=rel.device)
 return (gram[:,tri[0],tri[1]]**2).mean()

def effective_weights(y,beta=.999):
 counts=np.bincount(y,minlength=3); effective=(1-np.power(beta,counts))/(1-beta)
 w=1/np.maximum(effective,1e-12);return (w/w.mean()).astype('float32')

def ece_score(y,p,bins=15):
 conf=p.max(1);correct=(p.argmax(1)==y).astype(float); edges=np.linspace(0,1,bins+1)
 rows=[]; total=0
 for i in range(bins):
  mask=(conf>=edges[i]) & ((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
  if mask.any():
   acc=correct[mask].mean();c=conf[mask].mean();total+=mask.mean()*abs(acc-c)
   rows.append({'bin':i,'count':int(mask.sum()),'confidence':float(c),'accuracy':float(acc)})
 return float(total),rows

def scores(y,p):
 pred=p.argmax(1);prec,rec,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],average='macro',zero_division=0)
 onehot=np.eye(3)[y];ece,_=ece_score(y,p)
 out=dict(accuracy=accuracy_score(y,pred),precision=prec,recall=rec,f1=f1,
  mcc=matthews_corrcoef(y,pred),qwk=cohen_kappa_score(y,pred,weights='quadratic'),
  brier=np.mean(np.sum((p-onehot)**2,axis=1)),ece=ece,ordinal_mae=np.mean(abs(pred-y)),
  log_loss=log_loss(y,p,labels=[0,1,2]))
 try:out['roc_auc']=roc_auc_score(onehot,p,average='macro')
 except ValueError:out['roc_auc']=float('nan')
 try:out['pr_auc']=float(np.mean([average_precision_score(onehot[:,k],p[:,k]) for k in range(3)]))
 except ValueError:out['pr_auc']=float('nan')
 return {k:float(v) for k,v in out.items()}

def fit_preprocess(train,val,test):
 imp=SimpleImputer(strategy='median');scaler=StandardScaler()
 tr=imp.fit_transform(train);v=imp.transform(val);te=imp.transform(test)
 return scaler.fit_transform(tr).astype('float32'),scaler.transform(v).astype('float32'),scaler.transform(te).astype('float32')

def loader(x,y,batch,shuffle=False):
 ds=TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.long))
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)

def train_nn(xtr,ytr,xv,yv,config,device,variant='full',drop_branch=-1):
 seed_all(config.seed)
 net=FORLNet(config.emb,config.hidden,config.dropout,variant,drop_branch).to(device)
 opt=torch.optim.AdamW(net.parameters(),lr=config.lr,weight_decay=config.weight_decay)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config.epochs)
 weights=torch.tensor(effective_weights(ytr),device=device)
 crit=nn.CrossEntropyLoss(weight=weights)
 dl=loader(xtr,ytr,config.batch,True)
 xt=torch.tensor(xv,device=device);yt=torch.tensor(yv,device=device)
 best=float('inf');best_state=None;patience=0;history=[]
 start=time.perf_counter()
 for epoch in range(config.epochs):
  net.train();losses=[]
  for xx,yy in dl:
   xx=xx.to(device);yy=yy.to(device);opt.zero_grad()
   logits,aux=net(xx,True)
   loss=crit(logits,yy)
   if aux['relations'] is not None and variant!='no_cross':loss=loss+config.diversity*diversity(aux['relations'])
   loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  net.eval()
  with torch.no_grad():vl=float(crit(net(xt),yt).cpu())
  history.append({'epoch':epoch+1,'train_loss':np.mean(losses),'val_loss':vl})
  if vl<best-1e-6:
   best=vl;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};patience=0
  else:patience+=1
  scheduler.step()
  if patience>=config.patience:break
 net.load_state_dict(best_state);net.eval()
 return net,history,time.perf_counter()-start

def predict_nn(net,x,device,batch=512):
 probs=[];att=[];gates=[]
 with torch.no_grad():
  for offset in range(0,len(x),batch):
   xx=torch.tensor(x[offset:offset+batch],device=device)
   logits,aux=net(xx,True)
   probs.append(torch.softmax(logits,-1).cpu().numpy())
   att.append(aux['attention'].cpu().numpy());gates.append(aux['gate'].mean(1).cpu().numpy())
 return np.concatenate(probs),np.concatenate(att),np.concatenate(gates)

def train_baseline(name,xtr,ytr,xv,yv,xte,config,device):
 """Real baseline implementations; optional published packages are never silently substituted."""
 if name=='LogisticRegression':
  from sklearn.linear_model import LogisticRegression
  m=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=config.seed)
 elif name=='RandomForest':
  from sklearn.ensemble import RandomForestClassifier
  m=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=config.seed)
 elif name=='ExtraTrees':
  from sklearn.ensemble import ExtraTreesClassifier
  m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=config.seed)
 elif name=='XGBoost':
  from xgboost import XGBClassifier
  m=XGBClassifier(n_estimators=400,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.9,random_state=config.seed,objective='multi:softprob',num_class=3,n_jobs=4)
 elif name=='TabNet':
  from pytorch_tabnet.tab_model import TabNetClassifier
  m=TabNetClassifier(n_d=16,n_a=16,n_steps=3,seed=config.seed,device_name=str(device),verbose=0)
  m.fit(xtr,ytr,eval_set=[(xv,yv)],eval_metric=['accuracy'],max_epochs=config.epochs,patience=config.patience,batch_size=config.batch,virtual_batch_size=min(config.batch,32))
  return m.predict_proba(xte)
 elif name=='TabPFN_v2':
  from tabpfn import TabPFNClassifier
  m=TabPFNClassifier(device=str(device))
 else:raise ValueError('Unsupported baseline '+name)
 m.fit(xtr,ytr)
 p=m.predict_proba(xte)
 if p.shape[1]!=3:raise RuntimeError('Missing class in baseline output')
 return p

def savefig(path):plt.tight_layout();plt.savefig(path,dpi=180,bbox_inches='tight');plt.close()

def figures(df,out,history=None,attention=None,gate=None,ablation=None):
 out.mkdir(parents=True,exist_ok=True)
 if df.empty:return
 for name,g in df.groupby('model'):
  y=g.true_label.to_numpy(dtype=int);p=g[['prob_low','prob_medium','prob_high']].to_numpy(); pred=p.argmax(1)
  cm=confusion_matrix(y,pred,labels=[0,1,2]);np.savetxt(out/f'{name}_confusion_counts.csv',cm,fmt='%d',delimiter=',')
  for norm in [False,True]:
   m=cm.astype(float);m=m/np.maximum(m.sum(1,keepdims=True),1) if norm else m
   fig,ax=plt.subplots(figsize=(5.5,4.6));im=ax.imshow(m,cmap='Blues',vmin=0,vmax=1 if norm else None)
   for i in range(3):
    for j in range(3):ax.text(j,i,f'{m[i,j]:.1%}' if norm else str(int(m[i,j])),ha='center',va='center',color='black')
   ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');fig.colorbar(im,ax=ax)
   savefig(out/f'{name}_confusion_{"normalized" if norm else "counts"}.png')
  hot=np.eye(3)[y]
  fig,ax=plt.subplots(figsize=(6,5))
  for k in range(3):
   fpr,tpr,_=roc_curve(hot[:,k],p[:,k]);ax.plot(fpr,tpr,label=LABELS[k])
  ax.plot([0,1],[0,1],ls='--',color='gray');ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend()
  savefig(out/f'{name}_roc.png')
  fig,ax=plt.subplots(figsize=(6,5))
  for k in range(3):
   pr,re,_=precision_recall_curve(hot[:,k],p[:,k]);ax.plot(re,pr,label=LABELS[k])
  ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend();savefig(out/f'{name}_pr.png')
  ece,bins=ece_score(y,p);pd.DataFrame(bins).to_csv(out/f'{name}_calibration_bins.csv',index=False)
  fig,ax=plt.subplots(figsize=(5.5,5));ax.plot([0,1],[0,1],ls='--',color='gray')
  if bins:ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o')
  ax.set_xlabel('Confidence');ax.set_ylabel('Accuracy');savefig(out/f'{name}_calibration.png')
  fig,ax=plt.subplots(figsize=(6,4));ax.hist(p.max(1),bins=np.linspace(1/3,1,25),alpha=.8);ax.set_xlabel('Maximum predicted probability');ax.set_ylabel('Count');savefig(out/f'{name}_confidence.png')
 if history is not None and len(history):
  fig,ax=plt.subplots(figsize=(7,4));ax.plot(history.epoch,history.train_loss,label='Training');ax.plot(history.epoch,history.val_loss,label='Validation');ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.legend();savefig(out/'training_validation_loss.png')
 if attention is not None and len(attention):
  fig,ax=plt.subplots(figsize=(9,4));m=attention.groupby('true_label')[BRANCH_NAMES].mean().reindex([0,1,2]);im=ax.imshow(m.to_numpy(),aspect='auto',cmap='Blues');ax.set_yticks(range(3),LABELS);ax.set_xticks(range(7),BRANCH_NAMES,rotation=45,ha='right');fig.colorbar(im,ax=ax);savefig(out/'relationship_attention_by_class.png')
  fig,ax=plt.subplots(figsize=(8,4));ax.barh(BRANCH_NAMES,attention[BRANCH_NAMES].mean());ax.set_xlabel('Mean attention weight');savefig(out/'relationship_attention_mean.png')
 if gate is not None and len(gate):
  fig,ax=plt.subplots(figsize=(6,4));ax.hist(gate.gate_mean,bins=20);ax.set_xlabel('Mean financial control gate');ax.set_ylabel('Count');savefig(out/'financial_gate_distribution.png')
 if ablation is not None and len(ablation):
  s=ablation.sort_values('f1');fig,ax=plt.subplots(figsize=(9,6));ax.barh(s.model,s.f1);ax.set_xlabel('Macro F1');savefig(out/'ablation_f1.png')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data',default='digital_innovation_dataset_class.csv');ap.add_argument('--out',default='results')
 ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=12)
 ap.add_argument('--batch',type=int,default=32);ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--emb',type=int,default=64);ap.add_argument('--hidden',type=int,default=128)
 ap.add_argument('--dropout',type=float,default=.25);ap.add_argument('--lr',type=float,default=.001)
 ap.add_argument('--weight-decay',type=float,default=.0001);ap.add_argument('--diversity',type=float,default=.05)
 ap.add_argument('--folds',type=int,default=5);ap.add_argument('--variants',action='store_true')
 ap.add_argument('--branches',action='store_true');ap.add_argument('--baselines',default='',help='Comma separated: LogisticRegression,RandomForest,ExtraTrees,XGBoost,TabNet,TabPFN_v2')
 ap.add_argument('--skip-missing-baselines',action='store_true');ap.add_argument('--device',default='auto')
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 device=torch.device('cuda' if (a.device=='auto' and torch.cuda.is_available()) else ('cpu' if a.device=='auto' else a.device))
 df=pd.read_csv(a.data);missing=set(FEATURES+['digitaltrans_class'])-set(df.columns)
 if missing:raise ValueError(f'Missing required columns: {missing}')
 x=df[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float);y=df.digitaltrans_class.to_numpy(dtype=int)
 if not set(np.unique(y)).issubset({0,1,2}):raise ValueError('Target must contain 0,1,2')
 if len(df)!=len(np.unique(df.index)):raise ValueError('Duplicate row indices')
 seed_all(a.seed);split=StratifiedKFold(n_splits=a.folds,shuffle=True,random_state=a.seed)
 variants=['full']
 if a.variants:variants+=['mlp','no_within','no_cross','no_reliability','no_anchor','no_refine','equal_weights']
 if a.branches:variants += [f'branch_{j}' for j in range(7)]
 baselines=[n.strip() for n in a.baselines.split(',') if n.strip()]
 rows=[];foldrows=[];historyrows=[];auxrows=[];gaterows=[];runtime=[];missing_baselines=[]
 for fold,(development,test) in enumerate(split.split(x,y),1):
  tr,va=train_test_split(development,test_size=.2,stratify=y[development],random_state=a.seed+fold)
  xt,xv,xs=fit_preprocess(x[tr],x[va],x[test]);yt,yv,ys=y[tr],y[va],y[test]
  print(f'Fold {fold}/{a.folds}: train={len(tr)}, val={len(va)}, test={len(test)}',flush=True)
  for v in variants+baselines:
   start=time.perf_counter();att=gate=None
   try:
    if v in baselines:p=train_baseline(v,xt,yt,xv,yv,xs,a,device);params=np.nan;hist=[];elapsed=time.perf_counter()-start
    else:
     branch=int(v.split('_')[1]) if v.startswith('branch_') else -1
     kind='full' if branch>=0 else v
     net,hist,elapsed=train_nn(xt,yt,xv,yv,a,device,kind,branch)
     p,att,gate=predict_nn(net,xs,device)
     params=sum(z.numel() for z in net.parameters() if z.requires_grad)
     if v=='full':
      torch.save({'state_dict':net.state_dict(),'features':FEATURES,'groups':GROUPS,'config':vars(a)},out/f'FORLNet_fold{fold}.pt')
      for item in hist:historyrows.append({'fold':fold,**item})
    met=scores(ys,p);foldrows.append({'model':v,'fold':fold,'test_size':len(test),'train_size':len(tr),'val_size':len(va),**met})
    runtime.append({'model':v,'fold':fold,'train_seconds':elapsed,'parameters':params})
    for idx,row_id in enumerate(test):
     rows.append({'model':v,'fold':fold,'row_id':int(row_id),'true_label':int(ys[idx]),'predicted_label':int(p[idx].argmax()),
       'prob_low':float(p[idx,0]),'prob_medium':float(p[idx,1]),'prob_high':float(p[idx,2])})
     if v=='full':
      auxrows.append({'fold':fold,'row_id':int(row_id),'true_label':int(ys[idx]),**{BRANCH_NAMES[j]:float(att[idx,j]) for j in range(7)}})
      gaterows.append({'fold':fold,'row_id':int(row_id),'gate_mean':float(gate[idx])})
    print(f'  {v}: acc={met["accuracy"]:.4f} f1={met["f1"]:.4f} time={elapsed:.1f}s',flush=True)
   except (ImportError,ModuleNotFoundError) as exc:
    if a.skip_missing_baselines and v in baselines:
     warnings.warn(f'Skipping unavailable real baseline {v}: {exc}');missing_baselines.append({'model':v,'error':str(exc)});continue
    raise
  pd.DataFrame(rows).to_csv(out/'predictions_oof.csv',index=False)
  pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False)
 pred=pd.DataFrame(rows);folds=pd.DataFrame(foldrows);runtimes=pd.DataFrame(runtime)
 summary=[];classwise=[];cms=[];calibration=[]
 for name,g in pred.groupby('model'):
  ys=g.true_label.to_numpy(dtype=int);p=g[['prob_low','prob_medium','prob_high']].to_numpy()
  met=scores(ys,p);byfold=folds[folds.model==name]
  summary.append({'model':name,**met,**{f'{k}_fold_sd':float(byfold[k].std(ddof=1)) for k in met}})
  pp,rr,ff,supp=precision_recall_fscore_support(ys,p.argmax(1),labels=[0,1,2],zero_division=0)
  for j in range(3):
   oh=(ys==j).astype(int)
   classwise.append({'model':name,'class':LABELS[j],'support':int(supp[j]),'precision':pp[j],'recall':rr[j],'f1':ff[j],
    'roc_auc':roc_auc_score(oh,p[:,j]),'pr_auc':average_precision_score(oh,p[:,j])})
  cm=confusion_matrix(ys,p.argmax(1),labels=[0,1,2])
  for i in range(3):
   for j in range(3):cms.append({'model':name,'actual':LABELS[i],'predicted':LABELS[j],'count':int(cm[i,j])})
  _,bins=ece_score(ys,p)
  for item in bins:calibration.append({'model':name,**item})
 summ=pd.DataFrame(summary);summ.to_csv(out/'Table5_comparison_pooled_OOF.csv',index=False)
 folds.to_csv(out/'Table6_fold_metrics.csv',index=False)
 pd.DataFrame(classwise).to_csv(out/'Table7_classwise.csv',index=False)
 pd.DataFrame(cms).to_csv(out/'confusion_matrix_cells.csv',index=False)
 pd.DataFrame(calibration).to_csv(out/'calibration_bins.csv',index=False)
 runtimes.to_csv(out/'efficiency_training.csv',index=False)
 pd.DataFrame(historyrows).to_csv(out/'training_history.csv',index=False)
 pd.DataFrame(auxrows).to_csv(out/'relationship_attention_OOF.csv',index=False)
 pd.DataFrame(gaterows).to_csv(out/'financial_gate_OOF.csv',index=False)
 pd.DataFrame(missing_baselines).to_csv(out/'unavailable_baselines.csv',index=False)
 if a.variants or a.branches:
  ab=summ[summ.model.isin(variants)].copy();ab.to_csv(out/'ablation_metrics.csv',index=False)
  full=ab[ab.model=='full'].iloc[0]
  ab['accuracy_drop_from_full']=full.accuracy-ab.accuracy;ab['f1_drop_from_full']=full.f1-ab.f1
  ab.to_csv(out/'ablation_differences.csv',index=False)
  if a.branches:
   br=ab[ab.model.str.startswith('branch_')].copy()
   br['branch_name']=br.model.map({f'branch_{i}':n for i,n in enumerate(BRANCH_NAMES)})
   br.to_csv(out/'relationship_branch_removal.csv',index=False)
 if len(baselines):
  full=folds[folds.model=='full'].set_index('fold')
  comp=[]
  for name in baselines:
   other=folds[folds.model==name].set_index('fold')
   if len(other)!=a.folds:continue
   delta=full.f1-other.f1
   comp.append({'comparison':f'FORL-Net vs {name}','mean_fold_f1_difference':delta.mean(),
                'sd_fold_difference':delta.std(ddof=1)})
  pd.DataFrame(comp).to_csv(out/'descriptive_model_comparisons.csv',index=False)
 figures(pred,out/'figures',pd.DataFrame(historyrows),pd.DataFrame(auxrows),pd.DataFrame(gaterows),
  summ[summ.model.isin(variants)] if (a.variants or a.branches) else None)
 with open(out/'run_manifest.json','w') as f:json.dump({'configuration':vars(a),'device':str(device),'torch':torch.__version__,
  'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),
  'dataset_rows':len(df),'class_counts':np.bincount(y,minlength=3).tolist(),
  'predictors':FEATURES,'branches':BRANCH_NAMES,'reported_accuracy_is_not_enforced':True},f,indent=2)
 print('\nComplete. Results saved to:',out.resolve())
 print(summ[['model','accuracy','precision','recall','f1','mcc','qwk','roc_auc','pr_auc','brier','ece','ordinal_mae']].to_string(index=False))

if __name__=='__main__':main()
