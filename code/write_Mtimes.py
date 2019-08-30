#!/usr/bin/env python
# coding: utf-8

# Takes in .AWD, sleep log if it exists, calendar, and spits out some pretty pictures

# In[1]:


import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, '/data/MoodGroup/actigraphy/gavi/actigraPy') 

import actigraPy.actigraPy as act
import importlib
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


# If you want to run the test data, run this cell instead of the one below

# In[2]:


# data and log directories
sub = sys.argv[1]#subject NUMBER as STRING
sub_long = 'sub-MOA%s'%sub

data_dir = '/data/MoodGroup/actigraphy/KMOA/raw/%s'%sub_long #output directory
out_dir = '/data/MoodGroup/actigraphy/KMOA/derivatives/preproc/%s'%sub_long

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


# ## Subject and directory info

# In[3]:


fn = {}
#.AWD
awd_fn = os.path.join(data_dir, '%s.AWD'%sub_long)
#sleeplog
fn['cal']= os.path.join(data_dir, '%s_calendar_log.xls'%sub_long)
#calendar
fn['log'] = os.path.join(data_dir, '%s_sleeplog.xls'%sub_long)
logs = []

if os.path.isfile(awd_fn):
    for name in fn.keys():
        if os.path.isfile(fn[name]):
            print("%s has %s file at %s"%(sub,name,fn[name]))
            logs.append(fn[name])
        else:
            print("no " + name)
            fn[name] = ''
else:
    print("DO NOT CONTINUE!! THERE IS NO AWD FILE")


# ### Make Mtimes file

# Figure out where to clip awd file

# In[4]:


awd_dat = act.read_AWD(awd_fn)


# ## Now just run these cells in order to extract comments and write Mtimes

# In[5]:


importlib.reload(act)
#Make master mk_idx
mk_idx=act.get_markers(awd_dat)


# In[6]:


importlib.reload(act)
#Read logs
for name in fn.keys():
    if os.path.isfile(fn[name]):
        log_dat,kw_dat= act.read_log(fn[name],awd_dat)
        mk_idx[name]=log_dat['idx']
        


# Make passes their own marker

# In[7]:


mk_idx['cal']


# In[8]:


rm_idx=[]
for idx,block in enumerate(mk_idx['cal']):
    comment = block[2]
    if 'pass' in comment:
        rm_idx.append(idx)
rm_idx.sort(reverse=True)
mk_idx['pass']=[]
for idx in rm_idx:
    block = mk_idx['cal'].pop(idx)
    mk_idx['pass'].append(block)
mk_idx['pass']


# In[9]:


mk_idx['cal']


# Write Mtimes

# In[10]:


importlib.reload(act)
act.write_Mtimes(awd_dat,mk_idx,os.path.join(out_dir,sub_long))


# ## Clip AWD for graphs

# In[11]:


#read AWD file
awd_dat = act.read_AWD(awd_fn)
idx={'start':[0],'end':[len(awd_dat['dt_list'])-1]}
#get the start and stops from each log if they exist
for log in logs:
    print(log)
    _, kw_dat = act.read_log(log,awd_dat)
    
    if 'watch_on' in kw_dat.keys():
        on_date = kw_dat['watch_on'].iloc[0]['OnDate']
        on_time = kw_dat['watch_on'].iloc[0]['OnTime']
        on = datetime(on_date.year,on_date.month,on_date.day,on_time.hour,on_time.minute)
        #check if that time is in dt_list:
        if awd_dat['dt_list'].count(on) > 0:
            on_idx=awd_dat['dt_list'].index(on)
            idx['start'].append(on_idx)

    if 'watch_off' in kw_dat.keys():
        off_date = kw_dat['watch_off'].iloc[0]['OffDate']
        off_time = kw_dat['watch_off'].iloc[0]['OffTime']
        off = datetime(off_date.year,off_date.month,off_date.day,off_time.hour,off_time.minute)
        if awd_dat['dt_list'].count(off) > 0:
            off_idx=awd_dat['dt_list'].index(off)
            idx['end'].append(off_idx)
 
idx


# In[12]:


start = max(idx['start'])
end = min(idx['end'])
print('start = %d, end = %d'%(start,end))


# ## <font color='red'>Make sure the starts and ends make sense before clipping the data</font>
# 

# In[13]:


#put own number in start + end if disagree with above!
lim = [(awd_dat['DateTime'][start],awd_dat['DateTime'][end],"")]
clip_dat = act.clip_dat(lim,awd_dat)


# In[14]:


clip_dat['DateTime'][-1]


# In[15]:


log_dat,kw_dat = act.read_log(os.path.join(out_dir,sub_long+'_Mtimes.csv'),clip_dat)


# In[16]:


mk_idx=log_dat['mks']


# ## Make graphs

# In[17]:


plots = {}
for name in fn.keys():
    if name in mk_idx.keys():
        plots[name]=mk_idx[name]
if 'pass' in mk_idx.keys():
    plots['pass']=mk_idx['pass']


# In[18]:


plots


# In[ ]:


importlib.reload(act)

act.plot_awd(clip_dat,plots,max_act=1000, show=False,fn_pref=os.path.join(out_dir,sub_long+'_M+logs'),plot_type='single',debug=True)


# In[ ]:


importlib.reload(act)

act.plot_awd(clip_dat,plots,max_act=250,show=False,fn_pref=os.path.join(out_dir,sub_long+'_M+logs_zoom'),plot_type='single',debug=True)


# In[ ]:




